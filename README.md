<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

# DwarfStar for Strix Halo

> **This is a fork.** It is a downstream fork of
> [antirez/ds4](https://github.com/antirez/ds4) (**DwarfStar**), a small native
> inference engine for a few large open-weight models. This fork tunes it for
> **AMD Strix Halo** machines (Ryzen AI Max+ 395, Radeon 8060S `gfx1151`,
> 128 GB unified memory), such as the Framework Desktop, on one machine or
> two. For Metal, CUDA and the project as a whole, see upstream.

The fork follows upstream and merges it regularly; its own work is on the
ROCm backend, memory management on unified memory, pipelines across Strix
Halo machines, and the HTTP server. Changes that are not Strix Halo specific
are meant to go upstream.

## What this fork adds

**Speed on `gfx1151`.** Retuned ROCm kernels for DeepSeek V4 Flash prefill
(WMMA tiles for the IQ2_XXS, Q2_K and MXFP4 routed experts), decode, long
context attention and the vision encoder.

**Memory on a 128 GB unified memory box.**
- Context memory is committed on demand: `--ctx 1000000` costs a short
  conversation nothing.
- When memory runs short, idle server sessions go to the disk KV cache first,
  then the Q8->F16 weight cache shrinks, then cold routed experts are evicted
  and read back from the GGUF on demand, instead of failing the request.
- Sessions that take turns on the GPU share one indexer scratch buffer.

**Two machines as one.** Pipeline parallelism across two Strix Halo boxes
over ordinary Ethernet, to run the 145 GiB MXFP4 model resident: concurrent
prefills overlap across the stages, vision and DSpark work in pipeline mode,
a restarted worker is waited for and replayed into, and disk KV checkpoints
and live KV rewind work across stages.

**Server.** Disk KV cache for conversations with images, live KV rewind
instead of prefilling from zero, requests that wait for a busy session
instead of failing with 503, as many images per request as fit the context,
Prometheus metrics at `/metrics`, and `--reasoning-effort`.

**A Docker image** with every binary, built from `main` by CI and published
as `ghcr.io/felix920506/ds4`, plus ready-to-use compose files.

## Performance

DeepSeek V4 Flash, text path, measured with `ds4-bench` on 128 GB
Ryzen AI Max+ 395 machines in the published container:

| Setup | Model | Prefill | Decode |
| --- | --- | ---: | ---: |
| One machine | 0731 / Vision Exp Q2 | 617 tok/s at 2K-4K, 553 at 32K | 17.6 tok/s at 2K, 15.9 at 32K |
| Two machines, 2.5 GbE | Vision Exp MXFP4 | 800-850 tok/s to 98K | 14.6-14.9 tok/s at 2K, 13.2 at 100K, 12.5 at 400K |

Method and details: [single-node decode](speed-bench/gfx1151-decode.md),
[single-node memory and prefill](docs/STRIX_HALO.md#elastic-expert-weights),
[pipeline prefill](speed-bench/gfx1151-pipeline-prefill.md) and
[pipeline decode](speed-bench/gfx1151-pipeline-decode.md). On a pipeline,
decode right after a long prefill is slower for about a minute while the
SoC cools.

## Quick start with Docker

You need a 128 GB Strix Halo machine running Linux with the `amdgpu` driver
(`/dev/kfd` and `/dev/dri` present), Docker with Compose, and most of the
memory made visible to the GPU (see
[GPU-visible memory](docs/STRIX_HALO.md#gpu-visible-memory)). No ROCm
installation on the host is needed: it is in the image.

Clone the fork, which brings the model downloader and the compose files:

```sh
git clone https://github.com/felix920506/ds4.git
cd ds4
```

Pick one of the two recommended setups.

### One machine

DeepSeek V4 Flash Vision Experimental Q2 (about 81 GiB) with its vision
encoder, fully resident:

```sh
./download_model.sh ds4f-vision-q2     # model and encoder into gguf/
cd docker/single-node
cp .env.example .env
docker compose up -d
docker compose logs -f                 # loading takes a few minutes
```

### Two machines

DeepSeek V4 Flash Vision Experimental MXFP4 (about 145 GiB) with its vision
encoder, split in a pipeline: the coordinator runs layers 0-20 and the API,
the worker layers 21 to the output. Connect the machines with a direct
or switched link of 2.5 GbE or faster. On **both** machines:

```sh
./download_model.sh ds4f-vision-mxfp4
cd docker/two-node
cp .env.example .env
# edit .env: COMPOSE_PROFILES=coordinator on one machine, worker on the other,
#            COORDINATOR_IP=<coordinator's address on the link> on both
docker compose up -d
```

### Talk to it

The API listens on `127.0.0.1:8000` of the machine running the server or
coordinator (set `DS4_BIND=0.0.0.0` in `.env` to expose it; it has no
authentication):

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello!"}]}'
```

It speaks the OpenAI Chat Completions and Responses APIs and the Anthropic
Messages API. To use it from Pi, OpenCode, Codex CLI or Claude Code, see the
[client setup guide](docs/CLIENTS.md).

[Docker guide](docs/DOCKER.md): what the compose files set and why, image
tags, running the CLI, agent and benchmarks in the container, and upgrading.

## Building from source

On the host you need ROCm with HIP, hipBLAS, hipBLASLt, rocBLAS, rocWMMA and
hipCUB development files; [the Strix Halo guide](docs/STRIX_HALO.md) lists
the packages. Then:

```sh
make strix-halo
./download_model.sh ds4f-vision-q2
./ds4 -m gguf/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf \
  --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

To build the image yourself: `docker build -t ds4:rocm-gfx1151 .` from the
repository root, then set `DS4_IMAGE=ds4:rocm-gfx1151` in `.env`.

## Using DwarfStar

The image and the build ship the same tools:

| Binary | Use |
| --- | --- |
| `ds4-server` | OpenAI / Anthropic compatible HTTP server ([serving](docs/SERVER.md)) |
| `ds4` | Interactive CLI and one-shot prompts (`-p "..."`); also the pipeline worker |
| `ds4-agent` | Native coding agent that runs the model in-process |
| `ds4-bench` | Prefill and decode benchmarks ([performance](docs/PERFORMANCE.md)) |
| `ds4-eval` | Capability regression tests against a real GGUF |

Run any of them with `--help`. A few things worth knowing:

- Thinking is on by default. `--nothink` / `/nothink` turns it off;
  `--reasoning-effort max` on the server needs `--ctx` of at least 393216.
- Images: pass the matching encoder with `--vision FILE`, then use
  `/read image.png` in the CLI, `view_image` in the agent, or image content in
  API requests. PNG and JPEG decode in-process; other formats go through
  ImageMagick, which the image includes.
- The interactive CLI keeps a multi-turn conversation: `/help`, `/read FILE`,
  `/ctx N`, `/quit`. Ctrl+C interrupts generation.
- `ds4-agent` stores sessions in `~/.ds4/kvcache`: `/save`, `/list`,
  `/switch <sha>`, `/del <sha>`, `/strip <sha>`. Saved sessions and traces
  may contain private information.
- Sampling defaults are temperature 1, top-p 1, min-p 0.05; `--temp 0` is
  greedy.

## Models on Strix Halo

| Download target | Size | Fits | Notes |
| --- | ---: | --- | --- |
| `ds4f-vision-q2` | 81 GiB + encoder | One machine | **Recommended**; text and images |
| `ds4f-q2` | 81 GiB | One machine | Flash 0731, text only |
| `ds4f-vision-mxfp4` | 145 GiB + encoder | Two machines | **Recommended** for two; DeepSeek's native MXFP4 experts |
| `ds4f-mxfp4`, `ds4f-q4` | 153-156 GiB | Two machines | Flash 0731, text only |
| `ds41f-q2` | 341 GiB on disk | One machine, SSD streaming | DeepSeek V4.1 Flash; text only, slow large prefills |
| `glm53-q2` | 90 GiB | One machine, SSD streaming | GLM 5.3 Flash; images with `glm53-vision` |

Run `./download_model.sh --help` for every target. Downloads go to `gguf/`
(or `DS4_GGUF_DIR`) and resume when repeated. Commands for each model are in
the [Strix Halo guide](docs/STRIX_HALO.md); all models are described in
[models and vision](docs/MODELS.md).

## Guides

- [Strix Halo](docs/STRIX_HALO.md): host setup, GPU-visible memory, memory
  management, and every model on this hardware.
- [Docker](docs/DOCKER.md): the image and the compose files.
- [Inference across machines](docs/DISTRIBUTED.md): pipelines, layer splits,
  recovery and tuning.
- [Serving](docs/SERVER.md): APIs, images, multiple sessions, disk KV cache,
  live KV rewind, and metrics.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, Claude Code.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark and MTP.
- [Performance](docs/PERFORMANCE.md) and [speed-bench](speed-bench/README.md):
  benchmark commands and recorded results.
- [Testing and development](docs/TESTING.md): regression tests and tools.

### Other hardware

The Metal (Apple Silicon) and CUDA (DGX Spark, multi-GPU) backends are
inherited from upstream and still build from this tree, but this fork does
not test them. Use [upstream](https://github.com/antirez/ds4) for those, or
read the inherited guides: [Metal](docs/METAL.md), [DGX Spark](docs/DGX_SPARK.md),
[CUDA multi-GPU](docs/CUDA_MULTI_GPU.md), [SSD streaming](docs/SSD_STREAMING.md),
[Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md).

## Status and contributing

Like upstream, this is fast-moving beta software. Report problems seen on
Strix Halo or with the image and compose files to
[this fork](https://github.com/felix920506/ds4/issues), and problems on other
hardware upstream. Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a
pull request: correctness is checked before speed, and speed changes come with
measurements.

## AI disclosure

DwarfStar, upstream and in this fork, is developed with **strong assistance
from AI coding agents**, with humans leading the ideas, testing and debugging.
If you are not happy with AI-developed code, this software is not for you.

Upstream's suggestion applies here too: if you have a coding agent, use it to
explore the code and adapt it to your setup. This fork is itself an example of
that: hardware-specific work on top of a project that is deliberately narrow.

## Acknowledgements

DwarfStar is created and maintained by Salvatore Sanfilippo
([antirez](https://github.com/antirez)) and its contributors; this fork builds
on their work.

`ds4.c` does not link against GGML, but it **exists thanks to the path opened
by the llama.cpp project and the kernels, quantization formats, GGUF
ecosystem, and hard-won engineering knowledge developed there**. We are
thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Some source-level pieces are retained or adapted here
under the MIT license: GGUF quant layouts and tables, CPU quant/dot logic,
and certain kernels. For this reason, and out of gratitude, the GGML authors'
copyright notice is kept in the [`LICENSE`](LICENSE) file.

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino.
