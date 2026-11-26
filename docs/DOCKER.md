# Docker on Strix Halo

[README](../README.md) | [Strix Halo](STRIX_HALO.md) | [Inference across machines](DISTRIBUTED.md)

The image `ghcr.io/felix920506/ds4` holds all five binaries in `/opt/ds4`,
built for `gfx1151` on ROCm 7.14. The entrypoint is `ds4-server`. The host
needs only the `amdgpu` kernel driver, Docker and Compose; it does not need
ROCm.

## Image tags

| Tag | Contents |
| --- | --- |
| `dev-rocm` | Latest `main` |
| `dev-rocm-<sha>` | One `main` commit, by short SHA |
| `rocm-<version>` | A release tag |

CI builds the [Dockerfile](../Dockerfile) on every push to `main`. A machine
running two stages of a pipeline must run the same build on both, so pin
`DS4_IMAGE` to a `dev-rocm-<sha>` tag there, or pull on both machines at the
same time.

## Before you start

- **GPU-visible memory.** The models need most of the 128 GB as GPU memory.
  Set the kernel's GTT/TTM limits as in
  [GPU-visible memory](STRIX_HALO.md#gpu-visible-memory) first. The container
  cannot change them.
- **Models.** Download with `./download_model.sh TARGET` from a checkout of
  this repository. Files go to `gguf/`, which the compose files mount
  read-only at `/models`. To keep them elsewhere, download with
  `DS4_GGUF_DIR=/path` and set `DS4_MODELS=/path` in `.env`. Use a local SSD:
  loading reads the whole model, and evicted experts are read back from it.
- **One ds4 process per machine.** Two model processes on one unified memory
  machine can exhaust memory badly enough to panic the kernel. ds4 refuses to
  start a second instance through the lock file `/tmp/ds4-lock/ds4.lock`. The
  compose files mount the host's `/tmp/ds4-lock` at the same path, so a
  compose container and a natively started `ds4`, `ds4-server` or `ds4-bench`
  refuse each other. A container started with plain `docker run` has its own
  `/tmp` and is only covered when it also gets
  `-v /tmp/ds4-lock:/tmp/ds4-lock`. Setting `DS4_LOCK_FILE` moves the lock
  elsewhere and takes that process out of this sharing. Stop the server before
  running a benchmark or the CLI on the same machine.

## One machine

[`docker/single-node/compose.yaml`](../docker/single-node/compose.yaml)
serves DeepSeek V4 Flash Vision Experimental Q2 with its vision encoder:

```sh
./download_model.sh ds4f-vision-q2
cd docker/single-node
cp .env.example .env
docker compose up -d
```

What it sets, all adjustable in `.env`:

| Setting | Default | Why |
| --- | --- | --- |
| Model | Vision Experimental Q2 + encoder | 81 GiB, resident, text and images |
| `--ctx` | `1000000` | A reservation: context memory is committed as a conversation grows, and given back under pressure |
| `--kv-disk-dir` | volume `ds4-kv` | Resumed and edited conversations skip most of their prefill |
| `--kv-disk-space-mb` | `32768` | A 1M-token checkpoint is several GiB |
| Port | `127.0.0.1:8000` | The API has no authentication; set `DS4_BIND=0.0.0.0` deliberately |

To serve the text-only 0731 checkpoint instead, download `ds4f-q2`, set
`DS4_MODEL` to its file name, and delete the two `--vision` lines from the
compose file: that encoder belongs to Vision Experimental only.

## Two machines

[`docker/two-node/compose.yaml`](../docker/two-node/compose.yaml) runs
DeepSeek V4 Flash Vision Experimental MXFP4 as a two-stage pipeline. The same
file serves both machines; `COMPOSE_PROFILES` in `.env` selects the stage.

```sh
# On both machines.
./download_model.sh ds4f-vision-mxfp4
cd docker/two-node
cp .env.example .env
```

Edit `.env`:

- `COMPOSE_PROFILES=coordinator` on one machine, `worker` on the other.
- `COORDINATOR_IP` on both: the coordinator's address on the link between
  them. The worker connects to it on port `DS4_DIST_PORT` (9911).
- Leave every other value identical on the two machines.

Then `docker compose up -d` on both, in either order: the worker keeps
retrying until the coordinator listens. The API is on the coordinator.

What it sets:

| Setting | Default | Why |
| --- | --- | --- |
| Split | coordinator `0:20`, worker `21:output` | About half of the 145 GiB each, the split the measurements use |
| Network | host | The stages exchange 128 MiB of activations per 2048-token chunk |
| `--prefill-chunk` | `2048` on both | 11-13% faster than 4096 for growing conversations, and 2 GiB more context room per machine ([measurements](../speed-bench/gfx1151-pipeline-prefill.md#chunk-size)) |
| `--ctx` | `1000000` on both | Reservation, as on one machine |
| `--kv-disk-dir` | coordinator volume `ds4-kv` | Checkpoints hold one shard per stage |

`--ctx` and `--prefill-chunk` must be identical on both stages. Otherwise
requests still run, but disk KV checkpoints are skipped with
`distributed KV shards use different layouts`.

The link between the stages has no authentication or encryption: keep it on
a trusted network, or a direct cable. 2.5 GbE is enough for the throughput in
the [README](../README.md#performance); the wire is a small part of each step.

If the worker restarts, the coordinator holds requests for up to 120 seconds
while it comes back, then replays the interrupted conversations into it. For everything
else about pipelines, see [inference across machines](DISTRIBUTED.md).

## Operating

```sh
docker compose logs -f          # load progress, then requests
docker compose ps               # health: answers /v1/models once loaded
docker compose down             # stop; the KV cache volume is kept
docker compose pull && docker compose up -d   # upgrade
```

Loading reads the whole model from disk: about a minute from a fast NVMe
drive, longer from a slow one or on a pipeline. `docker compose down -v`
also deletes the disk KV cache volume.

Prometheus metrics are at `/metrics` on the API port; see
[serving](SERVER.md#metrics).

### The other tools in the container

Stop the server first (one ds4 process per machine), then run any binary
through the same service definition, which keeps the devices, models and
lock mounted:

```sh
docker compose down
docker compose run --rm --entrypoint /opt/ds4/ds4 ds4-server \
  -m /models/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf \
  --vision /models/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

The same pattern runs `/opt/ds4/ds4-agent`, `/opt/ds4/ds4-bench` and
`/opt/ds4/ds4-eval`. The agent stores sessions under `$HOME/.ds4`; add a
volume for it if they should survive the container. For a benchmark, see
[performance](PERFORMANCE.md) and the
[Strix Halo baseline](../speed-bench/strix_halo_q2_baseline.md) command.

## Building the image

```sh
docker build -t ds4:rocm-gfx1151 .
```

from the repository root, then set `DS4_IMAGE=ds4:rocm-gfx1151` in `.env`.
The build compiles with `-march=native` by default, which suits a build on
the machine that runs it; CI passes `--build-arg NATIVE_CPU_FLAG=-march=znver4`.
`.dockerignore` keeps `gguf/` out of the build context.
