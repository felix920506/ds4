# AMD Strix Halo

[README](../README.md) | [Quick start](../README.md#quick-start-with-docker) | [Docker](DOCKER.md)

The reference system is a 128 GB Strix Halo with Radeon 8060S (`gfx1151`),
such as the Framework Desktop: one machine, or two connected by 2.5 GbE for
a pipeline. The ROCm build uses the standard binary names and selects the
ROCm backend by default.

## Prerequisites

Whichever way you run DwarfStar, first make enough memory visible to the
GPU (next section).

The simplest setup is the published image and the compose files for one or
two machines: see [Docker](DOCKER.md). The host then needs only the `amdgpu`
driver and Docker. A community
[ROCm toolbox](https://github.com/kyuz0/strix-halo-ds4-toolbox/blob/main/toolboxes/Dockerfile.rocm-10.0),
which can be managed with
[AI Toolbox Cockpit](https://github.com/kyuz0/ai-toolbox-cockpit), is another
container option.

For a native Ubuntu build you need HIP, hipBLAS, hipBLASLt, rocBLAS, rocWMMA,
and hipCUB development files. The Ubuntu 26.04 setup used these packages:

```sh
sudo apt-get update
sudo apt-get install -y hipcc rocminfo rocm-smi \
  libamdhip64-dev libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev
sudo usermod -aG render,video "$USER"
```

Log out and back in after changing groups. `rocminfo` must report `gfx1151`
and be able to open `/dev/kfd` before DwarfStar can run.

Some packaged rocWMMA headers omit `rocwmma/internal/`. If compilation fails
there, install the complete headers matching your ROCm installation, or use
the container. Do not mix header versions as a general workaround.

## GPU-visible memory

Check the memory pool reported by `rocminfo`. Some 128 GB configurations expose
only about 62 GB to the GPU, which is insufficient for resident Flash Q2 plus
runtime buffers. Firmware and kernel GTT/TTM settings control this limit.

The native reference setup used these memory parameters:

```text
amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

They are a system-specific starting point, not an allocation budget for
DwarfStar. Preserve existing boot options and consult your kernel's settings
before changing them. Keep RAM available for the OS even when the GPU can
address most of it. Do not disable the IOMMU merely to copy another host's
configuration; doing so changes device isolation.

## Build and run Flash

```sh
make strix-halo
./download_model.sh ds4f-vision-q2
./ds4 -m gguf/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf \
  --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

`make rocm` is an alias. On one machine use a Q2 model: Vision Experimental
(`ds4f-vision-q2`, text and images) or 0731 (`ds4f-q2`, text only), both about
81 GiB. The Q4 and MXFP4 models (145 to 156 GiB) need two machines. Flash's
ROCm resident and pipeline paths should not be confused with the GLM
SSD-streaming path.

### Two machines

Split the MXFP4 model between two machines, coordinator on layers `0:20` and
worker on `21:output`, and start both stages with the same `--ctx` and
`--prefill-chunk 2048`:

```sh
./download_model.sh ds4f-vision-mxfp4      # on both machines
M=gguf/DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf
V=gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf

# Machine A, coordinator and HTTP API.
./ds4-server -m $M --vision $V --role coordinator --layers 0:20 \
  --listen 192.168.1.10 9911 --ctx 1000000 --prefill-chunk 2048

# Machine B, worker.
./ds4 -m $M --vision $V --role worker --layers 21:output \
  --coordinator 192.168.1.10 9911 --ctx 1000000 --prefill-chunk 2048
```

The [two-node compose file](DOCKER.md#two-machines) runs exactly this. See
[inference across machines](DISTRIBUTED.md) for recovery and tuning, and the
[chunk size measurements](../speed-bench/gfx1151-pipeline-prefill.md#chunk-size).
Never run both stages on one machine: two model processes on one unified
memory machine can panic the kernel.

### Context memory on demand

Resident Flash on ROCm commits a session's context memory as the context
fills. `--ctx` is a reservation: the compressed KV caches and the indexer
scratch start small and grow, so a 1M-token session costs a short prompt
nothing. With a 1M reservation, 2K-token prefill is 466 tok/s instead of 153,
because the Q8->F16 weight cache keeps the memory the context is not using.
`DS4_ROCM_KV_UPFRONT` set to any value, including `0`, restores up-front
allocation for comparison; unset it to go back.

Sessions that take turns on the GPU (the slots of `--batched-session`, the
sessions of a pipeline worker) share one indexer scratch, sized for the
longest of them, instead of holding one each: at 1M tokens that is 4 to 8 GiB
per extra session. `DS4_ROCM_KV_PRIVATE_SCRATCH` set to any value gives each
session its own again.

When an allocation does not fit, memory is taken back in this order:

1. Idle server slots: saved to the disk KV cache and freed (see
   [Serving](SERVER.md)).
2. The Q8->F16 weight cache, down to 512 MiB. A prefill batch of 128 tokens
   or more that finds its weights gone expands them again over the least
   recently used of what is left, which costs about 2% of prefill speed;
   smaller batches run the Q8 kernels.
3. Routed experts: read back from the GGUF on demand (next section). This is
   the last resort; it costs real speed.

Free memory goes back the other way: experts first, then the F16 cache. A
session is refused at creation only if its full reservation could not be
committed with all three given up. Output does not depend on the first and
third step. The second changes which kernels small batches run, as a small or
large `--ctx` already did.

### Elastic expert weights

Resident Flash on ROCm loads each layer's routed experts as its own device
block. `ds4_gpu_reclaim_weights()` can then evict cold experts when the context,
a new session or the fixed weights at load need memory, and they are read back from the GGUF on demand; attention,
shared experts and all other weights always stay resident. The mechanism and
its policy are documented in `rocm/ds4_rocm_elastic.cuh`. Results do not
change: evicted layers run the same kernels on the same bytes.

Routing is only mildly skewed, so eviction is not free. Over 4000 decode
tokens of `speed-bench/promessi_sposi.txt`, 90% of a layer's selections need a
median of 81 of its 256 experts (57 to 208 across layers) and 99% need 181;
layers 0-2 are hash-routed and uniform. A 4096-token prefill batch touches 208
to 256 experts per layer. Every evicted expert that is needed again is a
6.75 MiB read.

Measured on a 128 GB gfx1151 with a 2 GB/s NVMe drive, 0731 Q2 checkpoint,
`ds4-bench --rocm --prompt-file speed-bench/promessi_sposi.txt`, eviction
forced with `DS4_ROCM_ELASTIC_EVICT=<GiB>@<pos>` after the first frontier;
prefill is one 2048-token step after the eviction, decode is 255 steady tokens:

| Evicted | Prefill 2K->4K | Decode at 2K | Prefill 30.7K->32.8K | Decode at 30.7K |
|---:|---:|---:|---:|---:|
| 0 GiB | 617 tok/s | 17.6 tok/s | 553 tok/s | 16.1 tok/s |
| 10 GiB | 272 tok/s | 15.4 tok/s | 256 tok/s | 13.4 tok/s |
| 20 GiB | 166 tok/s | 12.8 tok/s | 159 tok/s | 11.7 tok/s |
| 40 GiB | 85 tok/s | 7.0 tok/s | 83 tok/s | 6.5 tok/s |

Both are bound by SSD reads; a faster drive moves every evicted row up.
`--ssd-streaming` on the same model and drive gives 15.6 tok/s prefill and
10.1 tok/s decode at 2K. The reclaim call itself takes about 2 to 6 seconds
for 10 to 40 GiB. At most 61.8 of the 72.6 GiB of routed experts can be
evicted; every layer keeps at least 32 experts.

Diagnostics: `DS4_ROCM_ELASTIC_STATS` prints reclaim and hit/miss counters,
`DS4_ROCM_ELASTIC_NO_GROW` stops evicted layers from growing back while
memory is free, `DS4_ROCM_ELASTIC_TRACE=FILE` records every routed selection.
`DS4_ROCM_ELASTIC_STATS`, `DS4_ROCM_ELASTIC_NO_GROW` and the two switches above
only test whether the variable is set: `=0` or an empty value turns them on,
so unset them to turn them off.

### Prefill precision and `--quality`

The default `gfx1151` Flash prefill path uses F16 WMMA attention, F16 hipBLASLt
dense projections (at 32+ tokens), and tiled F16 WMMA routed-MoE kernels (at
128+ tokens). Its logits are not bit-identical to upstream `antirez/ds4`
(typically max-abs 0.6 to 3.4, RMSE 0.12 to 0.62; see
[the prefill notes](../speed-bench/gfx1151-prefill-results.md)), so greedy
completions can differ where the top-two token margin is thin. Because DeepSeek
V4 Flash amplifies small rounding differences across E4M3 KV quantization and
43 layers of top-6/256 MoE routing, the fork and upstream default paths sit at
a comparable distance from `--quality` (RMSE 0.25 to 0.32 for the fork vs 0.16
to 0.91 for upstream across 16 to 4096 tokens).

Pass `--quality` to disable the fast F16/WMMA, hipBLASLt, and MMQ kernels and
run the conservative native path instead. `--quality` matches upstream
`--quality` (not upstream's default path) and runs substantially slower (~55
tok/s vs 450 to 590 tok/s at 2K to 4K prefill).

## DeepSeek V4.1 Flash

Use Q2 with SSD streaming on a 128 GB machine:

```sh
./download_model.sh ds41f-q2
./ds4 --rocm -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --ctx 4096
```

The 341 GiB GGUF includes disk-only Engram tables. Keep it on a fast local
SSD and start with automatic cache sizing. The matching
[V4.1 DSpark](SPECULATIVE_DECODING.md#deepseek-v41-flash-dspark) support file
is optional. V4.1 vision and network TP are not supported on ROCm.
Large prefills are not yet tuned on this backend and can be slow.

## GLM 5.3 Flash

The reference Q2 setup uses SSD streaming to leave room for its graph and KV
state. Begin with automatic cache sizing and a small context:

```sh
./download_model.sh glm53-q2
./ds4 --rocm -m gguf/GLM-5.3-Flash-Q2.gguf \
  --ssd-streaming --ctx 4096
```

GLM 5.2 also supports ROCm streaming. Full-model GLM 5.2 inference requires it;
distributed layer slices can be resident. See [SSD streaming](SSD_STREAMING.md)
before adjusting the cache budget.

Both GLM 5.3 Flash and DeepSeek Flash Vision Experimental support images on
ROCm. Add the matching encoder with `--vision FILE`, as described in
[models and vision](MODELS.md#vision). The DeepSeek encoder takes 0.11 to
0.28 s per image on this machine; see the
[encoder notes](../speed-bench/gfx1151-vision-encoder.md).

For a model-free routed-kernel check, use `make test-mxfp4-rocm` and, for the
tiled MXFP4 prefill kernels, `make test-rocm-moe-mxfp4`.
Full-model validation is described in [testing](TESTING.md).
