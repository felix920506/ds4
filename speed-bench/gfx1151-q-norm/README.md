# gfx1151 prefill Q head normalization

Measured 2026-11-17 on Ryzen AI Max+ 395 / Radeon 8060S (gfx1151),
128 GB unified memory. Pulled `origin/main` with `git pull --ff-only` to
`9bb45f0` before building or measuring. All builds, experiments, and tests ran
in Docker using `ghcr.io/felix920506/ds4:dev-rocm`, image
`sha256:26808d103d6581990d5d50151f8fedcaf72c65720730ee6ad2b3b9f714ba1d3a`.

Models: the resident IQ2XXS/w2Q2K/AProjQ8/SExpQ8/OutQ8 0731 and Vision Exp
checkpoints. Vision Exp measurements use text input; image encoding is outside
this change. Prompt: `speed-bench/promessi_sposi.txt`.

## Change

For 512-wide Q heads on gfx1151, the fused F16 Q projection's normalization
and RoPE use one wave per head and eight heads per block. The original uses
256 threads and block barriers per head. The new path keeps the original
reduction tree: eight partial sums per lane, three register reduction stages,
then five wave-shuffle stages. Explicit floating-point adds prevent
`-ffast-math` from reassociating the register sums. Other widths, devices and
decode retain their existing paths. `DS4_ROCM_Q_NORM_WAVE=0` restores the
original prefill normalization for diagnostics.

A standalone 2048-token / 64-head kernel experiment measured 2.80 ms before
and 1.93 ms after (31% less time), with identical outputs. Full-model gains
are much smaller because this kernel is only a small part of each layer.

## Baseline before edits

Built the freshly pulled tree with `make strix-halo -j16` and saved its binary.
For each model, ran:

```sh
./ds4-bench --rocm -m gguf/MODEL.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 8192 --step-incr 2048 --gen-tokens 0 \
  --dump-frontier-logits-dir /tmp/frontiers --csv /tmp/result.csv
```

The four `baseline-*.csv` / `candidate-*.csv` files record these independent
processes before and after. The 2K frontier includes first-use GEMM tuning;
subsequent frontiers time an additional 2K tokens. Their full-vocabulary logits
match exactly at all four frontiers for both models. Independent-process timing
is useful as a baseline but the alternating comparison below is the performance
measurement for this small change.

## Alternating comparison

Build after `make strix-halo` with `make rocm-prefill-variant-bench`, then:

```sh
./speed-bench/rocm_prefill_variant_bench -m gguf/MODEL.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --candidate-env DS4_ROCM_Q_NORM_WAVE --candidate-value 0 \
  --prefix-tokens 8192 --prefill-chunk 2048 --warmup-tokens 2048 --repeats 2
```

The control is the new default; the candidate is the original kernel. A single
loaded engine shares GEMM plans between both variants. Two four-run blocks use
ABBA then BAAB order, fresh sessions, full 8K prompts, and a 2K warmup for both
variants so the measured shapes have already been tuned. Startup/model loading
is excluded. The harness aborts on any full-vocabulary logit mismatch.

## Results

Median throughput across four measured 8K prefills per variant (tok/s):

| Model | Original | Optimized | Gain |
|---|---:|---:|---:|
| Flash 0731 | 586.36 | 592.18 | 0.99% |
| Flash Vision Exp (text) | 582.58 | 588.58 | 1.03% |

All 16 runs match exactly (1,034,240 compared logits per model). The aggregate
0731 throughput is 586.26 -> 593.24 tok/s (+1.19%). Vision's aggregate is
574.42 -> 588.81 (+2.51%), but its final original-kernel run was unusually slow
(549.02 tok/s versus 581.86–585.08 for the other three). Use the medians above
as the conservative headline, rather than attributing that outlier to the
optimization. These are small improvements on this machine and prompt.
Raw per-run timings and aggregate totals are in `ab-0731.log` and `ab-vision.log`.

## Validation

- Eight baseline/candidate frontier comparisons (2K through 8K, both models):
  all 129,280 logits identical at each frontier.
- `make test-rocm-head-norm`: 96 exact comparisons against the original GPU
  reduction applied to the production GEMM output. Covers 32/33/127/2048 tokens,
  1/3/64 heads, 256/512 dimensions, zeros and varying magnitude inputs, 64/128
  rotary dimensions, forward/inverse rotation, and YaRN. Output guards intact.
- `./ds4_test --server` and `./ds4-eval --self-test-extractors`: pass.
- All five ROCm binaries build and pass `--help` smoke checks.

Raw validation summary is in `validation.txt`. No claims are made for other
GPUs, quantizations, or image-encoder throughput.
