# gfx1151 routed-down prefill follow-up

Baseline: commit `6975cf4`, including the Q-head normalization optimization.
Measured 2026-11-17 on Ryzen AI Max+ 395 / Radeon 8060S, 128 GB unified memory.
All builds, experiments and tests ran in the existing ROCm Docker container
(image `ghcr.io/felix920506/ds4:dev-rocm`, digest
`sha256:26808d103d6581990d5d50151f8fedcaf72c65720730ee6ad2b3b9f714ba1d3a`).
Resident 0731 and Vision Exp IQ2XXS/w2Q2K/AProjQ8/SExpQ8/OutQ8 checkpoints;
Vision measurements use text input.

## What changed

The Q2_K routed down projection previously gave each wave four 16-row output
tiles, with eight waves per block (512 output rows). That consumed enough
registers to prevent weight prefetching at the larger token widths. The new
schedule gives each wave two output tiles and uses four waves per block
(128 output rows). Every token width can prefetch the next half-superblock.
Two threads stage each activation row instead of four. The dequantization,
WMMA accumulation order, F16 output and routed reduction are unchanged.

This deliberately reverses the earlier choice to maximize output rows per
block. The smaller tile does more activation staging but wins overall.
Gate/up retains its original layout; applying the same change there was
slower. `DS4_ROCM_MOE_DOWN_NARROW=0` restores the previous down schedule.

## Experiments

Isolated down projection, 32 experts, 4096 output rows, 2048 input columns,
40 repetitions; values are milliseconds per launch (diagnostic, not model
throughput):

| Token tile | Original | Smaller block |
|---|---:|---:|
| 16 | 0.715 | 0.654 |
| 32 | 0.891 | 0.794 |
| 48 | 1.025 | 1.008 |
| 64 | 1.480 | 1.199 |

All sampled outputs were bit-identical. A 256-thread/two-row-tile intermediate
also improved full-model Vision throughput by 0.93% with eight exact runs.

Other experiments were rejected:

- Q2_K four-value lookup: bit-identical, slower than direct conversion.
- Removing attention prefetching: 4.63 -> 6.19 ms in the isolated test.
- Transposing attention's shared K/V tile to favor P.V: 4.83 -> 8.13 ms;
  the extra cost on Q.K outweighed its benefit. Storing both layouts exceeded
  the 64 KiB LDS limit.
- Narrowing gate/up output tiles: slower; not retained.
- Prefetching four gate/up groups at the widest token tile: slower.
- Splitting Sinkhorn out of the fused HC normalization: only a small isolated
  benefit (1.05 -> 1.01 ms); not retained in this change.

## Reproduction

After `make strix-halo`, build `make rocm-prefill-variant-bench` and run each
model separately:

```sh
./speed-bench/rocm_prefill_variant_bench -m gguf/MODEL.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --candidate-env DS4_ROCM_MOE_DOWN_NARROW --candidate-value 0 \
  --prefix-tokens 8192 --prefill-chunk 2048 --warmup-tokens 2048 --repeats 2
```

Control is the new default; candidate is the original down kernel. Both use
one engine and the same hipBLASLt plans. ABBA/BAAB order, four runs per
variant, fresh sessions, 2K warmup to exclude tuning, full 8K prefill in 2K
chunks. Every final vocabulary logit must match bit for bit.

`make test-rocm-moe-down` compares both schedules through the production
routing API with 32 experts, six routes, real 4096/2048/4096 matrix dimensions,
128/137/300/2048 tokens and three routing distributions. It checks exact output
and guard values. `make test-rocm-head-norm` covers the previous optimization.

## Validation

The final production-size test passes all 12 routing cases with identical
outputs and intact guards. All 96 Q-head normalization/RoPE cases also pass.
All five ROCm binaries build and pass `--help` smoke tests. The full-model A/B
runs require exact equality across every one of the 129,280 output logits.

## Results versus the previous commit

Median tok/s over four 8K prefills per variant, 2K chunks:

| Model | Previous commit | New down schedule | Gain |
|---|---:|---:|---:|
| Flash 0731 | 586.60 | 596.78 | 1.73% |
| Flash Vision Exp (text) | 595.16 | 601.96 | 1.14% |

All 16 measured runs are exact (1,034,240 compared logits per model).
Aggregate tok/s: 0731 585.54 -> 597.30 (+2.01%); Vision 594.70 -> 602.73
(+1.35%). The 0731 original run 3 was slower than its other runs (575.83
versus 583.70–593.43 tok/s); the median is the more conservative comparison.
Both variants show some drift across the run; ABBA/BAAB ordering balances it.
These are small measured gains on this machine, workload and quantization.
Raw logs are in `ab-0731.log` and `ab-vision.log`.
