# gfx1151 pipeline prefill

## Scope

- Hardware: two AMD Ryzen AI Max+ 395 machines, Radeon 8060S (`gfx1151`),
  128 GB unified memory each, 2.5 GbE between them
- Runtime: ROCm 7.14, built and run inside the ROCm container
- Model: `DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf`
  (156 GB, text path), coordinator on layers `0:20`, worker on `21:output`
- Workload: `ds4-bench --rocm --role coordinator --prompt-file
  speed-bench/promessi_sposi.txt --ctx-alloc 1000000 --ctx-start 32768
  --ctx-max 98304 --step-incr 32768 --gen-tokens 0`; the worker runs with
  `--ctx 1000000` too.  Each step is one pipelined prefill of eight
  4096-token chunks.

"Before" is 56eb3c1 on both nodes.

## Result

| Context after the step | Before | After |
|---:|---:|---:|
| 32768 | 273 tok/s | 853 tok/s |
| 65536 | 274 tok/s | 852 tok/s |
| 98304 | 267 tok/s | 798 tok/s |

Run to run these move by about 5%.  A single 4096-token step,
which no stage can overlap, went from 140-160 to 340-520 tok/s.

## Where a chunk went

Both stages were busy all the time, each for about 13 s per chunk, and the
2.5 GbE link carried the chunk's 256 MiB of hidden state in 0.9 s of that on
its own thread.  So the time was in the layers, not in the pipeline.  Per
layer and 4096 tokens on the coordinator (`DS4_ROCM_LAYER_STAGE_PROFILE=1`,
which waits for the device after every stage):

| Stage | Before | After |
|---|---:|---:|
| Routed experts | 468 ms | 75 ms |
| Attention | 22 ms | 23 ms |
| Attention output projection | 22 ms | 22 ms |
| Q path | 16 ms | 16 ms |
| Everything else | 36 ms | 36 ms |

The Q2 checkpoints had WMMA kernels for their routed experts
([gfx1151-prefill-results.md](gfx1151-prefill-results.md)).  MXFP4 experts
still ran the integer kernels: activations quantized to Q8_K, eight tokens of
an expert to a block.

## Change

`moe_gate_up_mxfp4_tile_wmma_kernel` and `moe_down_mxfp4_tile_wmma_kernel`
put MXFP4 on the token tiles and the layout of the IQ2_XXS and Q2_K kernels:
up to 64 tokens of one expert to a block, F16 activations staged in LDS once
per 64-wide chunk, every wave dequantizing its own weight rows into the WMMA
operand.

- An MXFP4 weight is an E2M1 value times its block's power of two.  The eight
  magnitudes are F16 values with a zero low byte, so one byte permute turns
  four codes into four operand high bytes.
- The operand holds the E2M1 values and the accumulators carry the scale:
  they are kept in units of the current block's scale and multiplied by the
  ratio, a power of two, when the block changes.  The weights reach the
  product unrounded.
- Weights are fetched 128 to a row at a time (four blocks, 68 bytes), one
  fetch ahead of their use.

In a harness with 256 experts and a production-like route skew, 4096 tokens:
467 ms before, 62 ms after, of which gate/up is about 46 ms.  The down
projection uses the narrow shape (128 rows a block): 15 ms against 25 ms for
512 rows.  Batches under 128 tokens, `--quality` and `DS4_ROCM_MXFP4_TILES=0`
keep the integer kernels.

## Validation

- `make test-rocm-moe-mxfp4`: the launch path against a double-precision
  host evaluation (F16 activations, exact weights, F16 mid and down rows) at
  the production shape, 128 to 4096 tokens, three route patterns with partial
  tiles and empty experts, weights with zero-scale and out-of-range blocks.
  Relative RMS error of the output 9.3e-5 for the tiles, 7.3e-3 for the
  integer kernels on the same inputs.
- `make test-mxfp4-rocm` holds the integer kernels to their Q8_K oracle as
  before, with the tiles turned off.
- Frontier logits across the two nodes at 4096, 8192 and 12288 tokens,
  against `--quality` on both nodes:

  | Kernels | RMSE | Max-abs | Top-1 | Top-5 |
  |---|---:|---:|---|---|
  | Tiles | 0.33, 0.48, 0.44 | 1.8, 2.5, 2.3 | same at all three | 5, 5, 4 of 5 |
  | Integer | 0.28, 0.41, 0.48 | 1.5, 2.4, 2.4 | differs at 12288 | 5, 5, 5 of 5 |

  Both sit in the same band, inside the accepted max-abs 5.14 / RMSE 0.871
  envelope.  `--quality` also changes kernels outside the experts, and the
  hipBLASLt algorithm choice differs between processes, so the band is not
  the experts' error alone; the kernel test above is the measure of that.

## Tried and dropped

- Tile-major block order (all token tiles of an expert on the same weight
  rows in a row, as the integer kernel schedules them): 94 ms instead of 66.
  Row-major keeps a tile's activations in cache across its row blocks.
- Fetching weights without the one-step lookahead at the widest tile, as the
  IQ2 kernel does for lack of registers: 66 ms instead of 62.
- `--dist-activation-bits 16` halves what the link carries and changes
  nothing here (808, 840, 788 tok/s): the link is not the bound.

## Left on the table

- A stage now takes about 4 s per chunk and the profile accounts for 3.6 s of
  it.  The experts are still the largest stage, at about the rate of the Q2
  kernels; attention, its output projection and the Q path are 60 ms of a
  172 ms layer.
- A step ends with the worker finishing the last chunk alone, one chunk in
  eight at this step size.  Longer prefills amortize it.

## Long contexts: the F16 weight cache

Two slots (`ds4-server --batched-session 2`, 2048-token slices) growing to
160K tokens each on the coordinator, with `--simulate-used-memory 27GB`
standing in for the memory a longer context would take.  At about 120K tokens
per slot the contexts start taking the Q8->F16 weight cache (5.1 GiB for
layers `0:20`).

An evicted weight used to send its projection to the Q8 kernels, which are
built for decode.  Now the cache keeps 512 MiB, and a batch of 128 tokens or
more expands a missing weight over the least recently used entry of the same
size.  Per 2048-token slice on the coordinator, all 21 layers:

| Stage | Cache intact | Evicting, before | Evicting, after |
|---|---:|---:|---:|
| Attention output projection | 0.24 s | 5.65 s | 0.28 s |
| Q path | 0.17 s | 0.54 s | 0.19 s |
| Routed experts | 0.83 s | 1.46 s | 0.85 s |
| Whole slice at 140K tokens | 2.7 s | 9.7 s | 2.7 s |

The whole prefill of 2 x 160K tokens: 427 tok/s before, 761 after.

With the cache held at 512 MiB on both nodes from the start
(`DS4_ROCM_Q8_F16_CACHE_MB=512`, a diagnostic), the benchmark at the top
gives 841, 838 and 786 tok/s against 857, 853 and 797, and frontier logits at
4096, 8192 and 12288 tokens identical to the full cache.
`make test-rocm-q8-f16-reuse` cycles seven weights through a four-entry cache
and checks every product against a host evaluation.
