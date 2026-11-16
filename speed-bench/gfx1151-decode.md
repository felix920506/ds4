# gfx1151 DeepSeek V4 Flash decode

## Scope

- Hardware: AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB unified
  memory
- Runtime: ROCm 7.14, built and run inside the ROCm container
- Models: `DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf`
  (text path) and the 0731 checkpoint of the same quantization
- Workload: `ds4-bench --rocm --prompt-file speed-bench/promessi_sposi.txt
  --gen-tokens 128`, greedy single-token decode after the prefill frontier,
  full resident model

## Result

| Checkpoint, context | Before | After |
|---|---:|---:|
| Vision Experimental, 2048 | 14.31 tok/s | 17.61 tok/s |
| Vision Experimental, 8192 | 13.99 tok/s | 17.10 tok/s |
| Vision Experimental, 32768 | 13.07 tok/s | 15.88 tok/s |
| 0731, 2048 | 14.31 tok/s | 17.61 tok/s |

"Before" is `main` at db94614.  The two binaries were run back to back; the
2048 rows repeated within 0.01 tok/s.  Prefill is unchanged (448 tok/s at
2048 with either binary).  Every decode step produces the same logits bit
for bit as before; see Validation.

## What bounds decode

A token reads about 9.5 GB of weights: every Q8_0 and F16 projection of 43
layers, six of 256 routed experts per layer, and the output head.  A plain
read kernel on this machine peaks near 255 GB/s, so 37 ms of the original
70 ms per token were unavoidable and the Q8_0 projections (31 ms, 210 to
230 GB/s) already sat near that bound.  The rest was lost in four ways:

- Routed experts were bound by arithmetic, not memory.  With the weights
  cached, gate/up still took 158 of its 197 us per layer and down 117 of 167.
- A wave waits for memory on every load it consumes at once.  Loops with one
  load per add (the hyper-connection matmuls, the RMS norms, attention over
  the selected rows) ran at the latency of a cached load per element.
- More waves are not more throughput.  With one wave per output row, thirty
  waves share a compute unit, their weight lines overflow its 32 KiB L1 and
  are fetched again.  The 4096x1024 F16 compressor pair took 180 us that way,
  158 us in 32-row blocks, and 92 us with 120 waves in total.
- LDS is not free, and neither are cached global loads next to a weight
  stream: LDS traffic bounded the first gate/up rewrite, and moving its
  tables to global memory made the cached case faster and the real one
  slower.

## Retained changes

Per layer, in a kernel microbenchmark over weights that do not fit a cache:

| Kernel | Before | After |
|---|---:|---:|
| Routed gate/up, IQ2_XXS, six experts | 197 us | 125 us |
| Routed down, Q2_K, six experts | 167 us | 132 us |
| F16 compressor pair, 4096x1024 | 158 us | 92 us |
| F16 compressor pair, 4096x512 | 84 us | 54 us |
| Indexed attention, 128 raw + 511 selected rows | 121 us | 54 us |
| Window attention, 128 raw + 16 compressed rows | 36 us | 14 us |
| Q8_0 attention output B with hyper-connection expand | 176 us | 168 us |
| Q8_0 shared down with hyper-connection expand | 50 us | 43 us |
| Hyper-connection F16 matmul, 16384x24 | 26 us | 9.6 us |
| Router F16 matmul, 4096x256 | 15 us | 6.5 us |
| RMS norm over 16384 | 14 us | 5.2 us |

- Gate/up (`moe_gate_up_mid_decode_iq2_kernel`): a thread owns one
  256-weight block of a gate row and of the matching up row, fetches both up
  front, reads each activation dword once for the two, and signs the grid
  bytes with one xor and one add from a mask table.  That is about 15
  instructions per eight weights instead of 48.  The streaming variant shares
  the body.
- Down (`moe_down_q2K_sum_rows_w32_kernel`): the sub-block scales of a block
  arrive as two words and the two quant bytes once, and per-lane offsets live
  in the base pointers; 13 loads per block instead of 19.
- F16 projections (`matmul_f16_rows_w32_kernel`): `DS4_ROCM_DECODE_STREAMS`
  single-wave blocks walk the rows in turn, with no LDS staging.  The Q8_0
  hyper-connection expand kernel is launched the same way and fetches the
  row's residuals ahead of the dot product.
- Attention: the score dot products and the weighted value sums fetch several
  rows ahead of the arithmetic, a thread owns four value columns per row
  load, the raw ring index is a conditional subtract instead of a modulo, and
  the indexed kernel compacts the top-k rows on all lanes instead of one.
- The small F16 matmuls, the plain RMS norm and the fused hyper-connection
  sum fetch eight operands ahead of their sums.

Roughly where a 2048-context token goes now (56.7 ms): Q8_0 projections and the
output head 30 ms, routed experts 11 ms, F16 projections 6 ms,
attention 2 ms, the small kernels and launch gaps the rest.

## Validation

Every rewritten kernel keeps the values and the summation order of the one
it replaces, so results are compared bit for bit, not within a tolerance.

- `DS4_BENCH_DECODE_LOGITS_HASH=1 ds4-bench --teacher-forced-decode
  --prefill-chunk 16` hashes the full logits of every decode step.  The small
  prefill chunk keeps prefill off hipBLASLt, whose algorithm choice differs
  between processes.  Old and new binaries give identical hashes for: 96
  steps after 2048 tokens, 400 steps after 16 tokens, 48 steps after 8192
  tokens, 48 greedy steps after 300 tokens, 64 steps on the 0731 checkpoint,
  and 160 steps with `--ssd-streaming`.
- Each kernel was also run against a copy of the old one on random weights
  and activations, including odd shapes (short last blocks, three
  hyper-connection streams, two tokens, short raw windows): all outputs
  bit-identical.
- The order is pinned in the source, not left to the compiler.  The build
  uses `-ffast-math`, and the old loops' rounding was whatever it chose: a
  fused multiply-add per element, the first of four products left unfused,
  the eight weights of a Q2_K block added in the order 7, 3, 5, 1, 6, 2, 4, 0.
  The new kernels spell those out with `fmaf` and fixed loop orders.

- `make test-rocm`: the logprob vectors pass with the 0731 checkpoint.  With
  the Vision Experimental checkpoint one vector (`short_code_completion`,
  step 0) selects a different token, and so does `main` at db94614.  The
  `server` unit group fails four decode-stat formatting assertions, as it did
  before this change.

## Tried and dropped

- Staging the routed mid activations in LDS for the down kernel: bit-identical
  and no faster; LDS-heavy blocks lower occupancy.
- Computing each Q2_K sub-block scale once per wave and handing it to the
  lanes with a lane permute: fewer instructions, but the permutes cost more
  than they save (144 us against 132 us).
- Several accumulators per lane with wider weight loads in the F16 kernel:
  slower at every width; fewer waves each do more serial work.
- Row-walking streams for gate/up and for the plain Q8_0 kernels: within a
  few percent either way, so they keep their launch shapes.
- A 16-byte sign table entry (mask and carry) for gate/up: the wider LDS
  reads cost more than the two instructions they remove.

## Left on the table

The down projection is still bound by arithmetic at 132 us against a 70 us
memory floor.  Summing `scale * q * mid` per sub-block and applying `d` and
the `dmin * min` term once per block needs about a third of the instructions,
but it changes the rounding, so it would need a tolerance check instead of the
bit-for-bit one used here.
