# gfx1151 DeepSeek V4 Flash

## Scope

- Hardware: Framework Desktop, AMD Strix Halo `gfx1151`, 128 GB unified memory
- Runtime: ROCm 7.14
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`
- Prompt: `speed-bench/promessi_sposi.txt`
- Workload: pure prefill, 2048-token increments, ending at 4096 context

## Result

| Build | 2K prefill | Warm 4K prefill |
|---|---:|---:|
| Clean DS4 | - | 187.26 tok/s |
| Tuned gfx1151 path | 231.91 tok/s | 292.86 tok/s |

The warm 4K result is 56.4% faster than the clean baseline. The final run had
all tuning selectors unset; gfx1151 now selects this path by default.

## Retained changes

- Use the tuned x64/y64 IQ2 MMQ gate/up path and bound each expert by the input
  token count.
- Use shape-specific wave32 rocWMMA kernels for the hot Q8 and F16 projections.
- Use 32-head, 80-key rocWMMA attention tiles with vectorized F32-to-F16
  staging, plus the specialized attention-output B projection.
- Vectorize Q8 activation staging and stage each Q2_K weight block once per
  256-value K slab.

The environment variables remain as per-path debug overrides: setting one to
`0` disables that path. They are not required for normal gfx1151 execution.

## Validation

- The 512, 1024, 2048, and 4096 frontier logits are byte-identical to the
  previously validated 80-key attention reference.
- That reference preserves top-1 at all four frontiers against the earlier
  accepted stack. Its worst full-logit difference is max-abs 4.446 and RMSE
  0.770, inside the accepted max-abs 5.14 and RMSE 0.871 envelope.
- ROCm 7.14 builds and links `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
  `ds4-agent`; all five pass runtime help smoke tests.
- Answer-extractor self-tests pass. Q4_K and MXFP4 dot tests pass 4/4 each.

## ROCm 10.0 DSpark scheduler follow-up

The adaptive scheduler is enabled by default on gfx1151. It probes four speculative cycles and requires an average of four accepted tokens per cycle. Below that floor, the rest of the request bypasses the support model and uses ordinary target decode; the next prompt sync resets the decision.

| Workload | Aggregate generation | Steady generation | Scheduler result |
|---|---:|---:|---|
| 2K, low acceptance, 128 tokens | 15.40 tok/s | 15.91 tok/s | Bypass after four cycles |
| 16K, 100% acceptance, 128 tokens | 25.71 tok/s | 26.77 tok/s | Keep DSpark active |

The ordinary 2K control is 15.57 tok/s. The low-acceptance bypass therefore limits DSpark overhead to 1.1% while preserving the high-acceptance speedup.

The experimental compact IQ2 worklist was removed after a small-context GPU memory fault. The retained stock launch with the x64/y64 tile measures 294.38 tok/s at 4K and 268.51 tok/s at 16K in the 2K-step pure-prefill sweep.

`ds4-bench` restores local DSpark sweep frontiers from session snapshots. Restored and pure-prefill frontier logits are byte-identical at 2K and 4K; the restored 4K interval is 295.26 tok/s versus 295.27 tok/s pure.

## November 2026 prefill pass (ROCm 7.14, hipBLASLt)

Same workload as above: pure prefill in 2048-token steps, measured with
`ds4-bench --rocm --prompt-file speed-bench/promessi_sposi.txt --gen-tokens 0
--ctx-start 2048 --ctx-max 16384 --step-incr 2048`, model
`DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf` (text
only), inside the `ds4:rocm-gfx1151` container.

| Context after the step | Before | After |
|---:|---:|---:|
| 2048 (cold first chunk) | 160.41 tok/s | 347.28 tok/s |
| 4096 | 272.14 tok/s | 427.01 tok/s |
| 8192 | 264.60 tok/s | 420.84 tok/s |
| 16384 | 250.90 tok/s | 409.54 tok/s |

### Retained changes

- Every prefill-sized (32+ token) dense projection (q_a, q_b, kv, shared
  expert, attention output A/B, hc mix, indexer and compressor projections)
  runs as an F16 hipBLASLt GEMM.  Smaller batches (decode, DSpark
  verification) stay on the native kernels.  The library's heuristic order is not reliable on
  gfx1151 (its first F32-output choice can be 4x slower than the best of the
  first eight), so each shape class times the candidates once and keeps the
  fastest; the choice is shared across power-of-two token buckets.  The
  rocBLAS `GemmEx` F32-output path (4-5 TFLOPS) and the custom Q8/F16 WMMA
  row-tile kernels (5-10 TFLOPS) are gone; hipBLASLt reaches 20-27 TFLOPS
  with F32 outputs and 33-42 with F16 outputs at these shapes.
- The F16 staging buffer is allocated on ROCm too, so q_b, the attention
  output and the shared-expert down projection write F16 straight into the
  half-input norm/rope and hyper-connection kernels.  The transposed F16
  weight cache and its 74 ms per-layer cold dequant are removed.
- One 32-head WMMA attention kernel serves every prefill chunk: the first
  chunk of a prompt used to run the scalar two-pass kernel (40 ms per layer).
  The kernel stages each 32-key tile once for both Q.K^T and P.V, keeps Q in
  registers, splits Q.K^T over all 16 waves, keeps the P.V accumulators in
  registers with per-row online-softmax rescaling, and prefetches the next
  tile.  Indexed layers: 32.6 -> 19.7 ms per 2048 tokens.
- The Q2_K routed down projection runs as a tiled F16 WMMA kernel (64 tokens
  x 128 rows per block, 64-wide K chunks, raw superblocks staged in LDS,
  activations prefetched); the SwiGLU writes its F16 input directly.  It
  runs only when the MMQ gate/up pair produced every routed mid row; the
  other gate/up routes (`DS4_ROCM_MMQ_IQ2=0`, other GPUs, partly resident
  streamed layers, an MMQ failure) keep the hot/cold down split.  The
  gate/up pair stays on the vendored MMQ kernels without the redundant
  sanitize passes.  Down: 32 -> 18 ms per layer.
- Indexer scores keep the ReLU-weighted head sum in the WMMA accumulator
  registers and stage four heads per barrier: 6.9 -> 1.2 ms per layer at 4K
  context, and the cost no longer dominates long contexts.
- The deterministic sorted-pair scatter runs 256 threads per expert.

### Validation

- Frontier logits at 2048, 4096 and 6144 tokens against the previous image's
  binary: top-1 identical at all three frontiers, top-5 identical at two and
  4/5 at the third, max-abs 0.8-1.7 and RMSE 0.15-0.30 — inside the accepted
  max-abs 5.14 / RMSE 0.871 envelope.  Builds that only changed the WMMA
  attention or down tiles differ from each other by the same amount because
  the hipBLASLt algorithm choice can differ between processes.
- The vendored MMQ Q2_K kernel was tried for the down projection and
  rejected: 258 ms per layer on gfx1151.  `ROCM_MMQ_Y=128` gave no gain
  over 64.

## November 2026 routed MoE pass (ROCm 7.14)

Same machine and prompt, model
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`,
built and run inside the ROCm container.  Pure prefill, `--gen-tokens 0`:

| Workload | Before | Routed MoE | Glue pass |
|---|---:|---:|---:|
| 2048-token steps, context 2048 (cold first chunk) | 350.67 tok/s | 434.53 tok/s | 450.76 tok/s |
| 2048-token steps, context 4096 | 428.55 tok/s | 553.04 tok/s | 589.83 tok/s |
| 2048-token steps, context 8192 | 419.67 tok/s | 542.98 tok/s | 577.18 tok/s |
| 2048-token steps, context 16384 | 411.24 tok/s | 527.09 tok/s | 560.11 tok/s |
| 4096-token steps, context 8192 | 435.07 tok/s | 566.22 tok/s | 598.93 tok/s |
| 512-token steps, context 1024 | 280.62 tok/s | 431.45 tok/s | 449.58 tok/s |
| 128-token steps, context 256 | 135.49 tok/s | 233.81 tok/s | 241.44 tok/s |

The three binaries were built from the same tree state and run back to back.
"Routed MoE" is the kernel pass described next, "Glue pass" adds the changes
under "Follow-up" below.  Decode is unchanged (14.3 tok/s at 2K); batches
under 128 tokens do not reach the new kernels.  The Vision Experimental
checkpoint's text path moves the same way.

### What bounds these kernels

Measured by removing one piece at a time from the kernels:

- The WMMA instruction is cheap.  LDS traffic is not: operands read from LDS
  for every tile cost more than the multiplies, and row strides that are not
  an odd multiple of 16 bytes run up to three times slower.
- Experts receive anything from one token to the whole batch (about 8 of 256
  take half of all pairs, over 100 take 16 or fewer).  Wide token tiles are
  bound by the WMMA rate.  Narrow ones are bound by reading the expert's
  weights, which every tile does once whatever its width, and reading them a
  group at a time re-touches every cache line.
- The old down kernel launched a block for every (expert, 64-token slot up to
  the largest bucket); 94% of them returned at once.

### Retained changes

- Routed gate/up runs `moe_gate_up_iq2_tile_wmma_kernel` instead of the
  vendored MMQ pair and its SwiGLU pass.  Each expert bucket is cut into
  tiles of at most 64 tokens, launched per width (1 to 4 16-token tiles).
  Only the F16 activations go through LDS.  Every wave owns 32 rows of gate
  and of up and dequantizes them into the B operand in registers: exact F16
  grid magnitudes from an LDS table with the sign bits xored in, one lane per
  row, the other half of the wave filled by a lane permute.  The group scale
  is applied by rescaling the accumulators when the group changes.  Each
  operand serves all token tiles of the block, 16 WMMA tiles per wave and
  step at full width.  All but the widest tile fetch four groups of a row at
  a time.  39 -> 23 ms per layer.
- The Q2_K down projection is rebuilt on the same tiles and layout
  (`moe_down_q2K_tile_wmma_kernel`): no empty blocks, no weight staging in
  LDS, half a block of a row fetched at a time.  18.5 -> 10.6 ms per layer.
- `DS4_ROCM_MMQ_IQ2=0` still turns the whole resident prefill route off.  The
  MMQ pair remains for shapes that are not whole IQ2_XXS/Q2_K blocks.

### Validation

- The down kernel is bit-identical to the one it replaces: frontier logits
  at 2048, 4096 and 6144 tokens match exactly with the gate/up kernel held
  fixed.
- Gate/up against an exact double-precision host evaluation of sampled
  outputs (576 per layer, ten layers over two chunks): relative RMS error of
  the weighted SwiGLU output 0.02-0.04%, against 0.1-0.8% for the MMQ pair,
  which quantizes the activations to 8 bits.
- Frontier logits against the previous binary, with the hipBLASLt algorithm
  pinned so each binary is deterministic: top-1 identical at 2048, 4096 and
  6144, max-abs 1.3-3.7, RMSE 0.20-0.65.  Against `--quality` the two
  binaries sit in the same band (RMSE 0.24-0.60 new, 0.23-0.42 old at those
  frontiers; 0.34-0.67 new, 0.49-1.05 old at 128 to 2048 in doubling steps),
  top-1 identical everywhere.
- `make test-rocm` with the 0731 checkpoint: every model test passes.  The
  `server` unit group fails four decode-stat formatting assertions with or
  without this change.

### Tried and dropped

- Activations loaded from global memory straight into the A operand: twice
  as slow as staging them in LDS.
- Computing the RoPE rotation once per run of heads instead of once per
  pair: slower, the kernel is bound by its strided read-modify-write and
  serialising it inside a thread costs more than the trigonometry.
- Dispatching the two head groups of a token next to each other in the
  attention kernel: no change.

### Follow-up: what was left around the kernels

After the MoE pass a warm 2048-token layer took about 87 ms: routed MoE 33,
hipBLASLt GEMMs 22, attention 12.5 on average (19 on the layers with a
compressed-row top-k, 4.7 on the others), and about 19 of elementwise
kernels between them.  Three changes, all bit-identical in the frontier
logits to the build before them (2048 to 16384 tokens in 2048 and 4096
steps, 128 to 2048 in doubling steps, 300/1000/1700, both checkpoints, with
the hipBLASLt algorithm pinned):

- `attention_mixed_heads32_wmma_kernel<1>` built its list of visible top-k
  rows with one lane scanning 512 entries while the other fifteen waves of
  the block waited at the barrier.  All lanes now load their entry and place
  it by the count of kept entries before it (kept bits of the own wave, plus
  per-wave counts through LDS), which keeps the order.  19.1 -> 15.5 ms on
  those layers.
- The F16 output projection undoes the tail rotation of the heads while it
  packs them (`attention_pack_group_heads_inv_rope_f16_kernel`, one block
  per token, the token's angles computed once).  The separate in-place
  inverse rotation is gone from that path: it only touched an eighth of
  every head row but took as long as packing all of it (2.4 ms a layer).
  `ds4_gpu_attention_output_q8_batch_f16_tensor` takes the rotation
  parameters for this; the paths that do not use it rotate in place as
  before.  The rotation is written with the FMA contraction
  `rope_tail_kernel` compiles to, which is what makes the packed halves
  match bit for bit.
- The routed MoE kernels pick each row tile's half of the mirrored operand
  with a select instead of two masks (about 4% of their time).

What the remaining time is bound by, measured the same way:

- The wide gate/up tiles run at 75-80% of what the kernel does with decode,
  activation loads, staging and rescaling all removed.  Rescaling the
  accumulators by the group scale is the largest single piece (about 12%);
  reading every weight from one cached region changes it by 7%, and padding
  LDS to 63 KiB per block changes nothing, so neither memory traffic nor
  block occupancy limits them.
- The elementwise kernels between the GEMMs (HC expand and weighted sum,
  norms, F32/F16 conversions, the head pack) all move about 200 GB/s
  whatever they compute.  They are bound by memory traffic over F32
  activations, most of it the 268 MiB of Q and of attention output per
  layer; only keeping those in F16 end to end would remove it.
- The F32-output GEMM shapes have 7 hipBLASLt solutions against 670 for F16
  output; timing all of them finds nothing faster than the heuristic's pick.

Tried and dropped in this pass:

- One block per token for the standalone `rope_tail_kernel`, angles computed
  once: no faster, the kernel is bound by its strided read-modify-write.
- One wave per head row for `head_rms_norm_rope_tail_from_half_kernel`:
  3.05 -> 2.38 ms, but the logits no longer matched bit for bit and the
  difference was not traced to a single operation.
