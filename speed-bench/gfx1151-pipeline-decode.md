# gfx1151 pipeline decode

## Scope

- Hardware: two AMD Ryzen AI Max+ 395 machines, Radeon 8060S (`gfx1151`),
  128 GB unified memory each, 2.5 GbE between them
- Runtime: ROCm 7.14, built and run inside the ROCm container
- Model: `DeepSeek-V4-Flash-Vision-Exp-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out.gguf`
  (156 GB, text path), coordinator on layers `0:20`, worker on `21:output`
- Workload: `ds4-bench --rocm --role coordinator --prompt-file
  speed-bench/promessi_sposi.txt --ctx-alloc 1000000`, greedy single-token
  decode after the prefill frontier; the worker runs with `--ctx 1000000` too,
  so both nodes hold the context the way a 1M-token server does

"Before" is `main` at 69fc360 on both nodes.

## Result

Decode throughput once neither node is thermally throttled (see "After a long
prefill"):

| Context | Before | After |
|---:|---:|---:|
| 1600 | 14.58 tok/s | 14.6 to 14.9 tok/s |
| 6400 | 13.69 tok/s | 13.6 to 13.9 tok/s |
| 25600 | 13.10 tok/s | 13.5 to 13.8 tok/s |
| 102400 | 11.01 tok/s | 13.2 tok/s |
| 409600 | 6.77 tok/s | 12.45 tok/s |

Up to 25600 these are the 128 tokens after each frontier of a stepped run,
and "After" is the range of two such runs: the difference below 6400 tokens is
within the noise.  "Before" at 102400 is from the same stepped run, which was
not throttled there; the other long-context figures are the unthrottled steps
of 600- and 800-token runs, taken from `DS4_DIST_DECODE_PROFILE`.  Prefill to
409600 tokens went from 224 to 255 tok/s.

Decode logits are unchanged bit for bit; see Validation.

## Where a token goes

At 2K tokens a step takes 71 ms: 30 ms in the coordinator's 21 layers, 37 ms
in the worker's 22 layers and output head, 3.3 ms on the wire (the hidden
state out, 0.5 MB of logits back).  Each stage's kernels read their weights
at close to the memory's bandwidth, as in single-node decode
([gfx1151-decode.md](gfx1151-decode.md)), so short contexts leave little to
take.  What was left is everything that grows with the context, and what
happens to a stage that works in bursts.

## Changes

### Indexer scores

Every ratio-4 layer scores all of its compressed rows against the new token:
64 heads of 128 values per row.  `indexer_score_one_direct_kernel` spent a
block of four waves on each row and crossed LDS for every head: 0.9 ms per
layer at 131072 tokens, 19 ms of a 97 ms step, and about half of a step at a
million.  `indexer_score_one_rows_kernel` gives a row to a lane: the row's
128 values stay in registers, the query is read with scalar loads shared by
the wave, and a wave scores 32 rows without LDS or a barrier.

| Rows (context) | Before | After |
|---:|---:|---:|
| 8192 (32K) | 0.160 ms | 0.036 ms |
| 32769 (131K) | 0.611 ms | 0.089 ms |
| 102400 (410K) | 2.87 ms | 0.37 ms |
| 262144 (1M) | 7.97 ms | 0.82 ms |

The scores are the same bit for bit, which matters because the top-k compares
them: a four-term fused dot per original lane, the pairwise tree that
`warp_sum_f32` leaves in lane 0, the heads added in order.  Holding that
order under `-ffast-math` took more than the fp pragma: the backend still
fused the last multiply of each head into the running sum.  A value passed
through an empty asm operand is opaque to it.  Up to 2048 rows the old kernel
stays, since one wave of the new one takes 33 us for its 64 heads.

### Top-k

Above 8192 rows the top 512 were found by sorting the scores in 4096-row
chunks and merging the chunks eight at a time, each a full bitonic sort.
`indexer_topk_select_kernel` is a radix select in one block: a row's key is
its score above its reversed row number, so keys are distinct and the order
(score descending, row ascending) is exact; a histogram in LDS finds ten more
bits of the 512th largest key per pass, and the keys at or above it are
gathered and sorted.  It stops early once a digit separates exactly the rows
wanted.

| Rows | Before | After |
|---:|---:|---:|
| 32769 | 0.122 ms | 0.027 ms |
| 65536 | 0.122 ms | 0.044 ms |
| 262144 | 0.192 ms | 0.098 ms |
| 40000, 64 tokens | 1.17 ms | 0.10 ms |

The last row is the prefill shape, which takes the same path.  A row whose
scores are all equal is the worst case, 0.70 ms at 250000 rows: every key
lands in one histogram bin until the row-number digits.

### Small KV caches stay in device memory

A reservation as large as 1M tokens puts the KV caches in managed memory, so
that they can use the CMA region.  Attention reads managed memory 1.6 to 1.9
times slower than device memory on this machine, in every layer and on both
nodes, and advising huge pages for the range does not change it.  (A
microbenchmark showed the opposite: no difference on an idle GPU, a factor of
two while another job shared it, cured by huge pages.  Neither was the
pipeline.)  The size is all in the ratio-4 caches.  Of the 13 GiB a 1M-token
context grows to, the raw windows are 0.36 GiB and the ratio-128 caches about
as much, so those now stay in device memory: per layer at 25600 tokens, a
ratio-128 layer's attention went from 0.080 to 0.045 ms and a layer with only
a raw window from 0.146 to 0.091 ms.  The ratio-4 layers still gather their
512 selected rows from managed memory (0.145 ms against 0.10 ms).

### Prefix hash

The coordinator hashed the session's whole token prefix for every decode
step, 4 ms at a million tokens.  A step now starts from the hash the previous
step produced, as long as nothing else has moved the timeline.

## After a long prefill

Decode measured straight after a long prefill is not a measure of the
kernels on this machine.  The prefill leaves the die at its limit (Tctl
100 C).  A pipeline stage then decodes in bursts, idle while the other stages
hold the token, and while Tctl stays near 90 C each burst runs about 2.5
times slower in every weight-reading kernel: 82 ms for the coordinator's
slice instead of 33.  Single-node decode, which keeps the GPU busy, does not
show it, and neither does a stage that starts cool.  It is the same on `main`.

How long it lasts varies with how hot the box already was.  800 tokens after
a 409600-token prefill:

| Run | Throttled steps | 800 tokens | Unthrottled steps |
|---|---:|---:|---:|
| Before | 473 | 5.49 tok/s | 6.77 tok/s |
| After | 56 | 11.77 tok/s | 12.45 tok/s |
| After, sleeping waits (dropped, below) | 592 | 8.04 tok/s | 12.16 tok/s |

So compare the unthrottled steps.  A throttled stage shows in
`DS4_DIST_DECODE_PROFILE` as `local=`, or the worker's `eval=`, at 2.5 times
its usual value on some steps and not others, and in `sensors` as Tctl near
90 C.

## Validation

- `DS4_BENCH_DECODE_LOGITS_HASH=1 DS4_CUDA_NO_Q8_F16_CACHE=1 ds4-bench
  --teacher-forced-decode --prefill-chunk 16 --dist-prefill-chunk 16` across
  the two nodes: 48 decode steps after 40000 tokens, where both new kernels
  run in every ratio-4 layer, give identical logits hashes with old and new
  binaries.
- Single-node speed is unchanged: `ds4-bench` on the Q2 checkpoint decodes at
  17.57 tok/s at 2048 tokens and 17.23 at 8192 with either binary, and
  prefill is within 1%.
- `make test-rocm` with the 0731 checkpoint: every group passes except
  `server`, where four decode-stat formatting assertions fail when the group
  runs after a model test.  They read process-wide metrics; `./ds4_test
  --server` passes alone.
- `make test-rocm-indexer`: the scores against a copy of the old kernel on
  both sides of the 2048-row switch, smooth values, values spread over many
  binades and repeated rows, bit-identical; the top-k against a sort on the
  CPU for 8193 to 250000 rows, one and three tokens, k of 512, 1024 and
  2048, with few distinct values, signed zeros, one value everywhere and a
  causal mask.

## Tried and dropped

- Sleeping instead of spinning while a stage waits for the device
  (`hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`).  HIP's default keeps
  the waiting thread on a core: 63 ms of CPU for a 32 ms decode slice, 35 ms
  when it sleeps, and a cooler die: after a 102400-token prefill Tctl sat at
  85 C instead of 92 C and 58 of 600 steps were throttled instead of 369.
  But with blocking waits on either node a pipelined prefill in 16-token
  chunks hangs at its start, three runs of three, the main thread parked in
  the runtime.  It also costs a process that owns the whole model 5% of its
  decode speed.
- Huge pages for managed memory (`MADV_HUGEPAGE` on the allocation): the
  caches did come up in 2 MiB pages and attention did not get faster.
- `AMD_DIRECT_DISPATCH=1`, `ROC_CPU_WAIT_FOR_SIGNAL=0` and
  `DEBUG_HIP_BLOCK_SYNC=0` for the remaining spin of the runtime's queue
  thread: no change in CPU time or speed.
- Keeping the GPU occupied with a trivial kernel from a second process during
  the throttled phase: fewer slow steps, not none, and more heat.

## Left on the table

- The wire costs 3.3 ms a step, most of it the 0.5 MB of float32 logits at
  2.5 Gbit/s.  Sending less would mean deciding on the worker what the
  sampler may need.
- The throttling after a long prefill.  It is the machine's cooling and power
  policy (`power_dpm_force_performance_level`, the fan curve) and the CPU a
  waiting thread burns; the first needs root, the second a runtime whose
  blocking waits do not hang.
- The score kernel is at 0.82 ms for a 134 MB cache that a plain read covers
  in 0.53 ms.
- The ratio-4 layers' gather from managed memory.
