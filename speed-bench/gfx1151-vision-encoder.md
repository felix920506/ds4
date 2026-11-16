# gfx1151 DeepSeek V4 Flash Vision Experimental image encoder

## Scope

- Hardware: AMD Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB unified
  memory
- Runtime: ROCm 7.14, built and run inside the ROCm container
- Encoder: `DeepSeek-V4-Flash-Vision-Encoder.gguf` (32 blocks, 1024 wide,
  BF16), with the Vision Experimental Q2 language checkpoint
- Workload: `speed-bench/vision_encode_bench`, five passes, on the seven
  fixture images under `tests/vision-fixtures/glm53` and `qwen38`.  Times
  include image decode and preprocessing.

## Result

| Image | Image tokens | Before | After |
|---|---:|---:|---:|
| screenshot.png, 1400x900 | 308 | 3430 ms | 256 ms |
| diagram.png, 1200x700 | 336 | 5350 ms | 274 ms |
| spatial.png, 1000x800 | 320 | 3763 ms | 254 ms |
| text.png, 1000x800 | 320 | 4645 ms | 253 ms |
| maple.png, 640x480 | 192 | 1200 ms | 127 ms |
| orbit.png, 640x480 | 192 | 1200 ms | 127 ms |
| earth.jpg, 512x507 | 169 | 943 ms | 111 ms |
| First image of the process | 308 | 3779 ms | 598 ms |
| All seven, one call each | | 20.5 s | 1.40 s |
| All seven, one request | | 20.5 s | 1.31 s |

"Before" is `main` at fa73a49, which has no request path: a request was a
loop of single encodes.  Its large images varied between 3.3 s and 7.7 s from
run to run; the table is one three-pass run.

End to end, a first request to a freshly started `ds4-server` carrying five
uncached images (1519 prompt tokens, 25 generated) went from 32.1 s to 12.4 s
with the same answer.

## What changed

- Activations are stored as BF16 words.  The encoder is defined with a BF16
  rounding after every operation, so nothing is lost, the F32-to-BF16
  conversion passes before each GEMM are gone, and the rounding kernels are
  folded into the operations that precede them.
- Attention was a per-key scalar loop: 83 ms per block on a 2,800-patch image,
  2.7 s per image.  It is now a WMMA kernel at 2.5 ms per block: one wave owns
  16 queries of one head and walks the keys 16 at a time, with the scores
  computed transposed so a lane owns one query.
- The four projections of a block run on a WMMA kernel that reads both
  operands straight from memory: 17 to 30 TFLOPS depending on the shape,
  against 4 to 5 for the rocBLAS BF16 GEMM it replaces.
- RoPE reads a per-image table.  V is written transposed in contiguous
  32-byte runs, with the head as the fast grid axis: the first version, which
  scattered single words over 64 streams per head, took 4 ms per block.
- A request's images are still encoded one at a time, but each one is decoded
  on a second thread while the previous one is on the GPU
  (`ds4_engine_vision_encode_memory_batch`).  PNG and JPEG decode costs 7 to
  37 ms per fixture, all hidden except for the first image.

Where a large image now spends its time (2,820 patches, 254 ms): the block
projections 95 ms, attention 81 ms, the aligner and patch embedding (still
rocBLAS) 14 ms, SwiGLU 14 ms, RoPE and V 11 ms, the other row-wise kernels
8 ms, decode and preprocessing on the CPU about 30 ms.

## Tried and rejected

- hipBLASLt BF16 GEMMs.  They run at 25 to 34 TFLOPS in isolation but were
  no faster than the kernel above inside the encoder (95 ms of projections
  per large image either way); they saved 10 ms on the aligner.  Their
  kernels keep the running sum in the WMMA accumulator, and the instruction
  truncates that sum at the precision of its largest operand on every step.
  On the encoder's own inputs their outputs were 5 to 18 times less accurate
  than rocBLAS (relative RMS error 1.6e-6 to 2.7e-5), and 0.05 to 0.8% of
  the values rounded to a different BF16 word than float64, against 0.01 to
  0.04% for the kernel used.  That kernel chains four instructions from zero
  and adds the partial sums with IEEE adds; it measures 2 to 4 times more
  accurate than rocBLAS.  hipBLASLt also picks its algorithm by timing, so
  the same image could encode differently in two processes.
- The same truncation ruled out a plain WMMA P.V in attention.  P stays F32
  as in the reference definition: it is split exactly into three BF16 terms,
  each key tile is accumulated from zero and tiles are added with IEEE adds.
  A two-term split was two to four times less accurate than the scalar loop,
  and chaining the tiles through the accumulator ten times less.
- Stacking the images of a request into one GPU pass: between equal and 4%
  slower than one image at a time, for 2, 4 and 8 copies of every fixture
  (measured with the hipBLASLt projections).  One image already saturates the
  GPU.

## Validation

The encoder rounds to BF16 after every operation, which makes it chaotic: a
last-bit difference in one F32 sum flips a rounding somewhere, and within a few
operations two evaluations differ by the BF16 quantization noise itself.
Embeddings of two correct builds therefore cannot be compared bit for bit,
and the old and new encoder differ by a few percent.  Two checks replace that
comparison, both against `tests/deepseek4_vision_ref.py`, a float64 evaluation
of the same definition with the same rounding points.

Every operation, from the build's own inputs to that operation, on six images
(780 to 2,982 patches, square, tall and 8:1 wide, with and without pad rows):

| Operation | Values that round differently from float64 (worst block, worst image) |
|---|---:|
| Row-wise kernels: norms, RoPE, residuals, SwiGLU, aligner | at most 0.006% |
| Attention | 0.037% (the scalar loop it replaces: 0.030% on an image where this kernel has 0.019%) |
| Block projections, F32 output | relative RMS error at most 2.6e-7 (rocBLAS in the same place: up to 9.5e-7) |

No row of any operation is off by more than 0.5% of its norm.

The whole encoder from the patches, on 30 images (the seven fixtures plus 23
crops and rescales of them), as the distance of the output embedding from the
float64 result:

| Build | Mean relative RMS | Median token | 90th percentile token |
|---|---:|---:|---:|
| Before | 2.83% | 1.27% | 3.99% |
| Before, two dimensions swapped in the attention score sum | 3.30% | 1.29% | 4.15% |
| After | 3.16% | 1.26% | 4.02% |
| After, GEMM chained two instructions instead of four | 3.23% | 1.26% | 3.97% |

The second and fourth rows are equally valid evaluations of the same code, to
show how far the mean moves by chance.  A few unstable tokens per image,
largely the same ones in every build, carry most of the squared error (the
three worst hold 35 to 94% of it on the images examined), so the mean depends
on which way those fall.

With the language model, the six fixture questions in
`tests/vision-fixtures/glm53/cases.json` and a five-image request pass at
temperature 0 before and after, and `tests/test_server_vision_cache.py`
passes.  Over seven server runs of the new encoder, six of the seven answers
were byte-identical to the old encoder's every time.  The seventh (the
spatial case) was byte-identical in two runs and a rewording with the same
facts in five.  The embedding the server computed for that image was
bit-identical to the benchmark's, so that variation arises on the
language-model side.

## Reproducing

```sh
make strix-halo
make vision-encode-bench-rocm
./speed-bench/vision_encode_bench \
  gguf/DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8.gguf \
  gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf \
  tests/vision-fixtures/glm53/screenshot.png \
  tests/vision-fixtures/glm53/diagram.png \
  tests/vision-fixtures/glm53/earth.jpg \
  tests/vision-fixtures/glm53/spatial.png \
  tests/vision-fixtures/glm53/text.png \
  tests/vision-fixtures/qwen38/maple.png \
  tests/vision-fixtures/qwen38/orbit.png
```

The float64 check is described in [testing](../docs/TESTING.md).
