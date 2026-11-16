# Testing and Development

[README](../README.md)

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before proposing an inference change.
The [release QA guide](../QA_BEFORE_RELEASES.md) defines the hardware matrix,
checkpoint-matched quality tests, long-context agent tasks, and speed checks.
It also records which checks were not completed. A smoke test is not full QA.

## Focused checks

Model-free checks include:

```sh
make ds4_test ds4_agent_test test-session-state
./ds4_test --server
./ds4_agent_test
```

On Metal, small GPU tensor tests are available without loading a full GGUF:

```sh
make tests/test_session_state_gpu tests/test_glm53_kda tests/test_mxfp4_metal
./tests/test_session_state_gpu
./tests/test_glm53_kda
./tests/test_mxfp4_metal
```

`make test` also includes model-backed tests. Select the right GGUF and ensure
that it fits before running it; do not accidentally load a large model on a
single device during multi-GPU QA. ROCm has `make test-rocm`.

The ROCm DeepSeek Flash Vision Experimental encoder has a float64 reference.
Encode one image with `DS4_ROCM_VISION_DUMP_DIR` set, then check every
operation against it:

```sh
mkdir -p /tmp/vision-dump
DS4_ROCM_VISION_DUMP_DIR=/tmp/vision-dump ./speed-bench/vision_encode_bench \
  --passes 0 ds4flash.gguf gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf image.png
python3 tests/deepseek4_vision_ref.py \
  gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf /tmp/vision-dump --ops --chain
```

The dump holds every tensor of the encoder: about 2 GB per thousand image
patches, 6 GB for the largest images. `--ops` fails when a kernel differs from
float64 in more than 0.2% of its values; correct kernels are five to a hundred
times below that. Embeddings of two correct builds cannot be compared bit for
bit; the script and
[the encoder notes](../speed-bench/gfx1151-vision-encoder.md) explain why.

Official-vector tests must use continuations from the same checkpoint as the
GGUF. Flash 0731 and Vision Experimental are not interchangeable fixtures.
See [test vectors](../tests/test-vectors/README.md) and
[quality scoring](../gguf-tools/quality-testing/README.md).

## Investigating output

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt
```

Token dumps catch template differences without inference. Logits and
continuations help distinguish sampling changes from graph errors. Server
traces include cache and tool-parser decisions. Keep traces private when they
contain real conversations.

## Tools and source references

- [GGUF conversion and quantization](../gguf-tools/README.md)
- [Imatrix collection](../gguf-tools/imatrix/README.md) and
  [calibration corpus](../gguf-tools/imatrix/dataset/README.md)
- [Steering vectors](../dir-steering/README.md)
- [Benchmark scripts and charts](../speed-bench/README.md)
- [Evaluation data and licenses](../EVAL_DATA.md)
- [Session payload implementation](../ds4.c) and
  [cache header definitions](../ds4_kvstore.h)
- [Pipeline protocol](../ds4_distributed.c) and [TP protocol](../ds4_tp.c)

For changes to state handling, cover rewind/replay, save/load, images, and
multiple sessions as well as a fresh prompt. For distributed changes, exercise
both ranks and failures; a local command-parser test is not physical TP QA.
