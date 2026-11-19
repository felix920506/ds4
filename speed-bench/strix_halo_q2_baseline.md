# Strix Halo Q2 baseline

Measured 2026-11-14 on AMD Ryzen AI Max+ 395 with Radeon 8060S Graphics
(gfx1151), 128 GB shared memory, ROCm Docker image
`ghcr.io/felix920506/ds4:dev-rocm` (digest
`sha256:26808d103d6581990d5d50151f8fedcaf72c65720730ee6ad2b3b9f714ba1d3a`).
Both runs used the published image binaries, full resident model loading, the
same public-domain prompt, one 2,048-token prefill frontier, context allocation
of 2,177, and 128 greedy decode tokens. No vision input was provided in the
Vision Experimental run; this measures its text path.

| Checkpoint | Prefill | Decode | CSV |
| --- | ---: | ---: | --- |
| DeepSeek V4 Flash 0731 Q2 | 338.04 tok/s | 14.24 tok/s | [strix_halo_v4_flash_0731_q2.csv](strix_halo_v4_flash_0731_q2.csv) |
| DeepSeek V4 Flash Vision Experimental Q2 | 351.55 tok/s | 14.32 tok/s | [strix_halo_v4_vision_exp_q2.csv](strix_halo_v4_vision_exp_q2.csv) |

The Vision Experimental model's matching encoder was downloaded too. The
benchmark command for each was:

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$PWD/gguf:/models:ro" -v "$PWD/speed-bench:/bench:ro" \
  -v "$PWD/speed-bench:/out:rw" \
  --entrypoint /opt/ds4/ds4-bench ghcr.io/felix920506/ds4:dev-rocm \
  --rocm -m /models/MODEL.gguf \
  --prompt-file /bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2177 --gen-tokens 128 \
  --csv /out/RESULT.csv
```

Model files are in `gguf/`. The main checkpoints are each 86,720,111,xxx bytes;
the vision encoder is 932,857,760 bytes. The downloader verified model sizes
and checksums.
