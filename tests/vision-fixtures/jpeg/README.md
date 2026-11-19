# Progressive JPEG AC-refinement ZRL fixtures

These images exercise successive-approximation AC refinement (`Ss=1, Se=63,
Ah=2, Al=1`) with restart markers. The Iris decoder ds4 used before Wuffs mishandled ZRL
(`run=15, size=0`) in that scan: it kept consuming correction bits after the
16th currently-zero coefficient.

Generated with libjpeg-turbo 3.2.0 `cjpeg`. Re-run `./generate.sh` if you have
`cjpeg` on `PATH`.

| File | Size | Why it is here |
| --- | --- | --- |
| `prog_ac_refine_zrl_gray.jpg` | 24x16 grayscale | Iris decoded it to different pixels than libjpeg-turbo. The decode must be bit-exact with `djpeg`. |
| `prog_ac_refine_zrl_420.jpg` | 64x48 YCbCr 4:2:0 | Iris rejected it. The decode must be bit-exact with `djpeg`. |

`make test` covers both through `tests/test_image_decode`.
