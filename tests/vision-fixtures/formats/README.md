# Image format fixtures

`tests/test_image_decode` (part of `make test`) decodes every file here and
compares the pixel fingerprint with one computed from Pillow's decode of the
same file, so the expected values do not depend on the ds4 decoder.

All files show the same 97x61 test pattern.

| File | What it covers |
| --- | --- |
| `rgba16.png`, `rgb16.png` | 16 bits per sample. ImageMagick writes this by default. |
| `rgb16_adam7.png`, `rgba8_adam7.png` | Adam7 interlacing |
| `gray16.png`, `gray4.png`, `gray1_adam7.png` | Grayscale at 16, 4 and 1 bits |
| `pal4_adam7.png` | Interlaced palette |
| `gif_static.gif`, `gif_animated.gif` | GIF. Only the first frame is decoded. |
| `bmp_24.bmp` | BMP |
| `webp_lossless.webp` | Lossless WebP |
| `webp_lossy.webp` | Lossy WebP, rejected with a message that says so |
| `jpeg_cmyk.jpg` | CMYK JPEG |
| `jpeg_exif_rot6.jpg` | EXIF orientation 6: stored 97x61, shown 61x97 |
| `truncated.png` | First half of `rgba16.png`, rejected as truncated |

The PNGs were written by ImageMagick 6.9.12 (`convert -depth 16 PNG64:...`,
`-interlace PNG`, and so on), the rest by Pillow 12.3. A 16-bit sample decodes
to its high byte.
