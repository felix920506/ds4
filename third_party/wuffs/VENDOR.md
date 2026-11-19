# Vendored Wuffs

`wuffs-v0.4.c` is the single-file C release of
[Wuffs](https://github.com/google/wuffs), copied verbatim. `ds4_image.c`
includes it and uses it to decode every image the vision models receive.

## Why Wuffs

Images arrive from HTTP clients, so the decoder parses untrusted input. Wuffs
decoders are written in a language whose compiler proves that every array
index and arithmetic operation is in bounds, then emits plain C. It needs no
build system and no other library, which keeps ds4 free of third-party build
dependencies. Its JPEG output is bit-exact with libjpeg-turbo.

## Upstream pin

| Field | Value |
| --- | --- |
| Source | https://github.com/google/wuffs |
| Tag | `v0.4.0-alpha.9` (commit `3387e19512ba52957f29fa3a357f11360b415b08`) |
| File | `release/c/wuffs-v0.4.c` |
| SHA-256 | `51d2f73b757adfab105386fd3641aebd69dabc3a19126d5e060b6996ede94714` |
| License | Apache-2.0 OR MIT (`LICENSE`, `LICENSE-APACHE`, `LICENSE-MIT`) |

## What ds4 compiles

`ds4_image.c` defines `WUFFS_CONFIG__MODULES` and enables only the PNG, JPEG,
GIF, BMP and WebP decoders with the modules they depend on. Everything is
`static` (`WUFFS_CONFIG__STATIC_FUNCTIONS`), so no Wuffs symbol leaves
`ds4_image.o`.

This release decodes lossless WebP only. Lossy and animated WebP are rejected
with a message that says so.

## Updating

Replace `wuffs-v0.4.c` with the same file from a newer tag, update the table,
and run `make tests/test_image_decode && ./tests/test_image_decode`. Do not
edit the file.
