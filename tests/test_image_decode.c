#include "ds4_image.h"

#include <stdio.h>
#include <string.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_decode(const char *path, uint32_t width, uint32_t height,
                      const char *expected_fp) {
    ds4_image image = {0};
    char error[160] = {0};
    if (!ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        return 0;
    }
    char got[65] = {0};
    int ok = hex_fingerprint(image.fingerprint, got, sizeof(got)) &&
             image.width == width && image.height == height &&
             strcmp(got, expected_fp) == 0;
    if (!ok) {
        fprintf(stderr, "%s: got %ux%u fp=%s, expected %ux%u fp=%s\n",
                path, image.width, image.height, got,
                width, height, expected_fp);
    }
    ds4_image_free(&image);
    return ok;
}

/* PNG variants that IRIS used to reject (issue #13). Each set decodes to the
 * same pixels as its 8-bit source: RGB, 4-level gray, or black/white. The
 * fixtures are 13x11 so Adam7 has partial passes and sub-byte rows pad. */
static const struct {
    const char *name;
    uint32_t width, height;
    const char *fp;
} png_cases[] = {
    {"rgb16.png", 13u, 11u, "622047850ba6198c5bedf6b4dd5a99ec0078d5a65e40f5e1a35e3d60ccc25efb"},
    {"rgba16.png", 13u, 11u, "622047850ba6198c5bedf6b4dd5a99ec0078d5a65e40f5e1a35e3d60ccc25efb"},
    {"gray16.png", 13u, 11u, "75194fa2300eeca09e7a06cb05e18c9d0bbc7b8d1e8877dd70abbf8b1689868f"},
    {"graya16.png", 13u, 11u, "75194fa2300eeca09e7a06cb05e18c9d0bbc7b8d1e8877dd70abbf8b1689868f"},
    {"gray4.png", 13u, 11u, "75194fa2300eeca09e7a06cb05e18c9d0bbc7b8d1e8877dd70abbf8b1689868f"},
    {"gray2.png", 13u, 11u, "75194fa2300eeca09e7a06cb05e18c9d0bbc7b8d1e8877dd70abbf8b1689868f"},
    {"gray1.png", 13u, 11u, "f10f1fc9c01dbaeac6df50be035855f098b4e7735d20791ea3248052303ac6e3"},
    {"rgb8_adam7.png", 13u, 11u, "622047850ba6198c5bedf6b4dd5a99ec0078d5a65e40f5e1a35e3d60ccc25efb"},
    {"rgba8_adam7.png", 13u, 11u, "622047850ba6198c5bedf6b4dd5a99ec0078d5a65e40f5e1a35e3d60ccc25efb"},
    {"rgba16_adam7.png", 13u, 11u, "622047850ba6198c5bedf6b4dd5a99ec0078d5a65e40f5e1a35e3d60ccc25efb"},
    {"gray1_adam7.png", 13u, 11u, "f10f1fc9c01dbaeac6df50be035855f098b4e7735d20791ea3248052303ac6e3"},
    {"gray2_adam7.png", 13u, 11u, "75194fa2300eeca09e7a06cb05e18c9d0bbc7b8d1e8877dd70abbf8b1689868f"},
};

/* Rejected inputs must say why, so API clients can tell a corrupt upload
 * from a size limit or an unsupported encoding. */
static int check_error(const char *path, const uint8_t *data, size_t len,
                       const char *expected) {
    ds4_image image = {0};
    char error[160] = {0};
    int ok = path ? ds4_image_decode_file(&image, path, error, sizeof(error))
                  : ds4_image_decode_memory(&image, data, len, error, sizeof(error));
    if (ok) {
        fprintf(stderr, "%s: invalid image was accepted\n", path ? path : "memory");
        ds4_image_free(&image);
        return 0;
    }
    if (!strstr(error, expected)) {
        fprintf(stderr, "%s: error \"%s\" does not contain \"%s\"\n",
                path ? path : "memory", error, expected);
        return 0;
    }
    return 1;
}

static const struct {
    const char *name;
    const char *expected;
} error_cases[] = {
    {"png_truncated.png", "corrupt PNG: IDAT chunk extends past the end of the file (file truncated?)"},
    {"png_bad_crc.png", "corrupt PNG: CRC mismatch in IDAT chunk"},
    {"png_too_large.png", "PNG image too large: 20000x8 exceeds the 16384 pixel limit per side"},
    {"jpeg_truncated.jpg", "corrupt JPEG: image data ends early (file truncated?)"},
    {"jpeg_cmyk.jpg", "unsupported JPEG: CMYK"},
    {"jpeg_arithmetic.jpg", "unsupported JPEG: arithmetic coding"},
    {"jpeg_12bit.jpg", "unsupported JPEG: 12-bit sample precision"},
    {"jpeg_too_large.jpg", "JPEG image too large: 32x20000 exceeds the 16384 pixel limit per side"},
};

static int check_jpeg_rejected(const char *path) {
    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "%s: malformed JPEG was accepted\n", path);
        ds4_image_free(&image);
        return 0;
    }
    return 1;
}

/* 1x1 progressive grayscale JPEG with a single scan whose SOS carries the
 * given Ss/Se. Used for scan headers that must be rejected. */
static int check_jpeg_sos_rejected(uint8_t ss, uint8_t se) {
    uint8_t b[128];
    size_t n = 0;
    static const uint8_t sof2[] = {0xFF, 0xC2, 0x00, 0x0B, 0x08, 0x00, 0x01,
                                   0x00, 0x01, 0x01, 0x01, 0x11, 0x00};
    b[n++] = 0xFF; b[n++] = 0xD8;
    b[n++] = 0xFF; b[n++] = 0xDB; b[n++] = 0x00; b[n++] = 0x43; b[n++] = 0x00;
    for (int i = 0; i < 64; i++) b[n++] = 0x01;
    memcpy(b + n, sof2, sizeof(sof2)); n += sizeof(sof2);
    /* DC table 0: one 1-bit code for category 0. */
    b[n++] = 0xFF; b[n++] = 0xC4; b[n++] = 0x00; b[n++] = 0x14; b[n++] = 0x00;
    b[n++] = 0x01;
    for (int i = 1; i < 16; i++) b[n++] = 0x00;
    b[n++] = 0x00;
    b[n++] = 0xFF; b[n++] = 0xDA; b[n++] = 0x00; b[n++] = 0x08; b[n++] = 0x01;
    b[n++] = 0x01; b[n++] = 0x00; b[n++] = ss; b[n++] = se; b[n++] = 0x00;
    b[n++] = 0x00;
    b[n++] = 0xFF; b[n++] = 0xD9;

    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_memory(&image, b, n, error, sizeof(error))) {
        fprintf(stderr, "SOS Ss=%u Se=%u was accepted\n", ss, se);
        ds4_image_free(&image);
        return 0;
    }
    return 1;
}

int main(void) {
    /* 24x16 grayscale progressive JPEG. Unpatched Iris yields different
     * pixels; patched decode matches libjpeg-turbo djpeg bit-exactly. */
    if (!check_decode("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                    24u, 16u,
                    "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc"))
        return 1;
    /* 64x48 4:2:0 progressive JPEG. Unpatched jpeg_load returns NULL.
     * After the ZRL fix plus main's chroma interpolation, decode matches
     * libjpeg-turbo djpeg bit-exactly. */
    if (!check_decode("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                    64u, 48u,
                    "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4"))
        return 1;
    /* Progressive AC scan with Se=200 (issue #1292). Unpatched Iris indexes
     * jpeg_zigzag[] past 63 and writes outside the coefficient block. */
    if (!check_jpeg_rejected("tests/vision-fixtures/jpeg/prog_sos_se_oob.jpg"))
        return 1;
    /* DC scans must have Se == 0. */
    if (!check_jpeg_sos_rejected(0, 5)) return 1;
    for (size_t i = 0; i < sizeof(png_cases) / sizeof(png_cases[0]); i++) {
        char path[256];
        snprintf(path, sizeof(path), "tests/vision-fixtures/png/%s",
                 png_cases[i].name);
        if (!check_decode(path, png_cases[i].width, png_cases[i].height,
                        png_cases[i].fp))
            return 1;
    }
    for (size_t i = 0; i < sizeof(error_cases) / sizeof(error_cases[0]); i++) {
        char path[256];
        snprintf(path, sizeof(path), "tests/vision-fixtures/errors/%s",
                 error_cases[i].name);
        if (!check_error(path, NULL, 0, error_cases[i].expected)) return 1;
    }
    if (!check_error("tests/vision-fixtures/jpeg/prog_sos_se_oob.jpg", NULL, 0,
                     "corrupt JPEG: invalid progressive scan parameters"))
        return 1;
    static const uint8_t webp[] = "RIFF\x10\0\0\0WEBPVP8 ";
    if (!check_error(NULL, webp, sizeof(webp) - 1,
                     "unsupported image format: WebP (only JPEG and PNG are accepted)"))
        return 1;
    static const uint8_t gif[] = "GIF89a\x01\0\x01\0";
    if (!check_error(NULL, gif, sizeof(gif) - 1, "unsupported image format: GIF"))
        return 1;
    static const uint8_t junk[] = "hello world";
    if (!check_error(NULL, junk, sizeof(junk) - 1,
                     "unrecognized image data: not a JPEG or PNG file"))
        return 1;
    if (!check_error(NULL, junk, 0, "image is empty")) return 1;
    return 0;
}
