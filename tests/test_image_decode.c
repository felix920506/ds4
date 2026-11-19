#include "ds4_image.h"

#include <stdio.h>
#include <string.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_image(const char *path, uint32_t width, uint32_t height,
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

/* Decoding must fail, and the message must say why. */
static int check_rejected_with(const char *path, const char *needle) {
    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "%s: was accepted\n", path);
        ds4_image_free(&image);
        return 0;
    }
    if (!strstr(error, needle)) {
        fprintf(stderr, "%s: error \"%s\" does not mention \"%s\"\n",
                path, error, needle);
        return 0;
    }
    return 1;
}

/* tests/vision-fixtures/formats: the fingerprints come from Pillow's pixels,
 * not from this decoder. See the README there. */
#define SOURCE_FP "8c6661da841e2cddbeee3b61d789b3fa2f13bf109bb1e8184a06ab526c826d55"
#define GIF_FP "56db6ab6694300a7ed767aedf3fdecfa61a0dcdde324c02345ec0f789b499c07"
static const struct {
    const char *name;
    uint32_t width, height;
    const char *fingerprint;
} format_cases[] = {
    /* PNG layouts the previous decoder rejected (issue #13). */
    {"rgba16.png", 97u, 61u, SOURCE_FP},
    {"rgb16.png", 97u, 61u, SOURCE_FP},
    {"rgb16_adam7.png", 97u, 61u, SOURCE_FP},
    {"rgba8_adam7.png", 97u, 61u, SOURCE_FP},
    {"gray16.png", 97u, 61u,
     "9db06066699878ce30ad2ebb3bd1e85faab9e356709192d0055c5a36863da2a7"},
    {"gray4.png", 97u, 61u,
     "e29de2dbf88fc0a5a24338bc139f04f1b083eda523555412768f673a31f15f42"},
    {"gray1_adam7.png", 97u, 61u,
     "b5cd02a61b6a35839f0f7a845676ede4f89f28b7c0cd2f5ba8b52652a5db8719"},
    {"pal4_adam7.png", 97u, 61u,
     "0fafa496209ec50f09a3f60adfd2a91b6df53f572b4ed078a8ec92eea3179fde"},
    /* Other formats. An animated GIF decodes to its first frame. */
    {"gif_static.gif", 97u, 61u, GIF_FP},
    {"gif_animated.gif", 97u, 61u, GIF_FP},
    {"bmp_24.bmp", 97u, 61u, SOURCE_FP},
    {"webp_lossless.webp", 97u, 61u, SOURCE_FP},
    /* JPEG: CMYK, and EXIF orientation 6 (stored 97x61, shown 61x97). */
    {"jpeg_cmyk.jpg", 97u, 61u,
     "15a8a3a7a04694f7c0a865119e7e8db39f70cb2f369a0d83274b139748f83372"},
    {"jpeg_exif_rot6.jpg", 61u, 97u,
     "8a059513e5b37670590912735c5ed88e9ffe53c7afc641a65ab516853371b240"},
};

static int check_formats(void) {
    const char *dir = "tests/vision-fixtures/formats";
    char path[256];
    for (size_t i = 0; i < sizeof(format_cases) / sizeof(format_cases[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, format_cases[i].name);
        if (!check_image(path, format_cases[i].width, format_cases[i].height,
                         format_cases[i].fingerprint))
            return 0;
    }
    snprintf(path, sizeof(path), "%s/truncated.png", dir);
    if (!check_rejected_with(path, "PNG image: truncated input")) return 0;
    snprintf(path, sizeof(path), "%s/webp_lossy.webp", dir);
    if (!check_rejected_with(path, "lossy and animated WebP")) return 0;

    static const uint8_t tiff[] = {'I', 'I', 42, 0, 8, 0, 0, 0};
    ds4_image image = {0};
    char error[160] = {0};
    if (ds4_image_decode_memory(&image, tiff, sizeof(tiff), error, sizeof(error)) ||
        !strstr(error, "image must be")) {
        fprintf(stderr, "TIFF header: accepted or wrong error \"%s\"\n", error);
        ds4_image_free(&image);
        return 0;
    }
    return 1;
}

int main(void) {
    if (!check_formats()) return 1;
    /* 24x16 grayscale progressive JPEG with ZRL in an AC refinement scan.
     * The decode must match libjpeg-turbo djpeg bit-exactly. */
    if (!check_image("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                    24u, 16u,
                    "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc"))
        return 1;
    /* 64x48 4:2:0 progressive JPEG, same scan structure. The decode must
     * match libjpeg-turbo djpeg bit-exactly. */
    if (!check_image("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                    64u, 48u,
                    "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4"))
        return 1;
    /* Progressive AC scan with Se=200 (issue #1292): a decoder that trusts
     * it writes outside the coefficient block. */
    if (!check_jpeg_rejected("tests/vision-fixtures/jpeg/prog_sos_se_oob.jpg"))
        return 1;
    /* DC scans must have Se == 0. */
    if (!check_jpeg_sos_rejected(0, 5)) return 1;
    return 0;
}
