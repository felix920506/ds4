#include "ds4_image.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_jpeg(const char *path, uint32_t width, uint32_t height,
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

/* ---- PNG variants the built-in decoder rejects (issue #13) ---------------
 * Built in memory with a stored (uncompressed) deflate stream, so the test
 * needs no zlib and no fixture files. */

static uint32_t crc32_png(const uint8_t *p, size_t n, uint32_t crc) {
    crc = ~crc;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static void put_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
}

static size_t png_chunk(uint8_t *out, const char *type, const uint8_t *data, uint32_t len) {
    put_be32(out, len);
    memcpy(out + 4, type, 4);
    if (len) memcpy(out + 8, data, len);
    put_be32(out + 8 + len, crc32_png(out + 4, 4 + (size_t)len, 0));
    return 12 + (size_t)len;
}

/* `raw` is the filtered scanline stream (filter type 0 on every row). */
static size_t png_build(uint8_t *out, uint32_t w, uint32_t h, uint8_t depth,
                        uint8_t color, uint8_t interlace,
                        const uint8_t *raw, uint16_t raw_len) {
    size_t n = 0;
    memcpy(out, "\x89PNG\r\n\x1a\n", 8); n = 8;
    uint8_t ihdr[13];
    put_be32(ihdr, w); put_be32(ihdr + 4, h);
    ihdr[8] = depth; ihdr[9] = color; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = interlace;
    n += png_chunk(out + n, "IHDR", ihdr, sizeof(ihdr));
    uint8_t z[512];
    size_t zn = 0;
    z[zn++] = 0x78; z[zn++] = 0x01;
    z[zn++] = 0x01; /* final stored block */
    z[zn++] = (uint8_t)raw_len; z[zn++] = (uint8_t)(raw_len >> 8);
    z[zn++] = (uint8_t)~raw_len; z[zn++] = (uint8_t)(~raw_len >> 8);
    memcpy(z + zn, raw, raw_len); zn += raw_len;
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw_len; i++) { a = (a + raw[i]) % 65521u; b = (b + a) % 65521u; }
    put_be32(z + zn, (b << 16) | a); zn += 4;
    n += png_chunk(out + n, "IDAT", z, (uint32_t)zn);
    n += png_chunk(out + n, "IEND", NULL, 0);
    return n;
}

typedef struct {
    const char *name;
    uint8_t png[1024];
    size_t len;
    uint32_t width, height;
    uint8_t rgb[64];
    const char *reason; /* expected in the error with the fallback off */
} png_case;

static int build_png_cases(png_case *c) {
    /* 2x1 RGB, 16 bits per sample: v * 257 maps exactly onto 8-bit v. */
    static const uint8_t rgb16[] = {0, 0x0a,0x0a, 0x14,0x14, 0x1e,0x1e,
                                       0xc8,0xc8, 0x64,0x64, 0xff,0xff};
    c[0] = (png_case){.name = "rgb16", .width = 2, .height = 1,
                      .rgb = {10, 20, 30, 200, 100, 255}, .reason = "bit depth 16"};
    c[0].len = png_build(c[0].png, 2, 1, 16, 2, 0, rgb16, sizeof(rgb16));
    /* 8x1 gray, 1 bit per sample. */
    static const uint8_t gray1[] = {0, 0xb2}; /* 1011 0010 */
    c[1] = (png_case){.name = "gray1", .width = 8, .height = 1, .reason = "bit depth 1"};
    for (int i = 0; i < 8; i++) {
        uint8_t v = (0xb2 >> (7 - i)) & 1 ? 255 : 0;
        c[1].rgb[i * 3] = c[1].rgb[i * 3 + 1] = c[1].rgb[i * 3 + 2] = v;
    }
    c[1].len = png_build(c[1].png, 8, 1, 1, 0, 0, gray1, sizeof(gray1));
    /* 2x2 RGB, Adam7: pass 1 holds (0,0), pass 6 (1,0), pass 7 row 1. */
    static const uint8_t adam7[] = {0, 1, 2, 3,
                                    0, 4, 5, 6,
                                    0, 7, 8, 9, 10, 11, 12};
    c[2] = (png_case){.name = "adam7", .width = 2, .height = 2,
                      .rgb = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                      .reason = "Adam7"};
    c[2].len = png_build(c[2].png, 2, 2, 8, 2, 1, adam7, sizeof(adam7));
    return 3;
}

static int check_png_reason(const png_case *c) {
    ds4_image image = {0};
    char error[384] = {0};
    if (ds4_image_decode_memory(&image, c->png, c->len, error, sizeof(error))) {
        fprintf(stderr, "%s: built-in decoder unexpectedly accepted it\n", c->name);
        ds4_image_free(&image);
        return 0;
    }
    if (!strstr(error, c->reason) || !strstr(error, "install ImageMagick")) {
        fprintf(stderr, "%s: error lacks \"%s\" or the install hint: %s\n",
                c->name, c->reason, error);
        return 0;
    }
    return 1;
}

/* Returns 1 on a pixel-exact decode, 0 on a mismatch, -1 when the fallback
 * reported something matching `skip_on` (ImageMagick missing). */
static int check_png_decoded(const png_case *c) {
    ds4_image image = {0};
    char error[384] = {0};
    if (!ds4_image_decode_memory(&image, c->png, c->len, error, sizeof(error))) {
        if (strstr(error, "install ImageMagick")) return -1;
        fprintf(stderr, "%s: fallback decode failed: %s\n", c->name, error);
        return 0;
    }
    int ok = image.width == c->width && image.height == c->height &&
             !memcmp(image.rgb, c->rgb, (size_t)c->width * c->height * 3);
    if (!ok) {
        fprintf(stderr, "%s: got %ux%u", c->name, image.width, image.height);
        for (uint32_t i = 0; i < image.width * image.height * 3 && i < 24; i++)
            fprintf(stderr, " %u", image.rgb[i]);
        fprintf(stderr, "\n");
    }
    ds4_image_free(&image);
    return ok;
}

/* A stand-in decoder: a shell script, so the spawn/pipe/PNM path is tested
 * on machines without ImageMagick. */
static int write_script(char *path, size_t cap, const char *body) {
    snprintf(path, cap, "/tmp/ds4_image_decoder_XXXXXX");
    int fd = mkstemp(path);
    if (fd < 0) return 0;
    FILE *fp = fdopen(fd, "w");
    if (!fp) { close(fd); return 0; }
    fprintf(fp, "#!/bin/sh\n%s\n", body);
    fclose(fp);
    return chmod(path, 0700) == 0;
}

static int check_fake_decoder(const char *body, const uint8_t *in, size_t in_len,
                              const uint8_t *want, uint32_t w, uint32_t h,
                              const char *want_error) {
    char path[64];
    if (!write_script(path, sizeof(path), body)) {
        fprintf(stderr, "cannot write a decoder script\n");
        return 0;
    }
    setenv("DS4_IMAGE_DECODER", path, 1);
    ds4_image image = {0};
    char error[384] = {0};
    int got = ds4_image_decode_memory(&image, in, in_len, error, sizeof(error));
    unlink(path);
    int ok;
    if (want) {
        ok = got && image.width == w && image.height == h &&
             !memcmp(image.rgb, want, (size_t)w * h * 3);
        if (!ok) fprintf(stderr, "fake decoder `%s`: got=%d %ux%u err=%s\n",
                         body, got, image.width, image.height, error);
    } else {
        ok = !got && strstr(error, want_error) != NULL;
        if (!ok) fprintf(stderr, "fake decoder `%s`: got=%d err=%s (want \"%s\")\n",
                         body, got, error, want_error);
    }
    ds4_image_free(&image);
    return ok;
}

static int check_fallback(void) {
    png_case cases[3];
    int n = build_png_cases(cases);

    setenv("DS4_IMAGE_DECODER", "off", 1);
    for (int i = 0; i < n; i++)
        if (!check_png_reason(&cases[i])) return 0;

    /* Stand-in decoders: output parsing, 16-bit PNM scaling, comments,
     * failure exits, a child that never reads its stdin, bad headers. */
    const uint8_t *png = cases[0].png;
    size_t png_len = cases[0].len;
    static const uint8_t two[] = {1, 2, 3, 4, 5, 6};
    if (!check_fake_decoder("cat >/dev/null; printf 'P6\\n# comment\\n2 1\\n255\\n\\001\\002\\003\\004\\005\\006'",
                            png, png_len, two, 2, 1, NULL)) return 0;
    static const uint8_t gray16[] = {255, 255, 255, 128, 128, 128};
    if (!check_fake_decoder("cat >/dev/null; printf 'P5 2 1 65535\\n\\377\\377\\200\\200'",
                            png, png_len, gray16, 2, 1, NULL)) return 0;
    if (!check_fake_decoder("cat >/dev/null; echo 'no such coder' >&2; exit 1",
                            png, png_len, NULL, 0, 0, "no such coder")) return 0;
    if (!check_fake_decoder("cat >/dev/null; printf 'P6\\n99999 1\\n255\\n'",
                            png, png_len, NULL, 0, 0, "no usable image")) return 0;
    if (!check_fake_decoder("cat >/dev/null; printf 'P6\\n4 4\\n255\\n\\001'",
                            png, png_len, NULL, 0, 0, "no usable image")) return 0;
    {
        /* Issue #57: a child that closes stdout and stderr and keeps
         * running must still be cut off at the deadline, not waited for. */
        setenv("DS4_IMAGE_EXTERNAL_TIMEOUT_MS", "1000", 1);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int ok = check_fake_decoder("cat >/dev/null; exec sleep 6 >&- 2>&-",
                                    png, png_len, NULL, 0, 0, "timed out after 1 s");
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        unsetenv("DS4_IMAGE_EXTERNAL_TIMEOUT_MS");
        if (secs > 3.0) {
            fprintf(stderr, "closed-output decoder was waited for %.1f s\n", secs);
            ok = 0;
        }
        if (!ok) return 0;
    }
    {
        /* 8 MiB of input to a child that exits at once: the write side
         * hits EPIPE, which must not kill this process with SIGPIPE. */
        size_t big_len = (size_t)8 << 20;
        uint8_t *big = calloc(1, big_len);
        if (!big) return 0;
        memcpy(big, png, png_len);
        int ok = check_fake_decoder("exit 3", big, big_len, NULL, 0, 0, "status 3");
        free(big);
        if (!ok) return 0;
    }
    {
        /* Issue #42: exceeding ImageMagick's area limit spills the pixel
         * cache to a temp file, and a child killed for overrunning
         * out_cap cannot remove it.  The fallback must bound the spill,
         * aim it at a private directory (define beats MAGICK_TEMPORARY_
         * PATH, MAGICK_TMPDIR and TMPDIR alike, so the env overrides go)
         * and delete that directory once the child is gone.  Fake
         * decoder: checks the limits and defines, records where it was
         * told to spill and what TMPDIR it got, creates a spill file,
         * then writes past the output cap so it gets killed. */
        char probe[64];
        snprintf(probe, sizeof(probe), "/tmp/ds4_img_probe_%d", (int)getpid());
        char body[1024];
        snprintf(body, sizeof(body),
                 "cat >/dev/null\n"
                 "for a in \"$@\"; do case \"$a\" in registry:temporary-path=*/ds4-img-*) spill=1;; esac; done\n"
                 "[ \"$spill\" = 1 ] || { echo 'spill is not aimed at the private dir' >&2; exit 1; }\n"
                 "case \" $* \" in *\" -limit disk 2GiB \"*) : ;; *) echo 'spill is not bounded' >&2; exit 1 ;; esac\n"
                 "case \" $* \" in *\" -limit area %u \"*) : ;; *) echo 'area below the pixel ceiling' >&2; exit 1 ;; esac\n"
                 "[ -z \"${MAGICK_TEMPORARY_PATH:-}\" ] || { echo 'MAGICK_TEMPORARY_PATH reaches the child' >&2; exit 1; }\n"
                 "[ -z \"${MAGICK_TMPDIR:-}\" ] || { echo 'MAGICK_TMPDIR reaches the child' >&2; exit 1; }\n"
                 "echo \"${TMPDIR:-}\" > %s\n"
                 "case \"${TMPDIR:-}\" in */ds4-img-*) : ;; *) echo 'TMPDIR is not the private dir' >&2; exit 1 ;; esac\n"
                 ": > \"$TMPDIR/magick-spill\"; mkdir -p \"$TMPDIR/nested\"; "
                 "head -c 220000000 /dev/zero",
                 DS4_IMAGE_MAX_PIXELS + 1u, probe);
        char path[64];
        if (!write_script(path, sizeof(path), body)) return 0;
        setenv("DS4_IMAGE_DECODER", path, 1);
        setenv("MAGICK_TEMPORARY_PATH", "/tmp", 1);
        setenv("MAGICK_TMPDIR", "/tmp", 1);
        ds4_image image = {0};
        char error[384] = {0};
        int got = ds4_image_decode_memory(&image, png, png_len, error, sizeof(error));
        unlink(path);
        unsetenv("MAGICK_TEMPORARY_PATH");
        unsetenv("MAGICK_TMPDIR");
        ds4_image_free(&image);
        int ok = !got && strstr(error, "exceeds the size limit") != NULL;
        char dir[128] = {0};
        if (ok) {
            FILE *fp = fopen(probe, "r");
            if (fp) {
                if (fgets(dir, sizeof(dir), fp)) dir[strcspn(dir, "\n")] = '\0';
                fclose(fp);
                unlink(probe);
                ok = dir[0] && access(dir, F_OK) != 0;
            } else ok = 0;
            if (!ok) fprintf(stderr, "killed decoder left %s behind\n", dir);
        } else {
            fprintf(stderr, "spill test: got=%d err=%s\n", got, error);
            unlink(probe);
        }
        if (!ok) return 0;

        /* Fail-closed setup: an unusable TMPDIR refuses the fallback
         * rather than spawning a child whose spill nobody cleans up. */
        char ok_body[128];
        snprintf(ok_body, sizeof(ok_body),
                 "printf 'P6\\n1 1\\n255\\n\\001\\002\\003'");
        char ok_path[64];
        if (!write_script(ok_path, sizeof(ok_path), ok_body)) return 0;
        setenv("DS4_IMAGE_DECODER", ok_path, 1);
        setenv("TMPDIR", "/nonexistent/ds4-test-dir", 1);
        {
            ds4_image image = {0};
            char error[384] = {0};
            int g = ds4_image_decode_memory(&image, png, png_len, error, sizeof(error));
            ds4_image_free(&image);
            if (g || !strstr(error, "temp directory")) {
                fprintf(stderr, "unusable TMPDIR: got=%d err=%s\n", g, error);
                return 0;
            }
        }
        unsetenv("TMPDIR");
        unlink(ok_path);

        /* An oversized environment also refuses the fallback, before any
         * fd or spawn object exists, and a thousand such failures leak
         * nothing. */
        {
            char names[130][16], values[130][8];
            for (int i = 0; i < 130; i++) {
                snprintf(names[i], sizeof(names[i]), "DS4_TENV%d", i);
                snprintf(values[i], sizeof(values[i]), "%d", i);
                setenv(names[i], values[i], 1);
            }
            int open_fds = open("/dev/null", O_RDONLY);
            close(open_fds);
            ds4_image image = {0};
            char error[384] = {0};
            int g = ds4_image_decode_memory(&image, png, png_len, error, sizeof(error));
            ds4_image_free(&image);
            if (g || !strstr(error, "environment too large")) {
                fprintf(stderr, "oversized environment: got=%d err=%s\n", g, error);
                return 0;
            }
            for (int i = 0; i < 1000; i++) {
                ds4_image im2 = {0};
                ds4_image_decode_memory(&im2, png, png_len, error, sizeof(error));
                ds4_image_free(&im2);
            }
            int after = open("/dev/null", O_RDONLY);
            int leaked = after != open_fds; /* lowest free fd unchanged */
            close(after);
            for (int i = 0; i < 130; i++) unsetenv(names[i]);
            if (leaked) { fprintf(stderr, "failed decodes leaked fds\n"); return 0; }
        }
    }

    setenv("DS4_IMAGE_DECODER", "/nonexistent/magick", 1);
    {
        ds4_image image = {0};
        char error[384] = {0};
        if (ds4_image_decode_memory(&image, png, png_len, error, sizeof(error)) ||
            !strstr(error, "fallback failed")) {
            fprintf(stderr, "missing decoder program: %s\n", error);
            ds4_image_free(&image);
            return 0;
        }
    }

    /* The real thing, when ImageMagick is installed. */
    unsetenv("DS4_IMAGE_DECODER");
    int skipped = 0;
    for (int i = 0; i < n; i++) {
        int r = check_png_decoded(&cases[i]);
        if (r == 0) return 0;
        if (r < 0) skipped = 1;
    }
    if (skipped)
        fprintf(stderr, "test_image_decode: ImageMagick not found, real fallback decode skipped\n");
    return 1;
}

int main(void) {
    /* The rejection checks below are about the built-in decoder itself. */
    setenv("DS4_IMAGE_DECODER", "off", 1);
    /* 24x16 grayscale progressive JPEG. Unpatched Iris yields different
     * pixels; patched decode matches libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                    24u, 16u,
                    "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc"))
        return 1;
    /* 64x48 4:2:0 progressive JPEG. Unpatched jpeg_load returns NULL.
     * After the ZRL fix plus main's chroma interpolation, decode matches
     * libjpeg-turbo djpeg bit-exactly. */
    if (!check_jpeg("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                    64u, 48u,
                    "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4"))
        return 1;
    /* Progressive AC scan with Se=200 (issue #1292). Unpatched Iris indexes
     * jpeg_zigzag[] past 63 and writes outside the coefficient block. */
    if (!check_jpeg_rejected("tests/vision-fixtures/jpeg/prog_sos_se_oob.jpg"))
        return 1;
    /* DC scans must have Se == 0. */
    if (!check_jpeg_sos_rejected(0, 5)) return 1;
    if (!check_fallback()) return 1;
    return 0;
}
