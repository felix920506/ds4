#include "ds4_image.h"

#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define JPEG_IMPLEMENTATION
#define PNG_IMPLEMENTATION
#include "third_party/iris/jpeg.h"
#include "third_party/iris/png.h"

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} ds4_sha256;

static const uint32_t ds4_sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static uint32_t ds4_rotr32(uint32_t value, unsigned bits) {
    return (value >> bits) | (value << (32u - bits));
}

static void ds4_sha256_block(ds4_sha256 *sha, const uint8_t block[64]) {
    uint32_t w[64];
    for (unsigned i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (unsigned i = 16; i < 64; i++) {
        uint32_t s0 = ds4_rotr32(w[i - 15], 7) ^ ds4_rotr32(w[i - 15], 18) ^
                      (w[i - 15] >> 3);
        uint32_t s1 = ds4_rotr32(w[i - 2], 17) ^ ds4_rotr32(w[i - 2], 19) ^
                      (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = sha->state[0], b = sha->state[1];
    uint32_t c = sha->state[2], d = sha->state[3];
    uint32_t e = sha->state[4], f = sha->state[5];
    uint32_t g = sha->state[6], h = sha->state[7];
    for (unsigned i = 0; i < 64; i++) {
        uint32_t s1 = ds4_rotr32(e, 6) ^ ds4_rotr32(e, 11) ^ ds4_rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + ds4_sha256_k[i] + w[i];
        uint32_t s0 = ds4_rotr32(a, 2) ^ ds4_rotr32(a, 13) ^ ds4_rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    sha->state[0] += a; sha->state[1] += b;
    sha->state[2] += c; sha->state[3] += d;
    sha->state[4] += e; sha->state[5] += f;
    sha->state[6] += g; sha->state[7] += h;
}

static void ds4_sha256_init(ds4_sha256 *sha) {
    static const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    memcpy(sha->state, initial, sizeof(initial));
    sha->bytes = 0;
    sha->used = 0;
}

static void ds4_sha256_update(ds4_sha256 *sha, const void *data, size_t len) {
    const uint8_t *p = data;
    sha->bytes += len;
    while (len) {
        size_t take = 64 - sha->used;
        if (take > len) take = len;
        memcpy(sha->block + sha->used, p, take);
        sha->used += take;
        p += take;
        len -= take;
        if (sha->used == 64) {
            ds4_sha256_block(sha, sha->block);
            sha->used = 0;
        }
    }
}

static void ds4_sha256_final(ds4_sha256 *sha, uint8_t out[32]) {
    uint64_t bits = sha->bytes * 8u;
    sha->block[sha->used++] = 0x80;
    if (sha->used > 56) {
        memset(sha->block + sha->used, 0, 64 - sha->used);
        ds4_sha256_block(sha, sha->block);
        sha->used = 0;
    }
    memset(sha->block + sha->used, 0, 56 - sha->used);
    for (unsigned i = 0; i < 8; i++) {
        sha->block[63 - i] = (uint8_t)(bits >> (i * 8));
    }
    ds4_sha256_block(sha, sha->block);
    for (unsigned i = 0; i < 8; i++) {
        out[i * 4] = (uint8_t)(sha->state[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(sha->state[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(sha->state[i] >> 8);
        out[i * 4 + 3] = (uint8_t)sha->state[i];
    }
}

static void ds4_image_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message);
}

static uint16_t ds4_exif_u16(const uint8_t *p, int little) {
    return little ? (uint16_t)(p[0] | (p[1] << 8))
                  : (uint16_t)((p[0] << 8) | p[1]);
}

static uint32_t ds4_exif_u32(const uint8_t *p, int little) {
    if (little) {
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
               ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int ds4_jpeg_orientation(const uint8_t *data, size_t len) {
    size_t pos = 2;
    while (pos + 4 <= len && data[pos] == 0xff) {
        uint8_t marker = data[pos + 1];
        pos += 2;
        if (marker == 0xd8 || (marker >= 0xd0 && marker <= 0xd9)) continue;
        if (pos + 2 > len) break;
        size_t segment = ((size_t)data[pos] << 8) | data[pos + 1];
        if (segment < 2 || segment > len - pos) break;
        if (marker == 0xe1 && segment >= 14 &&
            memcmp(data + pos + 2, "Exif\0\0", 6) == 0) {
            const uint8_t *tiff = data + pos + 8;
            size_t tiff_len = segment - 8;
            if (tiff_len < 8) break;
            int little = tiff[0] == 'I' && tiff[1] == 'I';
            int big = tiff[0] == 'M' && tiff[1] == 'M';
            if ((!little && !big) || ds4_exif_u16(tiff + 2, little) != 42) break;
            uint32_t ifd = ds4_exif_u32(tiff + 4, little);
            if (ifd > tiff_len || tiff_len - ifd < 2) break;
            uint16_t count = ds4_exif_u16(tiff + ifd, little);
            size_t entries = (size_t)count * 12;
            if (entries > tiff_len - ifd - 2) break;
            for (uint16_t i = 0; i < count; i++) {
                const uint8_t *entry = tiff + ifd + 2 + (size_t)i * 12;
                if (ds4_exif_u16(entry, little) == 0x0112 &&
                    ds4_exif_u16(entry + 2, little) == 3 &&
                    ds4_exif_u32(entry + 4, little) == 1) {
                    int orientation = ds4_exif_u16(entry + 8, little);
                    return orientation >= 1 && orientation <= 8 ? orientation : 1;
                }
            }
        }
        if (marker == 0xda) break;
        pos += segment;
    }
    return 1;
}

static int ds4_oriented_rgb(
        ds4_image *out,
        const uint8_t *pixels,
        uint32_t width,
        uint32_t height,
        uint32_t channels,
        int orientation) {
    uint32_t out_width = orientation >= 5 ? height : width;
    uint32_t out_height = orientation >= 5 ? width : height;
    size_t bytes = (size_t)out_width * out_height * 3;
    uint8_t *rgb = malloc(bytes);
    if (!rgb) return 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t ox = x, oy = y;
            switch (orientation) {
                case 2: ox = width - 1 - x; break;
                case 3: ox = width - 1 - x; oy = height - 1 - y; break;
                case 4: oy = height - 1 - y; break;
                case 5: ox = y; oy = x; break;
                case 6: ox = height - 1 - y; oy = x; break;
                case 7: ox = height - 1 - y; oy = width - 1 - x; break;
                case 8: ox = y; oy = width - 1 - x; break;
                default: break;
            }
            const uint8_t *src = pixels + ((size_t)y * width + x) * channels;
            uint8_t *dst = rgb + ((size_t)oy * out_width + ox) * 3;
            if (channels < 3) {
                dst[0] = dst[1] = dst[2] = src[0];
            } else {
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
    }

    out->width = out_width;
    out->height = out_height;
    out->rgb = rgb;
    ds4_sha256 sha;
    ds4_sha256_init(&sha);
    ds4_sha256_update(&sha, &out_width, sizeof(out_width));
    ds4_sha256_update(&sha, &out_height, sizeof(out_height));
    ds4_sha256_update(&sha, rgb, bytes);
    ds4_sha256_final(&sha, out->fingerprint);
    return 1;
}

/* ---- External decoder fallback --------------------------------------------
 *
 * The embedded Iris decoders cover 8-bit non-interlaced PNG and baseline or
 * progressive 8-bit JPEG.  Anything they reject, and the formats they do not
 * read at all (WebP, GIF, TIFF, BMP), goes to ImageMagick when it is
 * installed.  It runs as a child process so a decoder crash cannot take the
 * engine down, gets the encoded bytes on stdin and writes binary PPM to
 * stdout.  The input coder is always named explicitly ("png:-") so a request
 * can never reach ImageMagick's script-like formats (MVG, SVG, PostScript).
 *
 * DS4_IMAGE_DECODER selects the program: a path to `magick` or `convert`, or
 * "off" to use only the embedded decoders.  Without it, `magick` (ImageMagick
 * 7) and then `convert` (ImageMagick 6) are looked up in PATH. */

#define DS4_IMAGE_EXTERNAL_TIMEOUT_MS 60000
#define DS4_IMAGE_EXTERNAL_HINT "install ImageMagick (magick or convert in PATH) to decode it"

typedef enum {
    DS4_IMAGE_FMT_UNKNOWN = 0,
    DS4_IMAGE_FMT_PNG,
    DS4_IMAGE_FMT_JPEG,
    DS4_IMAGE_FMT_WEBP,
    DS4_IMAGE_FMT_GIF,
    DS4_IMAGE_FMT_TIFF,
    DS4_IMAGE_FMT_BMP,
} ds4_image_format;

static ds4_image_format ds4_image_sniff(const uint8_t *d, size_t n) {
    if (n >= 8 && memcmp(d, "\x89PNG\r\n\x1a\n", 8) == 0) return DS4_IMAGE_FMT_PNG;
    if (n >= 2 && d[0] == 0xff && d[1] == 0xd8) return DS4_IMAGE_FMT_JPEG;
    if (n >= 12 && memcmp(d, "RIFF", 4) == 0 && memcmp(d + 8, "WEBP", 4) == 0)
        return DS4_IMAGE_FMT_WEBP;
    if (n >= 6 && (memcmp(d, "GIF87a", 6) == 0 || memcmp(d, "GIF89a", 6) == 0))
        return DS4_IMAGE_FMT_GIF;
    if (n >= 4 && (memcmp(d, "II*\0", 4) == 0 || memcmp(d, "MM\0*", 4) == 0))
        return DS4_IMAGE_FMT_TIFF;
    if (n >= 14 && d[0] == 'B' && d[1] == 'M') return DS4_IMAGE_FMT_BMP;
    return DS4_IMAGE_FMT_UNKNOWN;
}

int ds4_image_has_known_magic(const uint8_t *data, size_t len) {
    return data && ds4_image_sniff(data, len) != DS4_IMAGE_FMT_UNKNOWN;
}

static const char *ds4_image_format_name(ds4_image_format fmt) {
    switch (fmt) {
        case DS4_IMAGE_FMT_PNG: return "PNG";
        case DS4_IMAGE_FMT_JPEG: return "JPEG";
        case DS4_IMAGE_FMT_WEBP: return "WebP";
        case DS4_IMAGE_FMT_GIF: return "GIF";
        case DS4_IMAGE_FMT_TIFF: return "TIFF";
        case DS4_IMAGE_FMT_BMP: return "BMP";
        default: return "image";
    }
}

/* ImageMagick coder names; the "[0]" keeps only the first frame. */
static const char *ds4_image_format_coder(ds4_image_format fmt) {
    switch (fmt) {
        case DS4_IMAGE_FMT_PNG: return "png:-[0]";
        case DS4_IMAGE_FMT_JPEG: return "jpeg:-[0]";
        case DS4_IMAGE_FMT_WEBP: return "webp:-[0]";
        case DS4_IMAGE_FMT_GIF: return "gif:-[0]";
        case DS4_IMAGE_FMT_TIFF: return "tiff:-[0]";
        case DS4_IMAGE_FMT_BMP: return "bmp:-[0]";
        default: return NULL;
    }
}

/* Why the embedded decoder turned the image down, from its header.  Only
 * used for the error message, so it never needs to be complete. */
static void ds4_image_embedded_reason(ds4_image_format fmt, const uint8_t *d,
                                      size_t n, char *out, size_t cap) {
    snprintf(out, cap, "invalid or unsupported %s", ds4_image_format_name(fmt));
    if (fmt == DS4_IMAGE_FMT_PNG) {
        if (n < 33 || memcmp(d + 12, "IHDR", 4) != 0) return;
        unsigned depth = d[24], color = d[25], interlace = d[28];
        bool palette_depth = color == 3 && (depth == 1 || depth == 2 || depth == 4);
        if (depth != 8 && !palette_depth)
            snprintf(out, cap, "unsupported PNG: bit depth %u", depth);
        else if (interlace)
            snprintf(out, cap, "unsupported PNG: Adam7 interlacing");
    } else if (fmt == DS4_IMAGE_FMT_JPEG) {
        size_t pos = 2;
        while (pos + 9 < n && d[pos] == 0xff) {
            unsigned marker = d[pos + 1];
            size_t seg = ((size_t)d[pos + 2] << 8) | d[pos + 3];
            bool sof = marker >= 0xc0 && marker <= 0xcf &&
                       marker != 0xc4 && marker != 0xc8 && marker != 0xcc;
            if (sof) {
                unsigned precision = d[pos + 4], comps = d[pos + 9];
                if (marker != 0xc0 && marker != 0xc2)
                    snprintf(out, cap, "unsupported JPEG: SOF%u coding", marker - 0xc0);
                else if (precision != 8)
                    snprintf(out, cap, "unsupported JPEG: %u-bit samples", precision);
                else if (comps != 1 && comps != 3)
                    snprintf(out, cap, "unsupported JPEG: %u color components%s",
                             comps, comps == 4 ? " (CMYK)" : "");
                return;
            }
            if (marker == 0xda || seg < 2) return;
            pos += 2 + seg;
        }
    } else {
        snprintf(out, cap, "%s is not supported by the built-in decoder",
                 ds4_image_format_name(fmt));
    }
}

static bool ds4_image_find_in_path(const char *name, char *out, size_t cap) {
    const char *path = getenv("PATH");
    if (!path || !*path) path = "/usr/local/bin:/usr/bin:/bin";
    while (*path) {
        const char *colon = strchr(path, ':');
        size_t len = colon ? (size_t)(colon - path) : strlen(path);
        if (len > 0) {
            int n = snprintf(out, cap, "%.*s/%s", (int)len, path, name);
            if (n > 0 && (size_t)n < cap && access(out, X_OK) == 0) return true;
        }
        if (!colon) break;
        path = colon + 1;
    }
    out[0] = '\0';
    return false;
}

/* Resolved on every fallback: a few access() calls are nothing next to the
 * process spawn, and the environment can change between requests. */
static const char *ds4_image_external_program(char *buf, size_t cap) {
    const char *env = getenv("DS4_IMAGE_DECODER");
    if (env && *env) {
        if (!strcmp(env, "off") || !strcmp(env, "none") || !strcmp(env, "0")) return NULL;
        if (strchr(env, '/')) {
            snprintf(buf, cap, "%s", env);
            return buf;
        }
        return ds4_image_find_in_path(env, buf, cap) ? buf : NULL;
    }
    if (ds4_image_find_in_path("magick", buf, cap) ||
        ds4_image_find_in_path("convert", buf, cap)) return buf;
    return NULL;
}

static int64_t ds4_image_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static bool ds4_image_pipe(int fds[2]) {
    if (pipe(fds) != 0) return false;
    for (int i = 0; i < 2; i++) {
        if (fcntl(fds[i], F_SETFD, FD_CLOEXEC) != 0 ||
            fcntl(fds[i], F_SETFL, fcntl(fds[i], F_GETFL) | O_NONBLOCK) != 0) {
            close(fds[0]);
            close(fds[1]);
            return false;
        }
    }
    return true;
}

static void ds4_image_close(int *fd) {
    if (*fd >= 0) close(*fd);
    *fd = -1;
}

/* Parse "P5"/"P6" output.  Rewrites the samples in place as 8-bit values and
 * returns a pointer to them, or NULL when the header or the size is wrong. */
static uint8_t *ds4_image_parse_pnm(uint8_t *d, size_t n, uint32_t *width,
                                    uint32_t *height, uint32_t *channels) {
    if (n < 3 || d[0] != 'P' || (d[1] != '5' && d[1] != '6')) return NULL;
    uint64_t field[3];
    size_t pos = 2;
    for (int f = 0; f < 3; f++) {
        for (;;) {
            while (pos < n && (d[pos] == ' ' || d[pos] == '\t' || d[pos] == '\n' ||
                               d[pos] == '\r' || d[pos] == '\v' || d[pos] == '\f')) pos++;
            if (pos < n && d[pos] == '#') {
                while (pos < n && d[pos] != '\n') pos++;
                continue;
            }
            break;
        }
        if (pos >= n || d[pos] < '0' || d[pos] > '9') return NULL;
        uint64_t v = 0;
        while (pos < n && d[pos] >= '0' && d[pos] <= '9') {
            v = v * 10 + (uint64_t)(d[pos++] - '0');
            if (v > 0xffffffffu) return NULL;
        }
        field[f] = v;
    }
    if (pos >= n) return NULL;
    pos++; /* the single whitespace byte before the raster */
    const uint64_t w = field[0], h = field[1], maxval = field[2];
    const uint32_t ch = d[1] == '6' ? 3u : 1u;
    if (w == 0 || h == 0 || w > DS4_IMAGE_MAX_DIMENSION || h > DS4_IMAGE_MAX_DIMENSION ||
        w * h > DS4_IMAGE_MAX_PIXELS || maxval == 0 || maxval > 65535) return NULL;
    const size_t samples = (size_t)(w * h * ch);
    const size_t bps = maxval > 255 ? 2 : 1;
    if (n - pos < samples * bps) return NULL;
    uint8_t *px = d + pos;
    if (maxval != 255) {
        for (size_t i = 0; i < samples; i++) {
            uint32_t v = bps == 2 ? ((uint32_t)px[2 * i] << 8) | px[2 * i + 1] : px[i];
            if (v > maxval) v = (uint32_t)maxval;
            px[i] = (uint8_t)((v * 255u + (uint32_t)maxval / 2u) / (uint32_t)maxval);
        }
    }
    *width = (uint32_t)w;
    *height = (uint32_t)h;
    *channels = ch;
    return px;
}

static int ds4_image_rm_tree(const char *path, const struct stat *sb,
                             int flags, struct FTW *ftw) {
    (void)sb; (void)flags; (void)ftw;
    remove(path); /* unlink for files; empty dirs rmdir after their children */
    return 0;
}

/* Run `program <args>`, feed it `input` on stdin, collect stdout and the head
 * of stderr.  tmpdir, when set, becomes the child's TMPDIR: ImageMagick
 * keeps its pixel-cache spill files there, and this function deletes the
 * whole tree once the child is gone.  ImageMagick cleans up on a normal
 * exit, but a child killed for the timeout or for writing past out_cap
 * never can, and a multi-GiB temp file in (RAM-backed) /tmp would survive
 * every rejected request.
 * Returns the exit status, or -1 when the child could not be run, or got
 * killed for the timeout or for writing more than out_cap bytes. */
static int ds4_image_run(const char *program, char *const argv[],
                         const uint8_t *input, size_t input_len,
                         uint8_t **out, size_t *out_len, size_t out_cap,
                         char *err, size_t err_cap, const char *tmpdir) {
    char tmpdir_var[544];
    const char *child_env[128];
    int in[2] = {-1, -1}, op[2] = {-1, -1}, ep[2] = {-1, -1};
    *out = NULL;
    *out_len = 0;
    if (err_cap) err[0] = '\0';
    /* The child env is built before anything can fail, per call: pthreads
     * has no setenv that is safe against other threads' getenv, and the
     * caller's TMPDIR must not leak to the child either -- argv's
     * -define registry:temporary-path redirects the spill itself (it
     * outranks MAGICK_TEMPORARY_PATH, MAGICK_TMPDIR and TMPDIR alike;
     * only a site policy's resource:temporary-path still wins, and that
     * is an administrator's explicit directory choice).  A fixed array
     * keeps this allocation-free; a caller with more environment
     * variables than fit only loses the external fallback. */
    if (tmpdir) {
        static const char *const spill_vars[] = {
            "TMPDIR=", "MAGICK_TEMPORARY_PATH=", "MAGICK_TMPDIR=" };
        snprintf(tmpdir_var, sizeof(tmpdir_var), "TMPDIR=%s", tmpdir);
        size_t j = 0;
        for (size_t i = 0; environ[i]; i++) {
            int strip = 0;
            for (int v = 0; v < 3; v++)
                strip |= strncmp(environ[i], spill_vars[v],
                                 strlen(spill_vars[v])) == 0;
            if (strip) continue;
            if (j + 2 >= sizeof(child_env) / sizeof(child_env[0])) {
                remove(tmpdir);
                snprintf(err, err_cap, "environment too large");
                return -1;
            }
            child_env[j++] = environ[i];
        }
        child_env[j++] = tmpdir_var;
        child_env[j] = NULL;
    }
    if (!ds4_image_pipe(in) || !ds4_image_pipe(op) || !ds4_image_pipe(ep)) {
        ds4_image_close(&in[0]); ds4_image_close(&in[1]);
        ds4_image_close(&op[0]); ds4_image_close(&op[1]);
        if (tmpdir) remove(tmpdir);
        snprintf(err, err_cap, "pipe: %s", strerror(errno));
        return -1;
    }
    /* The child must see blocking pipes; only the parent ends poll. */
    fcntl(in[0], F_SETFL, fcntl(in[0], F_GETFL) & ~O_NONBLOCK);
    fcntl(op[1], F_SETFL, fcntl(op[1], F_GETFL) & ~O_NONBLOCK);
    fcntl(ep[1], F_SETFL, fcntl(ep[1], F_GETFL) & ~O_NONBLOCK);
#ifdef F_SETNOSIGPIPE
    fcntl(in[1], F_SETNOSIGPIPE, 1);
#endif

    posix_spawn_file_actions_t fa;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, op[1], 1);
    posix_spawn_file_actions_adddup2(&fa, ep[1], 2);
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 34)
    /* Engine sockets and model fds without O_CLOEXEC stay out of the child. */
    posix_spawn_file_actions_addclosefrom_np(&fa, 3);
#endif
#endif
    posix_spawnattr_init(&attr);
    sigset_t none, dfl;
    sigemptyset(&none);
    sigemptyset(&dfl);
    sigaddset(&dfl, SIGPIPE);
    posix_spawnattr_setsigmask(&attr, &none);
    posix_spawnattr_setsigdefault(&attr, &dfl);
    short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT; /* Darwin: only the dup2 targets survive. */
#endif
    posix_spawnattr_setflags(&attr, flags);
    pid_t pid = -1;
    int rc = posix_spawn(&pid, program, &fa, &attr, argv,
                         tmpdir ? (char **)child_env : environ);
    posix_spawn_file_actions_destroy(&fa);
    posix_spawnattr_destroy(&attr);
    ds4_image_close(&in[0]);
    ds4_image_close(&op[1]);
    ds4_image_close(&ep[1]);
    if (rc != 0) {
        ds4_image_close(&in[1]);
        ds4_image_close(&op[0]);
        ds4_image_close(&ep[0]);
        if (tmpdir) remove(tmpdir);
        snprintf(err, err_cap, "%.160s: %s", program, strerror(rc));
        return -1;
    }

#ifndef F_SETNOSIGPIPE
    /* A child that exits before reading all of stdin turns the next write
     * into SIGPIPE.  Block it on this thread and drop the one we caused. */
    sigset_t pipe_set, old_set;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &pipe_set, &old_set);
    sigset_t pending;
    sigpending(&pending);
    const bool pipe_was_pending = sigismember(&pending, SIGPIPE) == 1;
    bool raised_pipe = false;
#endif

    size_t written = 0, cap = 0, err_len = 0;
    uint8_t *buf = NULL;
    bool killed = false;
    const int64_t deadline = ds4_image_now_ms() + DS4_IMAGE_EXTERNAL_TIMEOUT_MS;
    if (input_len == 0) ds4_image_close(&in[1]);
    while (op[0] >= 0 || ep[0] >= 0) {
        struct pollfd pfd[3];
        int nfd = 0, iin = -1, iout = -1, ierr = -1;
        if (in[1] >= 0) { iin = nfd; pfd[nfd++] = (struct pollfd){.fd = in[1], .events = POLLOUT}; }
        if (op[0] >= 0) { iout = nfd; pfd[nfd++] = (struct pollfd){.fd = op[0], .events = POLLIN}; }
        if (ep[0] >= 0) { ierr = nfd; pfd[nfd++] = (struct pollfd){.fd = ep[0], .events = POLLIN}; }
        int64_t left = deadline - ds4_image_now_ms();
        if (left <= 0) {
            snprintf(err, err_cap, "timed out after %d s", DS4_IMAGE_EXTERNAL_TIMEOUT_MS / 1000);
            killed = true;
            break;
        }
        int pr = poll(pfd, (nfds_t)nfd, (int)left);
        if (pr < 0) {
            if (errno == EINTR) continue;
            snprintf(err, err_cap, "poll: %s", strerror(errno));
            killed = true;
            break;
        }
        if (iin >= 0 && pfd[iin].revents) {
            ssize_t w = write(in[1], input + written, input_len - written);
            if (w > 0) {
                written += (size_t)w;
                if (written == input_len) ds4_image_close(&in[1]);
            } else if (w < 0 && errno != EAGAIN && errno != EINTR) {
#ifndef F_SETNOSIGPIPE
                if (errno == EPIPE) raised_pipe = true;
#endif
                ds4_image_close(&in[1]);
            }
        }
        if (iout >= 0 && pfd[iout].revents) {
            if (cap - *out_len < 65536) {
                size_t grow = cap ? cap * 2 : (size_t)1 << 20;
                if (grow > out_cap) grow = out_cap;
                if (grow <= *out_len) {
                    snprintf(err, err_cap, "decoded output exceeds the size limit");
                    killed = true;
                    break;
                }
                uint8_t *nb = realloc(buf, grow);
                if (!nb) {
                    snprintf(err, err_cap, "out of memory");
                    killed = true;
                    break;
                }
                buf = nb;
                cap = grow;
            }
            ssize_t r = read(op[0], buf + *out_len, cap - *out_len);
            if (r > 0) *out_len += (size_t)r;
            else if (r == 0 || (errno != EAGAIN && errno != EINTR)) ds4_image_close(&op[0]);
        }
        if (ierr >= 0 && pfd[ierr].revents) {
            char tmp[512];
            ssize_t r = read(ep[0], tmp, sizeof(tmp));
            if (r > 0) {
                size_t keep = err_cap > err_len + 1 ? err_cap - err_len - 1 : 0;
                if ((size_t)r < keep) keep = (size_t)r;
                memcpy(err + err_len, tmp, keep);
                err_len += keep;
                if (err_cap) err[err_len] = '\0';
            } else if (r == 0 || (errno != EAGAIN && errno != EINTR)) {
                ds4_image_close(&ep[0]);
            }
        }
    }
    ds4_image_close(&in[1]);
    ds4_image_close(&op[0]);
    ds4_image_close(&ep[0]);
    if (killed) kill(pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (tmpdir) nftw(tmpdir, ds4_image_rm_tree, 8, FTW_DEPTH | FTW_PHYS);

#ifndef F_SETNOSIGPIPE
    if (raised_pipe && !pipe_was_pending) {
        struct timespec zero = {0, 0};
        while (sigtimedwait(&pipe_set, NULL, &zero) < 0 && errno == EINTR) {}
    }
    pthread_sigmask(SIG_SETMASK, &old_set, NULL);
#endif

    *out = buf;
    if (killed) return -1;
    if (WIFSIGNALED(status)) {
        if (!err_len) snprintf(err, err_cap, "killed by signal %d", WTERMSIG(status));
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void ds4_image_first_line(char *s) {
    while (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t') memmove(s, s + 1, strlen(s));
    char *nl = strpbrk(s, "\r\n");
    if (nl) *nl = '\0';
}

static int ds4_image_decode_external(ds4_image *out, ds4_image_format fmt,
                                     const char *program,
                                     const uint8_t *encoded, size_t encoded_len,
                                     char *why, size_t why_cap) {
    char lim_w[16], lim_h[16], lim_area[24];
    snprintf(lim_w, sizeof(lim_w), "%u", DS4_IMAGE_MAX_DIMENSION);
    snprintf(lim_h, sizeof(lim_h), "%u", DS4_IMAGE_MAX_DIMENSION);
    /* ImageMagick's area limit counts pixels, but exceeding it does not
     * reject the image: the pixel cache spills to backing files under
     * TMPDIR instead, and a child killed for writing past out_cap cannot
     * remove them -- one 16384x16384 PNG left 2.8 GiB of files in the
     * tmpfs /tmp of a Strix Halo box.  So: spill to a private directory
     * that the parent deletes after waitpid, and cap it at the 2 GiB
     * ImageMagick's own disk default already allows: every image ds4
     * accepts then decodes exactly as before, while an oversized one
     * fills the cap and errors out instead of writing without bound.
     * Area is compared strictly (size < limit), so the +1 keeps a
     * legitimate exactly-64 MP image on its in-RAM cache. */
    snprintf(lim_area, sizeof(lim_area), "%u", DS4_IMAGE_MAX_PIXELS + 1u);
    /* Private scratch dir under the caller's TMPDIR.  The fallback refuses
     * to run without it: a child we later kill can only be cleaned up by
     * us if its files are somewhere we know. */
    const char *scratch = getenv("TMPDIR");
    if (!scratch || !*scratch) scratch = "/tmp";
    size_t scratch_len = strlen(scratch);
    while (scratch_len > 1 && scratch[scratch_len - 1] == '/') scratch_len--;
    char tmpdir_path[512];
    if (scratch_len + 24 >= sizeof(tmpdir_path)) {
        snprintf(why, why_cap, "TMPDIR is too long");
        return 0;
    }
    snprintf(tmpdir_path, sizeof(tmpdir_path), "%.*s/ds4-img-XXXXXX",
             (int)scratch_len, scratch);
    if (!mkdtemp(tmpdir_path)) {
        snprintf(why, why_cap, "cannot create the decoder temp directory");
        return 0;
    }
    const char *tmpdir = tmpdir_path;
    /* ImageMagick picks its spill directory from a define, an explicit
     * site policy, or MAGICK_TEMPORARY_PATH / MAGICK_TMPDIR / TMPDIR, in
     * that order.  The define outranks the rest, so the spill always
     * lands in the private directory; a policy's resource:temporary-path
     * still wins, and that is an administrator's explicit choice. */
    char temp_dir_def[560];
    snprintf(temp_dir_def, sizeof(temp_dir_def),
             "registry:temporary-path=%s", tmpdir);
    char *argv[] = {
        (char *)program,
        "-limit", "width", lim_w,
        "-limit", "height", lim_h,
        "-limit", "area", lim_area,
        "-limit", "disk", "2GiB",
        "-limit", "time", "60",
        "-define", temp_dir_def,
        (char *)ds4_image_format_coder(fmt),
        "-auto-orient",
        "-alpha", "off",
        "-depth", "8",
        "ppm:-",
        NULL,
    };
    uint8_t *pnm = NULL;
    size_t pnm_len = 0;
    char err[256];
    const size_t out_cap = 64 + (size_t)DS4_IMAGE_MAX_PIXELS * 3u;
    int status = ds4_image_run(program, argv, encoded, encoded_len,
                               &pnm, &pnm_len, out_cap, err, sizeof(err), tmpdir);
    ds4_image_first_line(err);
    uint32_t w = 0, h = 0, ch = 0;
    uint8_t *px = status == 0 ? ds4_image_parse_pnm(pnm, pnm_len, &w, &h, &ch) : NULL;
    int ok = 0;
    if (px) {
        ok = ds4_oriented_rgb(out, px, w, h, ch, 1);
        if (!ok) snprintf(why, why_cap, "unable to allocate decoded pixels");
    } else if (status == 0) {
        snprintf(why, why_cap, "%.160s wrote no usable image", program);
    } else if (err[0]) {
        snprintf(why, why_cap, "%s", err);
    } else {
        snprintf(why, why_cap, "%.160s exited with status %d", program, status);
    }
    free(pnm);
    return ok;
}

int ds4_image_decode_memory(
        ds4_image *out,
        const uint8_t *encoded,
        size_t encoded_len,
        char *error,
        size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!encoded || encoded_len == 0 || encoded_len > DS4_IMAGE_MAX_ENCODED_BYTES) {
        ds4_image_error(error, error_cap, "image is empty or exceeds the 64 MiB encoded limit");
        return 0;
    }

    const ds4_image_format fmt = ds4_image_sniff(encoded, encoded_len);
    if (fmt == DS4_IMAGE_FMT_UNKNOWN) {
        ds4_image_error(error, error_cap,
                        "image must be PNG, JPEG, WebP, GIF, TIFF or BMP");
        return 0;
    }
    if (fmt == DS4_IMAGE_FMT_PNG) {
        png_image *decoded = png_load_mem(encoded, encoded_len);
        if (decoded) {
            int ok = ds4_oriented_rgb(out, decoded->data,
                                      (uint32_t)decoded->width,
                                      (uint32_t)decoded->height,
                                      (uint32_t)decoded->channels, 1);
            png_free(decoded);
            if (!ok) ds4_image_error(error, error_cap, "unable to allocate decoded PNG pixels");
            return ok;
        }
    } else if (fmt == DS4_IMAGE_FMT_JPEG) {
        jpeg_image *decoded = jpeg_load_mem(encoded, encoded_len);
        if (decoded) {
            int orientation = ds4_jpeg_orientation(encoded, encoded_len);
            int ok = ds4_oriented_rgb(out, decoded->data,
                                      (uint32_t)decoded->width,
                                      (uint32_t)decoded->height,
                                      (uint32_t)decoded->channels, orientation);
            jpeg_free(decoded);
            if (!ok) ds4_image_error(error, error_cap, "unable to allocate decoded JPEG pixels");
            return ok;
        }
    }

    /* The embedded decoder could not take it: try ImageMagick. */
    char reason[128];
    ds4_image_embedded_reason(fmt, encoded, encoded_len, reason, sizeof(reason));
    char program_buf[4096];
    const char *program = ds4_image_external_program(program_buf, sizeof(program_buf));
    if (!program) {
        if (error && error_cap)
            snprintf(error, error_cap, "%s; " DS4_IMAGE_EXTERNAL_HINT, reason);
        return 0;
    }
    char why[256];
    if (ds4_image_decode_external(out, fmt, program, encoded, encoded_len,
                                  why, sizeof(why)))
        return 1;
    if (error && error_cap) snprintf(error, error_cap, "%s; ImageMagick fallback failed: %s",
                                     reason, why);
    return 0;
}

int ds4_image_decode_file(
        ds4_image *out,
        const char *path,
        char *error,
        size_t error_cap) {
    if (!out || !path || !path[0]) {
        ds4_image_error(error, error_cap, "image path is empty");
        return 0;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        if (error && error_cap) snprintf(error, error_cap, "%s: %s", path, strerror(errno));
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) goto io_error;
    long end = ftell(fp);
    if (end <= 0 || (unsigned long)end > DS4_IMAGE_MAX_ENCODED_BYTES) {
        fclose(fp);
        ds4_image_error(error, error_cap, "image is empty or exceeds the 64 MiB encoded limit");
        return 0;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) goto io_error;
    size_t len = (size_t)end;
    uint8_t *data = malloc(len);
    if (!data) {
        fclose(fp);
        ds4_image_error(error, error_cap, "unable to allocate encoded image buffer");
        return 0;
    }
    if (fread(data, 1, len, fp) != len) {
        free(data);
        goto io_error;
    }
    fclose(fp);
    int ok = ds4_image_decode_memory(out, data, len, error, error_cap);
    free(data);
    return ok;

io_error:
    if (error && error_cap) snprintf(error, error_cap, "%s: %s", path, strerror(errno));
    fclose(fp);
    return 0;
}

void ds4_image_free(ds4_image *image) {
    if (!image) return;
    free(image->rgb);
    memset(image, 0, sizeof(*image));
}

static uint32_t ds4_align_u32(uint32_t value, uint32_t factor) {
    return (value + factor - 1) / factor * factor;
}

static int ds4_glm53_smart_resize(
        uint32_t height,
        uint32_t width,
        uint32_t min_tokens,
        uint32_t max_tokens,
        uint32_t *target_height,
        uint32_t *target_width) {
    const uint32_t temporal = 2, factor = 28;
    uint64_t pixels_per_token = (uint64_t)temporal * factor * factor;
    uint64_t min_pixels = (uint64_t)min_tokens * pixels_per_token;
    uint64_t max_pixels = (uint64_t)max_tokens * pixels_per_token;
    uint32_t aligned_height = ds4_align_u32(height, factor);
    uint32_t aligned_width = ds4_align_u32(width, factor);
    uint64_t budget = (uint64_t)temporal * aligned_height * aligned_width;

    if (budget < min_pixels) {
        double scale = sqrt((double)min_pixels /
                            ((double)temporal * height * width));
        aligned_height = ds4_align_u32((uint32_t)ceil(height * scale), factor);
        aligned_width = ds4_align_u32((uint32_t)ceil(width * scale), factor);
        budget = (uint64_t)temporal * aligned_height * aligned_width;
    }
    if (budget > max_pixels) {
        if (max_pixels < (uint64_t)temporal * factor * factor) return 0;
        uint32_t low = 1, high = height;
        aligned_height = aligned_width = factor;
        while (low <= high) {
            uint32_t content_height = low + (high - low) / 2;
            uint32_t content_width = (uint32_t)floor(
                    (double)width * content_height / height);
            if (content_width < 1) content_width = 1;
            uint32_t candidate_height = ds4_align_u32(content_height, factor);
            uint32_t candidate_width = ds4_align_u32(content_width, factor);
            uint64_t candidate = (uint64_t)temporal * candidate_height * candidate_width;
            if (candidate <= max_pixels) {
                aligned_height = candidate_height;
                aligned_width = candidate_width;
                low = content_height + 1;
            } else {
                high = content_height - 1;
            }
        }
    }
    *target_height = aligned_height;
    *target_width = aligned_width;
    return 1;
}

static double ds4_cubic(double x, double a) {
    x = fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

typedef struct {
    const uint8_t *src;
    float *dst;
    uint32_t src_width, src_height, dst_width, dst_height, dst_stride;
    uint32_t first_y, end_y;
} ds4_resize_rows;

static void *ds4_resize_bicubic_rows(void *context) {
    const ds4_resize_rows *rows = context;
    const uint8_t *src = rows->src;
    float *dst = rows->dst;
    const uint32_t src_width = rows->src_width, src_height = rows->src_height;
    const uint32_t dst_width = rows->dst_width, dst_height = rows->dst_height;
    const uint32_t dst_stride = rows->dst_stride;
    double scale_x = (double)src_width / dst_width;
    double scale_y = (double)src_height / dst_height;
    double filter_x = scale_x >= 1.0 ? 1.0 / scale_x : 1.0;
    double filter_y = scale_y >= 1.0 ? 1.0 / scale_y : 1.0;
    double support_x = scale_x >= 1.0 ? 2.0 * scale_x : 2.0;
    double support_y = scale_y >= 1.0 ? 2.0 * scale_y : 2.0;

    for (uint32_t dy = rows->first_y; dy < rows->end_y; dy++) {
        double center_y = scale_y * ((double)dy + 0.5);
        int y0 = (int)(center_y - support_y + 0.5);
        int y1 = (int)(center_y + support_y + 0.5);
        if (y0 < 0) y0 = 0;
        if (y1 > (int)src_height) y1 = (int)src_height;
        for (uint32_t dx = 0; dx < dst_width; dx++) {
            double center_x = scale_x * ((double)dx + 0.5);
            int x0 = (int)(center_x - support_x + 0.5);
            int x1 = (int)(center_x + support_x + 0.5);
            if (x0 < 0) x0 = 0;
            if (x1 > (int)src_width) x1 = (int)src_width;
            double sum[3] = {0, 0, 0};
            double weight_sum = 0;
            for (int iy = y0; iy < y1; iy++) {
                double wy = ds4_cubic(((double)iy + 0.5 - center_y) * filter_y,
                                      -0.5);
                for (int ix = x0; ix < x1; ix++) {
                    double wx = ds4_cubic(((double)ix + 0.5 - center_x) * filter_x,
                                          -0.5);
                    double weight = wx * wy;
                    const uint8_t *pixel = src +
                        ((size_t)iy * src_width + (uint32_t)ix) * 3;
                    sum[0] += pixel[0] * weight;
                    sum[1] += pixel[1] * weight;
                    sum[2] += pixel[2] * weight;
                    weight_sum += weight;
                }
            }
            float *pixel = dst + ((size_t)dy * dst_stride + dx) * 3;
            for (unsigned c = 0; c < 3; c++) {
                double value = round(sum[c] / weight_sum);
                if (value < 0.0) value = 0.0;
                if (value > 255.0) value = 255.0;
                pixel[c] = (float)value;
            }
        }
    }
    return NULL;
}

static void ds4_resize_rgb_bicubic(
        const uint8_t *src,
        uint32_t src_width,
        uint32_t src_height,
        float *dst,
        uint32_t dst_width,
        uint32_t dst_height,
        uint32_t dst_stride) {
    enum { MAX_WORKERS = 8 };
    /* Separate output row ranges use the original horizontal and vertical
     * coordinates, including when the destination is a padded canvas. */
    const uint64_t work = (uint64_t)(src_width > dst_width ? src_width : dst_width) *
                          (src_height > dst_height ? src_height : dst_height);
    uint32_t workers = work / 65536u < MAX_WORKERS ? (uint32_t)(work / 65536u) : MAX_WORKERS;
    if (workers > dst_height) workers = dst_height;
    ds4_resize_rows whole = {src, dst, src_width, src_height, dst_width, dst_height,
                             dst_stride, 0, dst_height};
    if (workers < 2u) {
        ds4_resize_bicubic_rows(&whole);
        return;
    }
    ds4_resize_rows rows[MAX_WORKERS];
    pthread_t threads[MAX_WORKERS - 1];
    uint32_t started = 0;
    for (; started + 1u < workers; started++) {
        rows[started] = whole;
        rows[started].first_y = (uint64_t)dst_height * started / workers;
        rows[started].end_y = (uint64_t)dst_height * (started + 1u) / workers;
        if (pthread_create(&threads[started], NULL, ds4_resize_bicubic_rows, &rows[started])) break;
    }
    whole.first_y = (uint64_t)dst_height * started / workers;
    ds4_resize_bicubic_rows(&whole);
    for (uint32_t i = 0; i < started; i++) {
        if (pthread_join(threads[i], NULL)) abort();
    }
}

int ds4_image_preprocess_glm53(
        ds4_image_patches *out,
        const ds4_image *image,
        uint32_t min_image_tokens,
        uint32_t max_image_tokens,
        char *error,
        size_t error_cap) {
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float stddev[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!image || !image->rgb || image->width == 0 || image->height == 0 ||
        min_image_tokens == 0 || max_image_tokens < min_image_tokens ||
        max_image_tokens > 8000) {
        ds4_image_error(error, error_cap, "invalid GLM 5.3 image preprocessing parameters");
        return 0;
    }

    uint32_t target_height, target_width;
    if (!ds4_glm53_smart_resize(image->height, image->width,
                                min_image_tokens, max_image_tokens,
                                &target_height, &target_width)) {
        ds4_image_error(error, error_cap, "image token budget is too small");
        return 0;
    }
    double scale = fmin((double)target_height / image->height,
                        (double)target_width / image->width);
    uint64_t min_pixels = (uint64_t)2 * 28 * 28 * min_image_tokens;
    if ((uint64_t)2 * image->height * image->width >= min_pixels && scale > 1.0) {
        scale = 1.0;
    }
    uint32_t content_height = (uint32_t)floor(image->height * scale);
    uint32_t content_width = (uint32_t)floor(image->width * scale);
    if (content_height < 1) content_height = 1;
    if (content_width < 1) content_width = 1;
    if (content_height > target_height) content_height = target_height;
    if (content_width > target_width) content_width = target_width;

    size_t canvas_values = (size_t)target_height * target_width * 3;
    float *canvas = calloc(canvas_values, sizeof(float));
    if (!canvas) {
        ds4_image_error(error, error_cap, "unable to allocate resized image");
        return 0;
    }
    if (content_width == image->width && content_height == image->height) {
        for (uint32_t y = 0; y < content_height; y++) {
            for (uint32_t x = 0; x < content_width; x++) {
                const uint8_t *src = image->rgb + ((size_t)y * image->width + x) * 3;
                float *dst = canvas + ((size_t)y * target_width + x) * 3;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
    } else {
        ds4_resize_rgb_bicubic(image->rgb, image->width, image->height,
                               canvas, content_width, content_height, target_width);
    }

    for (uint32_t y = 0; y < target_height; y++) {
        for (uint32_t x = 0; x < target_width; x++) {
            float *pixel = canvas + ((size_t)y * target_width + x) * 3;
            for (unsigned c = 0; c < 3; c++) {
                float value = (x < content_width && y < content_height) ? pixel[c] : 0.0f;
                pixel[c] = (value / 255.0f - mean[c]) / stddev[c];
            }
        }
    }

    uint32_t grid_height = target_height / 14;
    uint32_t grid_width = target_width / 14;
    uint32_t patch_count = grid_height * grid_width;
    size_t patch_values = (size_t)patch_count * 3 * 2 * 14 * 14;
    float *patches = malloc(patch_values * sizeof(float));
    if (!patches) {
        free(canvas);
        ds4_image_error(error, error_cap, "unable to allocate vision patches");
        return 0;
    }

    size_t index = 0;
    for (uint32_t block_y = 0; block_y < grid_height / 2; block_y++) {
        for (uint32_t block_x = 0; block_x < grid_width / 2; block_x++) {
            for (uint32_t merge_y = 0; merge_y < 2; merge_y++) {
                for (uint32_t merge_x = 0; merge_x < 2; merge_x++) {
                    uint32_t patch_y = block_y * 2 + merge_y;
                    uint32_t patch_x = block_x * 2 + merge_x;
                    for (uint32_t channel = 0; channel < 3; channel++) {
                        for (uint32_t temporal = 0; temporal < 2; temporal++) {
                            (void)temporal;
                            for (uint32_t y = 0; y < 14; y++) {
                                for (uint32_t x = 0; x < 14; x++) {
                                    const float *pixel = canvas +
                                        ((size_t)(patch_y * 14 + y) * target_width +
                                         patch_x * 14 + x) * 3;
                                    patches[index++] = pixel[channel];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    free(canvas);
    if (index != patch_values) {
        free(patches);
        ds4_image_error(error, error_cap, "internal vision patch layout mismatch");
        return 0;
    }

    out->content_width = content_width;
    out->content_height = content_height;
    out->padded_width = target_width;
    out->padded_height = target_height;
    out->grid_width = grid_width;
    out->grid_height = grid_height;
    out->patch_count = patch_count;
    out->image_token_count = patch_count / 4;
    out->patches = patches;
    return 1;
}

int ds4_image_preprocess_qwen4(
        ds4_image_patches *out,
        const ds4_image *image,
        uint32_t min_image_tokens,
        uint32_t max_image_tokens,
        char *error,
        size_t error_cap) {
    const uint32_t patch = 16, factor = 32;
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!image || !image->rgb || image->width == 0 || image->height == 0 ||
        min_image_tokens == 0 || max_image_tokens < min_image_tokens) {
        ds4_image_error(error, error_cap, "invalid Qwen3.8 image preprocessing parameters");
        return 0;
    }
    const double h = image->height, w = image->width;
    if (fmax(h, w) / fmin(h, w) > 200.0) {
        ds4_image_error(error, error_cap, "image aspect ratio must be below 200");
        return 0;
    }
    /* Qwen2VL smart_resize: round to the factor, then scale into [min, max] pixels */
    const double min_pixels = (double)min_image_tokens * factor * factor;
    const double max_pixels = (double)max_image_tokens * factor * factor;
    double h_bar = fmax(factor, round(h / factor) * factor);
    double w_bar = fmax(factor, round(w / factor) * factor);
    if (h_bar * w_bar > max_pixels) {
        const double beta = sqrt(h * w / max_pixels);
        h_bar = fmax(factor, floor(h / beta / factor) * factor);
        w_bar = fmax(factor, floor(w / beta / factor) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        const double beta = sqrt(min_pixels / (h * w));
        h_bar = ceil(h * beta / factor) * factor;
        w_bar = ceil(w * beta / factor) * factor;
    }
    const uint32_t target_height = (uint32_t)h_bar, target_width = (uint32_t)w_bar;
    if (target_width > DS4_IMAGE_MAX_DIMENSION || target_height > DS4_IMAGE_MAX_DIMENSION ||
        (uint64_t)target_width * target_height > DS4_IMAGE_MAX_PIXELS) {
        ds4_image_error(error, error_cap, "resized image is too large");
        return 0;
    }
    float *canvas = malloc((size_t)target_height * target_width * 3 * sizeof(float));
    if (!canvas) {
        ds4_image_error(error, error_cap, "unable to allocate resized image");
        return 0;
    }
    if (target_width == image->width && target_height == image->height) {
        for (size_t i = 0; i < (size_t)target_height * target_width * 3; i++) canvas[i] = image->rgb[i];
    } else {
        ds4_resize_rgb_bicubic(image->rgb, image->width, image->height,
                               canvas, target_width, target_height, target_width);
    }
    for (size_t i = 0; i < (size_t)target_height * target_width * 3; i++) {
        canvas[i] = (canvas[i] / 255.0f - 0.5f) / 0.5f;
    }
    const uint32_t grid_height = target_height / patch, grid_width = target_width / patch;
    const uint32_t patch_count = grid_height * grid_width;
    const size_t patch_values = (size_t)patch_count * 3 * patch * patch;
    float *patches = malloc(patch_values * sizeof(float));
    if (!patches) {
        free(canvas);
        ds4_image_error(error, error_cap, "unable to allocate vision patches");
        return 0;
    }
    size_t index = 0;
    for (uint32_t block_y = 0; block_y < grid_height / 2; block_y++) {
        for (uint32_t block_x = 0; block_x < grid_width / 2; block_x++) {
            for (uint32_t merge_y = 0; merge_y < 2; merge_y++) {
                for (uint32_t merge_x = 0; merge_x < 2; merge_x++) {
                    const uint32_t patch_y = block_y * 2 + merge_y, patch_x = block_x * 2 + merge_x;
                    for (uint32_t channel = 0; channel < 3; channel++) {
                        for (uint32_t y = 0; y < patch; y++) {
                            for (uint32_t x = 0; x < patch; x++) {
                                patches[index++] = canvas[((size_t)(patch_y * patch + y) * target_width +
                                                           patch_x * patch + x) * 3 + channel];
                            }
                        }
                    }
                }
            }
        }
    }
    free(canvas);
    out->content_width = target_width;
    out->content_height = target_height;
    out->padded_width = target_width;
    out->padded_height = target_height;
    out->grid_width = grid_width;
    out->grid_height = grid_height;
    out->patch_count = patch_count;
    out->image_token_count = patch_count / 4;
    out->patches = patches;
    return 1;
}

void ds4_image_patches_free(ds4_image_patches *patches) {
    if (!patches) return;
    free(patches->patches);
    memset(patches, 0, sizeof(*patches));
}

static uint32_t ds4_deepseek4_grid_tokens(
        uint32_t height,
        uint32_t width,
        uint32_t *grid_height,
        uint32_t *grid_width,
        bool v41) {
    const uint32_t h = (height / 14u + 2u) / 3u;
    const uint32_t w = (width / 14u + 2u) / 3u;
    uint64_t count = (uint64_t)h * (w + 1u) + 2u;
    if (!v41) {
        if (h & 1u) count += w + 1u;
        if ((((h + 1u) / 2u) * (w + 1u)) & 1u) count += 2u;
    }
    if (grid_height) *grid_height = h;
    if (grid_width) *grid_width = w;
    return count <= UINT32_MAX ? (uint32_t)count : UINT32_MAX;
}

static int ds4_deepseek4_solve_resize(
        uint32_t height,
        uint32_t width,
        uint32_t budget,
        uint32_t *best_height,
        uint32_t *best_width,
        bool v41) {
    const double ratio = (double)height / width;
    const double max_w_float = sqrt(((double)budget - 2.0) / ratio + 0.25) - 0.5;
    const double max_h_float = max_w_float * ratio;
    uint32_t max_h, max_w;
    if (max_w_float < 1.0) {
        max_w = 1;
        max_h = (budget - 2u) / (max_w + 1u);
        if (!v41) max_h &= ~1u;
        if (max_h == 0) return 0;
        *best_width = max_w * 42u;
        *best_height = max_h * 42u;
    } else if (max_h_float < (v41 ? 1.0 : 2.0)) {
        max_h = v41 ? 1u : 2u;
        max_w = (budget - 2u) / max_h - 1u;
        if (max_w <= 1u) return 0;
        *best_width = max_w * 42u;
        *best_height = max_h * 42u;
    } else {
        max_w = (uint32_t)floor(max_w_float);
        max_h = (uint32_t)floor(max_h_float);
        if (!v41) max_h &= ~1u;
        if (max_w == 0 || max_h == 0) return 0;
        const double scale = fmin((double)max_w * 42.0 / width,
                                  (double)max_h * 42.0 / height);
        *best_width = (uint32_t)floor(width * scale / 14.0) * 14u;
        *best_height = (uint32_t)floor(height * scale / 14.0) * 14u;
    }
    return *best_width != 0 && *best_height != 0;
}

static int ds4_deepseek4_safe_resize(
        uint32_t height,
        uint32_t width,
        uint32_t *best_height,
        uint32_t *best_width,
        uint32_t *llm_height,
        uint32_t *llm_width,
        bool v41) {
    const uint32_t max_tokens = v41 ? 1024u : 381u;
    uint32_t budget = max_tokens;
    uint32_t tokens = ds4_deepseek4_grid_tokens(
            *best_height, *best_width, llm_height, llm_width, v41);
    while (tokens > max_tokens) {
        if (budget <= 4u ||
            !ds4_deepseek4_solve_resize(height, width, budget,
                                        best_height, best_width, v41)) {
            return 0;
        }
        tokens = ds4_deepseek4_grid_tokens(
                *best_height, *best_width, llm_height, llm_width, v41);
        budget--;
    }
    return 1;
}

static int ds4_image_preprocess_deepseek(
        ds4_deepseek4_image_patches *out,
        const ds4_image *image,
        char *error,
        size_t error_cap,
        bool v41) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!image || !image->rgb || image->width == 0 || image->height == 0) {
        ds4_image_error(error, error_cap,
                        "invalid DeepSeek V4 image preprocessing input");
        return 0;
    }

    uint32_t planned_width = image->width;
    uint32_t planned_height = image->height;
    if (!v41 && (uint64_t)planned_width > (uint64_t)planned_height * 8u) {
        planned_width = planned_height * 8u;
    }
    const uint64_t planned_pixels =
        (uint64_t)planned_width * planned_height;
    const uint32_t min_pixels = v41 ? 295936u : 147456u;
    if (planned_pixels < min_pixels) {
        const double scale = sqrt((double)min_pixels / planned_pixels);
        planned_width = (uint32_t)(planned_width * scale);
        planned_height = (uint32_t)(planned_height * scale);
        if (planned_width == 0) planned_width = 1;
        if (planned_height == 0) planned_height = 1;
    }

    uint32_t best_width = ds4_align_u32(planned_width, 14u);
    uint32_t best_height = ds4_align_u32(planned_height, 14u);
    uint32_t llm_height = 0, llm_width = 0;
    if (!ds4_deepseek4_safe_resize(planned_height, planned_width,
                                   &best_height, &best_width,
                                   &llm_height, &llm_width, v41)) {
        ds4_image_error(error, error_cap,
                        "unable to fit image within the vision token budget");
        return 0;
    }
    if (best_width > DS4_IMAGE_MAX_DIMENSION ||
        best_height > DS4_IMAGE_MAX_DIMENSION ||
        (uint64_t)best_width * best_height > DS4_IMAGE_MAX_PIXELS) {
        ds4_image_error(error, error_cap,
                        "resized image exceeds the decoded image limits");
        return 0;
    }

    size_t canvas_values = (size_t)best_width * best_height * 3u;
    float *canvas = malloc(canvas_values * sizeof(*canvas));
    if (!canvas) {
        ds4_image_error(error, error_cap, "unable to allocate resized image");
        return 0;
    }
    for (size_t i = 0; i < canvas_values; i++) canvas[i] = 127.0f;

    uint32_t content_width = best_width;
    uint32_t content_height = best_height;
    uint32_t offset_x = 0, offset_y = 0;
    const bool force_resize =
        !v41 && (uint64_t)image->width >= (uint64_t)image->height * 8u;
    if (!force_resize) {
        const double scale = fmin((double)best_width / image->width,
                                  (double)best_height / image->height);
        content_width = (uint32_t)lrint(image->width * scale);
        content_height = (uint32_t)lrint(image->height * scale);
        if (content_width < 1) content_width = 1;
        if (content_height < 1) content_height = 1;
        if (content_width > best_width) content_width = best_width;
        if (content_height > best_height) content_height = best_height;
        offset_x = (uint32_t)lrint((best_width - content_width) * 0.5);
        offset_y = (uint32_t)lrint((best_height - content_height) * 0.5);
    }

    float *resized = canvas + ((size_t)offset_y * best_width + offset_x) * 3u;
    ds4_resize_rgb_bicubic(image->rgb, image->width, image->height,
                           resized, content_width, content_height, best_width);
    for (size_t i = 0; i < canvas_values; i++) {
        canvas[i] = canvas[i] / 127.5f - 1.0f;
    }

    const uint32_t grid_height = best_height / 14u;
    const uint32_t grid_width = best_width / 14u;
    const uint32_t patch_count = grid_height * grid_width;
    const size_t patch_values = (size_t)patch_count * 3u * 14u * 14u;
    float *patches = malloc(patch_values * sizeof(*patches));
    if (!patches) {
        free(canvas);
        ds4_image_error(error, error_cap, "unable to allocate vision patches");
        return 0;
    }
    size_t index = 0;
    for (uint32_t patch_y = 0; patch_y < grid_height; patch_y++) {
        for (uint32_t patch_x = 0; patch_x < grid_width; patch_x++) {
            for (uint32_t channel = 0; channel < 3u; channel++) {
                for (uint32_t y = 0; y < 14u; y++) {
                    for (uint32_t x = 0; x < 14u; x++) {
                        const float *pixel = canvas +
                            ((size_t)(patch_y * 14u + y) * best_width +
                             patch_x * 14u + x) * 3u;
                        patches[index++] = pixel[channel];
                    }
                }
            }
        }
    }
    free(canvas);
    if (index != patch_values) {
        free(patches);
        ds4_image_error(error, error_cap,
                        "internal DeepSeek vision patch layout mismatch");
        return 0;
    }

    out->content_width = content_width;
    out->content_height = content_height;
    out->padded_width = best_width;
    out->padded_height = best_height;
    out->grid_width = grid_width;
    out->grid_height = grid_height;
    out->llm_grid_width = llm_width;
    out->llm_grid_height = llm_height;
    out->patch_count = patch_count;
    out->patches = patches;
    return 1;
}

int ds4_image_preprocess_deepseek4(
        ds4_deepseek4_image_patches *out, const ds4_image *image,
        char *error, size_t error_cap) {
    return ds4_image_preprocess_deepseek(out, image, error, error_cap, false);
}

int ds4_image_preprocess_deepseek41(
        ds4_deepseek4_image_patches *out, const ds4_image *image,
        char *error, size_t error_cap) {
    return ds4_image_preprocess_deepseek(out, image, error, error_cap, true);
}

int ds4_deepseek41_image_layout_build(
        ds4_deepseek4_image_layout *out, uint32_t height, uint32_t width,
        char *error, size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    const uint64_t count = (uint64_t)height * ((uint64_t)width + 1u) + 2u;
    if (!height || !width || count > 1024u) {
        ds4_image_error(error, error_cap, "invalid V4.1 image token grid");
        return 0;
    }
    uint8_t *types = malloc((size_t)count);
    uint32_t *perm = malloc((size_t)height * width * sizeof(*perm));
    if (!types || !perm) {
        free(types); free(perm);
        ds4_image_error(error, error_cap, "unable to allocate V4.1 image layout");
        return 0;
    }
    uint32_t token = 0, image = 0;
    types[token++] = DS4_DEEPSEEK4_IMAGE_START;
    for (uint32_t row = 0; row < height; row++) {
        for (uint32_t col = 0; col < width; col++) {
            types[token++] = DS4_DEEPSEEK4_IMAGE;
            perm[image] = image;
            image++;
        }
        types[token++] = DS4_DEEPSEEK4_IMAGE_NEWLINE;
    }
    types[token++] = DS4_DEEPSEEK4_IMAGE_END;
    *out = (ds4_deepseek4_image_layout){token, image, types, perm};
    return 1;
}

int ds4_deepseek4_image_layout_build(
        ds4_deepseek4_image_layout *out,
        uint32_t grid_height,
        uint32_t grid_width,
        uint32_t start_pos,
        char *error,
        size_t error_cap) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (grid_height == 0 || grid_width == 0) {
        ds4_image_error(error, error_cap, "invalid DeepSeek image token grid");
        return 0;
    }
    const uint32_t pad_height = grid_height & 1u;
    const uint32_t rows = grid_height + pad_height;
    const uint32_t row_length = grid_width + 1u;
    const uint32_t compress_pad = 3u - start_pos % 4u;
    const uint32_t pad_last = (((rows / 2u) * row_length) & 1u) * 2u;
    const uint64_t body_count64 = (uint64_t)rows * row_length;
    const uint64_t token_count64 =
        compress_pad + 1u + body_count64 + pad_last + 1u;
    const uint64_t image_count64 = (uint64_t)grid_height * grid_width;
    if (token_count64 > 384u || token_count64 > UINT32_MAX ||
        image_count64 > UINT32_MAX) {
        ds4_image_error(error, error_cap,
                        "DeepSeek image token layout exceeds its model limit");
        return 0;
    }

    const uint32_t body_count = (uint32_t)body_count64;
    uint8_t *body_types = malloc(body_count);
    int32_t *body_images = malloc((size_t)body_count * sizeof(*body_images));
    uint8_t *types = malloc((size_t)token_count64);
    uint32_t *perm = malloc((size_t)image_count64 * sizeof(*perm));
    if (!body_types || !body_images || !types || !perm) {
        free(body_types);
        free(body_images);
        free(types);
        free(perm);
        ds4_image_error(error, error_cap,
                        "unable to allocate DeepSeek image token layout");
        return 0;
    }

    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < row_length; column++) {
            const uint32_t index = row * row_length + column;
            if (row < grid_height && column < grid_width) {
                body_types[index] = DS4_DEEPSEEK4_IMAGE;
                body_images[index] = (int32_t)(row * grid_width + column);
            } else {
                body_types[index] = row < grid_height
                    ? DS4_DEEPSEEK4_IMAGE_NEWLINE
                    : DS4_DEEPSEEK4_IMAGE_PAD;
                body_images[index] = -1;
            }
        }
    }

    uint32_t type_pos = 0, perm_pos = 0;
    for (uint32_t i = 0; i < compress_pad; i++)
        types[type_pos++] = DS4_DEEPSEEK4_IMAGE_PAD;
    types[type_pos++] = DS4_DEEPSEEK4_IMAGE_START;
    for (uint32_t pair = 0; pair < rows / 2u; pair++) {
        for (uint32_t column = 0; column < row_length; column++) {
            for (uint32_t row_in_pair = 0; row_in_pair < 2u; row_in_pair++) {
                const uint32_t source =
                    (pair * 2u + row_in_pair) * row_length + column;
                types[type_pos++] = body_types[source];
                if (body_images[source] >= 0)
                    perm[perm_pos++] = (uint32_t)body_images[source];
            }
        }
    }
    for (uint32_t i = 0; i < pad_last; i++)
        types[type_pos++] = DS4_DEEPSEEK4_IMAGE_PAD;
    types[type_pos++] = DS4_DEEPSEEK4_IMAGE_END;
    free(body_images);
    free(body_types);
    if (type_pos != token_count64 || perm_pos != image_count64) {
        free(types);
        free(perm);
        ds4_image_error(error, error_cap,
                        "internal DeepSeek image token layout mismatch");
        return 0;
    }
    out->token_count = type_pos;
    out->image_count = perm_pos;
    out->types = types;
    out->perm = perm;
    return 1;
}

void ds4_deepseek4_image_patches_free(ds4_deepseek4_image_patches *patches) {
    if (!patches) return;
    free(patches->patches);
    memset(patches, 0, sizeof(*patches));
}

void ds4_deepseek4_image_layout_free(ds4_deepseek4_image_layout *layout) {
    if (!layout) return;
    free(layout->types);
    free(layout->perm);
    memset(layout, 0, sizeof(*layout));
}

int ds4_deepseek4_next_image_span(
        const int *tokens,
        uint32_t   token_count,
        uint32_t   vocab_size,
        uint32_t  *cursor,
        uint32_t  *block_start,
        uint32_t  *image_start,
        uint32_t  *image_end) {
    if (!tokens || !cursor || !block_start || !image_start || !image_end ||
        vocab_size == 0 || *cursor > token_count) return -1;

    uint32_t i = *cursor;
    while (i < token_count && tokens[i] >= 0 &&
           (uint32_t)tokens[i] < vocab_size) i++;
    if (i == token_count) {
        *cursor = i;
        return 0;
    }
    if (tokens[i] < 0) return -1;

    const uint32_t block = i;
    const int pad = (int)vocab_size + DS4_DEEPSEEK4_IMAGE_PAD;
    while (i < token_count && tokens[i] == pad) i++;
    if (i == token_count ||
        tokens[i] != (int)vocab_size + DS4_DEEPSEEK4_IMAGE_START) {
        return -1;
    }
    const uint32_t start = i++;
    while (i < token_count) {
        const int token = tokens[i];
        if (token < (int)vocab_size ||
            token > (int)vocab_size + DS4_DEEPSEEK4_IMAGE_END ||
            token == (int)vocab_size + DS4_DEEPSEEK4_IMAGE_START) {
            return -1;
        }
        if (token == (int)vocab_size + DS4_DEEPSEEK4_IMAGE_END) {
            *block_start = block;
            *image_start = start;
            *image_end = i;
            *cursor = i + 1u;
            return 1;
        }
        i++;
    }
    return -1;
}

int ds4_deepseek4_prefill_chunk(
        const int *tokens,
        uint32_t   token_count,
        uint32_t   vocab_size,
        uint32_t   pos0,
        uint32_t   cap,
        uint32_t  *chunk) {
    if (!tokens || !chunk || vocab_size == 0 || cap == 0 ||
        pos0 >= token_count) return 0;
    uint32_t limit = pos0 +
        (token_count - pos0 < cap ? token_count - pos0 : cap);
    uint32_t cursor = pos0;
    for (;;) {
        uint32_t block_start, image_start, image_end;
        const int parsed = ds4_deepseek4_next_image_span(
                tokens, token_count, vocab_size, &cursor,
                &block_start, &image_start, &image_end);
        (void)image_start;
        if (parsed < 0) return 0;
        if (parsed == 0 || block_start >= limit) break;
        const uint32_t after_image = image_end + 1u;
        if (after_image > limit) {
            if (after_image - pos0 > cap) {
                if (block_start == pos0) return 0;
                limit = block_start;
                break;
            }
            limit = after_image;
        }
    }
    *chunk = limit - pos0;
    return *chunk != 0;
}

int ds4_deepseek4_attention_bounds(
        const int *tokens,
        uint32_t   token_count,
        uint32_t   vocab_size,
        uint32_t   pos0,
        uint32_t   n_raw,
        uint32_t   window,
        uint32_t  *bounds) {
    if (!tokens || !bounds || token_count == 0 || vocab_size == 0 ||
        pos0 > UINT32_MAX - token_count || n_raw == 0 ||
        n_raw > pos0 + token_count) return 0;
    const uint32_t first_raw_pos = pos0 + token_count - n_raw;
    const uint32_t last_raw_pos = first_raw_pos + n_raw - 1u;
    for (uint32_t q = 0; q < token_count; q++) {
        const uint32_t qpos = pos0 + q;
        uint32_t lo = first_raw_pos;
        if (window != 0 && qpos + 1u > window) {
            const uint32_t window_lo = qpos + 1u - window;
            if (window_lo > lo) lo = window_lo;
        }
        bounds[2u * q] = lo;
        bounds[2u * q + 1u] = qpos < last_raw_pos ? qpos : last_raw_pos;
    }

    uint32_t cursor = 0;
    for (;;) {
        uint32_t block_start, image_start, image_end;
        const int parsed = ds4_deepseek4_next_image_span(
                tokens, token_count, vocab_size, &cursor,
                &block_start, &image_start, &image_end);
        (void)block_start;
        if (parsed < 0) return 0;
        if (parsed == 0) break;
        for (uint32_t q = image_start; q <= image_end; q++) {
            const uint32_t qpos = pos0 + q;
            const uint32_t left = q - image_start < 383u
                ? q - image_start : 383u;
            const uint32_t right = image_end - q < 384u
                ? image_end - q : 384u;
            const uint32_t left_add = window != 0 && left >= window
                ? left - (window - 1u) : 0u;
            uint32_t lo = 0;
            if (window != 0) {
                const uint32_t back = window - 1u + left_add;
                lo = qpos > back ? qpos - back : 0u;
            }
            uint32_t hi = qpos + right;
            if (lo < first_raw_pos) lo = first_raw_pos;
            if (hi > last_raw_pos) hi = last_raw_pos;
            bounds[2u * q] = lo;
            bounds[2u * q + 1u] = hi;
        }
    }
    return 1;
}
