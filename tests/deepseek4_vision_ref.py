#!/usr/bin/env python3
"""Float64 reference for the DeepSeek V4 Flash Vision-Exp image encoder.

The encoder is defined with a BF16 rounding after every operation.  This
script evaluates that definition in float64 with the same rounding points and
checks the tensors a ROCm build writes with DS4_ROCM_VISION_DUMP_DIR=DIR (one
image per process):

  --ops    every operation from the build's own inputs to that operation.
           A correct kernel differs from float64 only where its F32
           arithmetic lands on the other side of a BF16 rounding boundary:
           a few values in ten thousand, and no row off by more than a few
           such values.  A GEMM is reported by the error of its F32 output.
  --chain  the whole encoder from the patches.  BF16 rounding amplifies any
           last-bit difference to the quantization noise floor within a few
           operations, so every correct build ends a few percent (relative
           RMS) from the float64 result, and two correct builds are that far
           from each other.  Most of that distance sits in one to three
           unstable output rows per image, so it varies from 1% to 9% between
           images and between equally valid summation orders.  This only
           tells a working build from a broken one.

    python3 tests/deepseek4_vision_ref.py ENCODER.gguf DIR [--ops] [--chain]
        [--x-dir DIR2]...

--x-dir (repeatable) takes the last residual stream (x32.bin, F32 or BF16)
and out00.bin from another directory, to run --chain on builds that only write
those.
"""

import argparse
import os
import struct
import sys

import numpy as np

EMBD, HEADS, HEAD_DIM, MLP, PATCH, OUT = 1024, 16, 64, 2816, 588, 4096
LAYERS = 32


def read_gguf_tensors(path):
    """Return {name: (shape, type, absolute offset)} of a GGUF v3 file."""
    with open(path, "rb") as fp:
        data = fp.read(1 << 24)
    if data[:4] != b"GGUF":
        raise SystemExit("not a GGUF file: " + path)
    pos = 8   # magic, version

    def take(fmt):
        nonlocal pos
        value = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return value[0]

    def string():
        nonlocal pos
        n = take("Q")
        s = data[pos:pos + n].decode()
        pos += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
              10: "Q", 11: "q", 12: "d"}

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            item = take("I")
            return [value(item) for _ in range(take("Q"))]
        return take(scalar[kind])

    n_tensors, n_kv = take("Q"), take("Q")
    alignment = 32
    for _ in range(n_kv):
        key = string()
        v = value(take("I"))
        if key == "general.alignment":
            alignment = v
    tensors = {}
    for _ in range(n_tensors):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        kind, offset = take("I"), take("Q")
        tensors[name] = (tuple(reversed(dims)), kind, offset)
    base = (pos + alignment - 1) // alignment * alignment
    return {k: (s, t, base + o) for k, (s, t, o) in tensors.items()}


def bf16_to_f64(words):
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def rnd(x):
    """Round float64 to the nearest BF16 value, ties to even."""
    m, e = np.frexp(x)
    return np.ldexp(np.rint(m * 256.0) / 256.0, e)


class Encoder:
    """The encoder, one method per operation of the GPU pipeline."""

    def __init__(self, gguf, pos_y, pos_x, rope):
        self.tensors = read_gguf_tensors(gguf)
        self.map = np.memmap(gguf, dtype=np.uint16, mode="r")
        # Pair (lane, lane + 32) of a head; lanes 0..15 use y, 16..31 use x.
        self.cs = np.concatenate([rope[pos_y, :, 0], rope[pos_x, :, 0]], 1)[:, None]
        self.sn = np.concatenate([rope[pos_y, :, 1], rope[pos_x, :, 1]], 1)[:, None]

    def w(self, name):
        shape, kind, offset = self.tensors[name]
        if kind != 30:
            raise SystemExit(name + " is not BF16")
        count = int(np.prod(shape))
        return bf16_to_f64(np.asarray(
            self.map[offset // 2:offset // 2 + count])).reshape(shape)

    def gemm(self, x, name):
        return x @ self.w(name).T

    def rms(self, x, name):
        inv = 1.0 / np.sqrt((x * x).mean(axis=1, keepdims=True) +
                            float(np.float32(1.0e-6)))
        return rnd(x * inv * self.w(name))

    def patch_bias(self, f):
        return rnd(f + self.w("vision.patch_embed.proj.bias"))

    def qkv_rope(self, f, name):
        """F32 GEMM output -> q, k, v as [row][head][64]."""
        rows = f.shape[0]
        qkv = rnd(f + self.w(name))
        q, k, v = (qkv[:, i * EMBD:(i + 1) * EMBD].reshape(rows, HEADS, 2, 32)
                   for i in range(3))

        def rotate(t):
            return np.stack(
                [rnd(t[:, :, 0] * self.cs - t[:, :, 1] * self.sn),
                 rnd(t[:, :, 1] * self.cs + t[:, :, 0] * self.sn)],
                axis=2).reshape(rows, HEADS, HEAD_DIM)

        return rotate(q), rotate(k), v.reshape(rows, HEADS, HEAD_DIM)

    @staticmethod
    def attention(q, k, v):
        rows = q.shape[0]
        out = np.empty((rows, HEADS, HEAD_DIM))
        for h in range(HEADS):
            s = (q[:, h] @ k[:, h].T) * 0.125
            p = np.exp(s - s.max(axis=1, keepdims=True))
            out[:, h] = (p @ v[:, h]) / p.sum(axis=1, keepdims=True)
        return rnd(out).reshape(rows, EMBD)

    def attn_residual(self, f, x, name):
        return rnd(f + self.w(name) + x)

    @staticmethod
    def swiglu(f):
        h = rnd(f)
        gate, up = h[:, :MLP], h[:, MLP:]
        return rnd(gate / (1.0 + np.exp(-gate)) * up)

    @staticmethod
    def mlp_residual(f, x):
        return rnd(rnd(f) + x)

    def block(self, il, x):
        name = "vision.blocks.%d." % il
        q, k, v = self.qkv_rope(
            self.gemm(self.rms(x, name + "norm1.weight"), name + "attn.wqkv.weight"),
            name + "attn.wqkv.bias")
        x = self.attn_residual(
            self.gemm(self.attention(q, k, v), name + "attn.wo.weight"),
            x, name + "attn.wo.bias")
        mid = self.swiglu(self.gemm(self.rms(x, name + "norm2.weight"),
                                    name + "mlp.w1.weight"))
        return self.mlp_residual(self.gemm(mid, name + "mlp.w2.weight"), x)

    @staticmethod
    def align_reorder(n, grid_h, grid_w):
        out_h, out_w = (grid_h + 2) // 3, (grid_w + 2) // 3
        padded = np.zeros((out_h * 3, out_w * 3, EMBD))
        padded[:grid_h, :grid_w] = n.reshape(grid_h, grid_w, EMBD)
        # [block_y][block_x][channel][3x3 position]
        return padded.reshape(out_h, 3, out_w, 3, EMBD).transpose(
            0, 2, 4, 1, 3).reshape(out_h * out_w, EMBD * 9)

    def gelu_bias(self, f):
        a = rnd(f + self.w("aligner.w1.bias"))
        z = a * float(np.float32(0.7071067811865475))
        t = 1.0 / (1.0 + float(np.float32(0.3275911)) * np.abs(z))
        c = [float(np.float32(c)) for c in
             (1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592)]
        poly = ((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t
        return rnd(0.5 * a * (1.0 + np.sign(z) * (1.0 - poly * np.exp(-z * z))))

    def output_bias(self, f):
        return rnd(f + self.w("aligner.w2.bias"))

    def project(self, x, grid_h, grid_w):
        a = self.align_reorder(self.rms(x, "vision.norm.weight"), grid_h, grid_w)
        return self.output_bias(self.gemm(
            self.gelu_bias(self.gemm(a, "aligner.w1.weight")), "aligner.w2.weight"))


def bf16_word(x):
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.int64)


class Report:
    """Worst case per operation over all blocks."""

    def __init__(self):
        self.rows = {}

    def values(self, label, ideal, got):
        """A rounded tensor: the fraction of BF16 words that differ, and the
        largest difference relative to the RMS of the tensor."""
        diff = (bf16_word(ideal) != bf16_word(got)).mean()
        a2, b2 = ideal.reshape(ideal.shape[0], -1), got.reshape(got.shape[0], -1)
        worst = (np.linalg.norm(b2 - a2, axis=1) /
                 np.maximum(np.linalg.norm(a2, axis=1), 1e-30)).max()
        row = self.rows.setdefault(label, [0.0, 0.0, 0.0, 0])
        row[0] = max(row[0], diff)
        row[1] = max(row[1], worst)
        row[2] += diff
        row[3] += 1

    def gemm(self, label, ideal, got):
        """An F32 GEMM output: its error, and how often that error moves a
        value across a BF16 rounding boundary."""
        scale = np.sqrt((ideal * ideal).mean())
        err = np.abs(got - ideal)
        row = self.rows.setdefault(label, [0.0, 0.0, 0.0, 0])
        row[0] = max(row[0], np.sqrt((err * err).mean()) / scale)
        row[1] = max(row[1], (bf16_word(rnd(ideal)) != bf16_word(rnd(got))).mean())
        row[2] = max(row[2], (np.linalg.norm(err, axis=1) /
                              np.linalg.norm(ideal, axis=1)).max())
        row[3] += 1

    def show(self):
        for label, row in self.rows.items():
            if label.startswith("gemm"):
                print("%-22s rel RMS error %.2e, rounds differently %.4f%%, "
                      "worst row error %.1e" % (
                          label, row[0], 100.0 * row[1], row[2]))
            else:
                print("%-22s differ: worst %.4f%%, mean %.4f%%, worst row "
                      "error %.1e" % (
                          label, 100.0 * row[0], 100.0 * row[2] / row[3], row[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("dump")
    ap.add_argument("--x-dir", action="append")
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--chain", action="store_true")
    args = ap.parse_args()

    grid_h, grid_w = (int(v) for v in np.fromfile(
        os.path.join(args.dump, "grid00.bin"), dtype=np.uint32))
    rows = grid_h * grid_w
    padded = (rows + 31) // 32 * 32   # the build pads to whole GEMM tiles
    pos_y, pos_x = np.divmod(np.arange(rows), grid_w)

    def raw(name, directory=args.dump):
        return np.fromfile(os.path.join(directory, name + ".bin"), dtype=np.uint8)

    def bf16(name, width=EMBD):
        return bf16_to_f64(raw(name).view(np.uint16)).reshape(padded, width)[:rows]

    def f32(name, width):
        return raw(name).view(np.float32).astype(np.float64).reshape(-1, width)

    def heads(name):   # [head][row][64] -> [row][head][64]
        return bf16_to_f64(raw(name).view(np.uint16)).reshape(
            HEADS, padded, HEAD_DIM)[:, :rows].transpose(1, 0, 2)

    def state(name, directory):   # residual stream of either build
        data = raw(name, directory)
        if data.size == rows * EMBD * 4:
            return data.view(np.float32).astype(np.float64).reshape(rows, EMBD)
        return bf16_to_f64(data.view(np.uint16)).reshape(-1, EMBD)[:rows]

    def output(directory):
        return raw("out00", directory).view(np.float32).astype(
            np.float64).reshape(-1, OUT)

    rope = raw("rope00").view(np.float32).astype(np.float64).reshape(-1, 16, 2)
    enc = Encoder(args.gguf, pos_y, pos_x, rope)
    patch = bf16("patch00", PATCH)
    out = output(args.dump)
    print("grid %dx%d, %d rows, %d output rows" % (grid_h, grid_w, rows, out.shape[0]))

    status = 0
    if args.ops:
        rep = Report()
        f = f32("patch_w00", EMBD)[:rows]
        rep.gemm("gemm patch", enc.gemm(patch, "vision.patch_embed.proj.weight"), f)
        rep.values("patch bias", enc.patch_bias(f), bf16("x00"))
        for il in range(LAYERS):
            name, tag = "vision.blocks.%d." % il, "%02d" % il
            x = bf16("x" + tag)
            n1 = bf16("norm1_" + tag)
            rep.values("norm1", enc.rms(x, name + "norm1.weight"), n1)
            f = f32("qkv_" + tag, 3 * EMBD)[:rows]
            rep.gemm("gemm qkv", enc.gemm(n1, name + "attn.wqkv.weight"), f)
            q, k = heads("q_" + tag), heads("k_" + tag)
            v = bf16_to_f64(raw("vt_" + tag).view(np.uint16)).reshape(
                HEADS, HEAD_DIM, padded)[:, :, :rows].transpose(2, 0, 1)
            iq, ik, iv = enc.qkv_rope(f, name + "attn.wqkv.bias")
            rep.values("qkv rope q", iq, q)
            rep.values("qkv rope k", ik, k)
            rep.values("qkv rope v", iv, v)
            attn = bf16("attn_" + tag)
            rep.values("attention", enc.attention(q, k, v), attn)
            f = f32("proj_" + tag, EMBD)[:rows]
            rep.gemm("gemm attention out", enc.gemm(attn, name + "attn.wo.weight"), f)
            xa = bf16("xattn_" + tag)
            rep.values("attention residual",
                       enc.attn_residual(f, x, name + "attn.wo.bias"), xa)
            n2 = bf16("norm2_" + tag)
            rep.values("norm2", enc.rms(xa, name + "norm2.weight"), n2)
            f = f32("w1_" + tag, 2 * MLP)[:rows]
            rep.gemm("gemm mlp w1", enc.gemm(n2, name + "mlp.w1.weight"), f)
            mid = bf16("mid_" + tag, MLP)
            rep.values("swiglu", enc.swiglu(f), mid)
            f = f32("w2_" + tag, EMBD)[:rows]
            rep.gemm("gemm mlp w2", enc.gemm(mid, name + "mlp.w2.weight"), f)
            rep.values("mlp residual", enc.mlp_residual(f, xa),
                       bf16("x%02d" % (il + 1)))
            print("block %d checked" % il, flush=True)
        n = bf16("norm00")
        rep.values("final norm", enc.rms(bf16("x%02d" % LAYERS), "vision.norm.weight"), n)
        out_rows = out.shape[0]
        a = bf16_to_f64(raw("align_in00").view(np.uint16)).reshape(out_rows, 9 * EMBD)
        rep.values("aligner reorder", enc.align_reorder(n, grid_h, grid_w), a)
        f = f32("align_w1_00", OUT)
        rep.gemm("gemm aligner w1", enc.gemm(a, "aligner.w1.weight"), f)
        g = bf16_to_f64(raw("align_mid00").view(np.uint16)).reshape(out_rows, OUT)
        rep.values("aligner gelu", enc.gelu_bias(f), g)
        f = f32("align_w2_00", OUT)
        rep.gemm("gemm aligner w2", enc.gemm(g, "aligner.w2.weight"), f)
        rep.values("output bias", enc.output_bias(f), out)
        rep.show()
        ops = [row for label, row in rep.rows.items()
               if not label.startswith("gemm")]
        if max(row[0] for row in ops) > 2.0e-3 or max(row[1] for row in ops) > 2.0e-2:
            print("FAIL: an operation is not the float64 result up to F32 "
                  "rounding")
            status = 1
    if args.chain:
        x = enc.patch_bias(enc.gemm(patch, "vision.patch_embed.proj.weight"))
        for il in range(LAYERS):
            x = enc.block(il, x)
        ideal = enc.project(x, grid_h, grid_w)
        for directory in args.x_dir or [args.dump]:
            err = state("x%02d" % LAYERS, directory) - x
            got = output(directory)
            oerr = got - ideal
            cos = (ideal * got).sum(1) / (np.linalg.norm(ideal, axis=1) *
                                          np.linalg.norm(got, axis=1))
            rel = np.sqrt((oerr * oerr).mean() / (ideal * ideal).mean())
            row = np.linalg.norm(oerr, axis=1) / np.linalg.norm(ideal, axis=1)
            print("chain %s: last block rel RMS %.2e; output rel RMS %.2e, "
                  "row error median %.2e p90 %.2e max %.2e, "
                  "row cosine min %.6f mean %.7f" % (
                      os.path.basename(directory.rstrip("/")),
                      np.sqrt((err * err).mean() / (x * x).mean()), rel,
                      np.median(row), np.percentile(row, 90), row.max(),
                      cos.min(), cos.mean()))
            if rel > 0.2 or cos.mean() < 0.99:
                print("FAIL: the output is not the float64 result up to "
                      "BF16 noise")
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
