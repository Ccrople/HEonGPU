#!/usr/bin/env python3
# Copyright 2026
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch the REAL Llama-3-8B parameters for one transformer block and compute
# that block's real input activations, for profile_llama3_rect_boot.cpp.
#
# WHY A PREFIX RUN AND NOT JUST A DOWNLOAD
# ----------------------------------------
# The profile measures one block. Feeding block L real weights but random
# activations would miss the entire point of Section 3.1.1: the outliers the
# rotations and the sink prefix exist to mitigate are a property of the REAL
# residual stream, and the stream reaching block L is the output of blocks
# 0 .. L-1 run on real weights. So this script runs the true model prefix --
# RoPE included, in fp32 -- and writes the layer-L input alongside the layer-L
# weights. --scan prints every layer's input bound first, which is how L is
# chosen rather than assumed.
#
# The prefix here is the TRUE Llama-3, not the profiled circuit: the FHE block
# omits RoPE, and that is a documented simplification of the circuit, not of
# the data it runs on.
#
# TWO INPUTS, NOT ONE
# -------------------
# input.f32 embeds the calibration text behind Section 3.1.1's sink prefix
# (BOS and a delimiter run); input_nosink.f32 embeds the same text without it.
# The profile calibrates on both to measure what the prefix is worth, which is
# a number the paper states and this pipeline can check.
#
# WEIGHT LAYOUT
# -------------
# HuggingFace stores a projection as [out_features, in_features]. The rect
# operator wants row-major in_channels x out_channels -- the transpose -- so
# every matrix is transposed HERE, once, on the way out. The C++ side reads
# flat little-endian f32 and does no reshaping.
#
# Usage, on a machine with numpy + tokenizers and ~17 GiB of cache space:
#
#   python3 fetch_llama3_weights.py --cache /tmp/llama3_cache --scan
#   python3 fetch_llama3_weights.py --cache /tmp/llama3_cache \
#       --layer 2 --dest /path/to/llama3_real
#
# The model comes from an ungated mirror of Meta-Llama-3-8B; --mirror points
# elsewhere if that ever moves.

import argparse
import json
import os
import struct
import sys
import urllib.request

import numpy as np

VOCAB = 128256
D_MODEL = 4096
HIDDEN = 14336
LAYERS = 32
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 500000.0
RMS_EPS = 1e-5
BOS_ID = 128000

# ~190 words of neutral prose, comfortably past the 124 text tokens a
# 128-token window needs behind a 4-token sink prefix.
CALIBRATION_TEXT = (
    "The harbor town woke slowly under a thin autumn fog. Fishing boats "
    "rocked against their moorings while gulls circled the empty market "
    "stalls, waiting for the first crates of the morning catch. Along the "
    "sea wall an old keeper checked each lamp in turn, wiping salt from the "
    "glass and noting the wind in a small leather book he had carried for "
    "thirty years. By eight the bakery had drawn a quiet line of customers, "
    "and the smell of bread moved down the narrow streets faster than any "
    "news. School children crossed the square in twos and threes, trading "
    "cards and arguments, their voices bright against the grey water. A "
    "ferry sounded twice in the channel, answered by nothing but its own "
    "echo. In the afternoon the fog lifted enough to show the far headland, "
    "green and patient, and the fishermen argued pleasantly about weather "
    "they could not change. When evening came the lamps went on one by one, "
    "each small light claiming its own piece of the dark, and the town "
    "settled into the steady rhythm it had kept for longer than anyone "
    "could remember."
)
# The sink prefix: BOS plus a short delimiter run. These are the tokens that
# consistently attract the attention-sink mass, they carry no user
# information, and prepending them is Section 3.1.1's first mitigation.
SINK_TEXT = ".\n\n"


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    tmp = dest + ".part"
    print(f"[fetch] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fideslib-prof"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    os.rename(tmp, dest)


def load_safetensors_headers(cache, shards):
    headers = {}
    for shard in shards:
        path = os.path.join(cache, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[shard] = (json.loads(f.read(n)), 8 + n)
    return headers


def read_tensor(cache, index, headers, name):
    shard = index["weight_map"][name]
    header, base = headers[shard]
    info = header[name]
    if info["dtype"] != "BF16":
        raise RuntimeError(f"{name}: expected BF16, got {info['dtype']}")
    begin, end = info["data_offsets"]
    with open(os.path.join(cache, shard), "rb") as f:
        f.seek(base + begin)
        raw = f.read(end - begin)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    # bf16 -> f32 is a left shift into the high half of the word.
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(info["shape"]).astype(np.float32)


def rms_norm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + RMS_EPS) * w


def rope_tables(positions):
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64)
                                / HEAD_DIM))
    ang = np.outer(positions, inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin):
    # x: [tokens, heads, head_dim], HF convention: rotate_half pairs channel
    # c with c + head_dim/2.
    h = HEAD_DIM // 2
    x1, x2 = x[..., :h], x[..., h:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def layer_forward(x, w, cos, sin):
    """One true Llama-3 block in fp32. x: [tokens, 4096]."""
    tokens = x.shape[0]
    h = rms_norm(x, w["input_layernorm"])
    q = (h @ w["q"].T).reshape(tokens, HEADS, HEAD_DIM)
    k = (h @ w["k"].T).reshape(tokens, KV_HEADS, HEAD_DIM)
    v = (h @ w["v"].T).reshape(tokens, KV_HEADS, HEAD_DIM)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    group = HEADS // KV_HEADS
    out = np.empty((tokens, HEADS, HEAD_DIM), dtype=np.float32)
    causal = np.tril(np.ones((tokens, tokens), dtype=bool))
    for head in range(HEADS):
        kv = head // group
        scores = (q[:, head, :] @ k[:, kv, :].T) / np.sqrt(HEAD_DIM)
        scores = np.where(causal, scores, -np.inf)
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        out[:, head, :] = p @ v[:, kv, :]
    x = x + out.reshape(tokens, HEADS * HEAD_DIM) @ w["o"].T
    h = rms_norm(x, w["post_attention_layernorm"])
    gate = h @ w["gate"].T
    up = h @ w["up"].T
    silu = gate / (1.0 + np.exp(-gate))
    x = x + (silu * up) @ w["down"].T
    return x


def layer_weights(cache, index, headers, layer):
    p = f"model.layers.{layer}."
    return {
        "q": read_tensor(cache, index, headers, p + "self_attn.q_proj.weight"),
        "k": read_tensor(cache, index, headers, p + "self_attn.k_proj.weight"),
        "v": read_tensor(cache, index, headers, p + "self_attn.v_proj.weight"),
        "o": read_tensor(cache, index, headers, p + "self_attn.o_proj.weight"),
        "gate": read_tensor(cache, index, headers, p + "mlp.gate_proj.weight"),
        "up": read_tensor(cache, index, headers, p + "mlp.up_proj.weight"),
        "down": read_tensor(cache, index, headers, p + "mlp.down_proj.weight"),
        "input_layernorm": read_tensor(cache, index, headers,
                                       p + "input_layernorm.weight"),
        "post_attention_layernorm": read_tensor(
            cache, index, headers, p + "post_attention_layernorm.weight"),
    }


def tokenize(cache, text):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(cache, "tokenizer.json")) \
        .encode(text).ids


def build_prompt(cache, tokens, with_sink):
    ids = [BOS_ID]
    if with_sink:
        ids += tokenize(cache, SINK_TEXT)
    ids += tokenize(cache, CALIBRATION_TEXT)
    if len(ids) < tokens:
        raise RuntimeError(
            f"Calibration text gives {len(ids)} tokens, need {tokens}")
    return ids[:tokens]


def run_prefix(cache, index, headers, ids, upto, report=None):
    embed = read_tensor(cache, index, headers, "model.embed_tokens.weight")
    x = embed[np.asarray(ids)].astype(np.float32)
    del embed
    cos, sin = rope_tables(np.arange(len(ids)))
    for layer in range(upto):
        if report is not None:
            report(layer, x)
        x = layer_forward(x, layer_weights(cache, index, headers, layer),
                          cos, sin)
    if report is not None:
        report(upto, x)
    return x


def write_f32(path, array):
    np.ascontiguousarray(array, dtype="<f4").tofile(path)
    print(f"[write] {path}  {array.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="Shard cache; ~17 GiB on first run")
    ap.add_argument("--dest", help="Bundle output directory")
    ap.add_argument("--layer", type=int, default=2,
                    help="Block whose weights and input to export")
    ap.add_argument("--tokens", type=int, default=128,
                    help="Window length; must equal the profile's d")
    ap.add_argument("--mirror", default="NousResearch/Meta-Llama-3-8B")
    ap.add_argument("--scan", action="store_true",
                    help="Print each layer's input bound and exit")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    base = f"https://huggingface.co/{args.mirror}/resolve/main"
    fetch(f"{base}/model.safetensors.index.json",
          os.path.join(args.cache, "index.json"))
    fetch(f"{base}/tokenizer.json", os.path.join(args.cache, "tokenizer.json"))
    with open(os.path.join(args.cache, "index.json")) as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        fetch(f"{base}/{shard}", os.path.join(args.cache, shard))
    headers = load_safetensors_headers(args.cache, shards)

    ids = build_prompt(args.cache, args.tokens, with_sink=True)
    ids_nosink = build_prompt(args.cache, args.tokens, with_sink=False)

    if args.scan:
        print(f"{'layer':>5} {'|input|max':>12} {'rms lo':>10} {'rms hi':>10}")

        def report(layer, x):
            sums = np.sum(x * x, axis=-1)
            print(f"{layer:>5} {np.abs(x).max():>12.2f} "
                  f"{np.sqrt(sums.min() / D_MODEL):>10.4f} "
                  f"{np.sqrt(sums.max() / D_MODEL):>10.4f}")

        run_prefix(args.cache, index, headers, ids, LAYERS, report)
        return

    if not args.dest:
        ap.error("--dest is required unless --scan")
    os.makedirs(args.dest, exist_ok=True)

    x = run_prefix(args.cache, index, headers, ids, args.layer)
    x_nosink = run_prefix(args.cache, index, headers, ids_nosink, args.layer)
    w = layer_weights(args.cache, index, headers, args.layer)

    write_f32(os.path.join(args.dest, "input.f32"), x)
    write_f32(os.path.join(args.dest, "input_nosink.f32"), x_nosink)
    for name in ("q", "k", "v", "o", "gate", "up", "down"):
        write_f32(os.path.join(args.dest, f"w{name}.f32"), w[name].T)
    write_f32(os.path.join(args.dest, "attn_norm.f32"), w["input_layernorm"])
    write_f32(os.path.join(args.dest, "ffn_norm.f32"),
              w["post_attention_layernorm"])

    with open(os.path.join(args.dest, "meta.txt"), "w") as f:
        f.write(f"layer {args.layer}\n")
        f.write(f"tokens {args.tokens}\n")
        f.write(f"channels {D_MODEL}\n")
        f.write(f"kv_channels {KV_HEADS * HEAD_DIM}\n")
        f.write(f"head_dim {HEAD_DIM}\n")
        f.write(f"hidden {HIDDEN}\n")
        f.write(f"sink_tokens {1 + len(tokenize(args.cache, SINK_TEXT))}\n")
        f.write(f"mirror {args.mirror}\n")
        f.write("prompt " + " ".join(map(str, ids)) + "\n")
        f.write("prompt_nosink " + " ".join(map(str, ids_nosink)) + "\n")
    print("[done] bundle in", args.dest)


if __name__ == "__main__":
    main()
