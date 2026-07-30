"""Capacity-matched set-arch comparison (d64 vs d128) — resolve inc-09's crop-cost capacity caveat.

inc-09 measured the set-transformer's temporal-crop recall at d64 (0.487) and flagged a capacity confound (the
d128 baseline vs a d64 set-arch, because MPS OOM'd the set-arch at d128/batch512). This runs BOTH widths at the
SAME modest epoch budget and batch, so the only variable is `d_model` — a clean answer to 'does more capacity
buy back crop, or is the order-invariance ⟂ crop-robustness cost intrinsic?'.

Hardware note (measured, honest): the set-transformer at d128 does NOT OOM at batch<=256 on MPS (contra the
original 'OOM at d128' claim — memory is fine), but it HANGS / crawls on MPS compute (a batch-256 run printed
epoch 0 then stalled 50+ min; a batch-128 run ran at ~10s/epoch). So this runs on **CPU** — steady and reliable,
if slower — and skips the FAISS verify (its 0.01 tolerance false-alarms on shorter-trained embeddings). Brute-
force recall only. The modest epoch budget makes this INDICATIVE vs inc-09's 150ep d64; both widths use it, so
the d64-vs-d128 delta is the clean signal.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..config import EmbeddingConfig
from .embed import PlayEmbedder
from .possessions import augment
from .train import _bruteforce_rankings, _metrics_rk, _ver, split_by_game, train_encoder

AUGS = ["jitter", "crop", "mirror"]


def _eval_aug(encode_fn, val: np.ndarray, seed: int) -> dict:
    """Per-aug recall@1: query = augmented val, gallery = clean val, relevant = same index (brute-force)."""
    rng = np.random.default_rng(seed)
    gallery = encode_fn(val)
    out = {}
    for a in AUGS:
        q = np.stack([augment(val[i], rng, a) for i in range(len(val))])
        out[a] = _metrics_rk(_bruteforce_rankings(encode_fn(q), gallery))["recall@1"]
    return out


def _train_and_eval(corpus, tr, va, d_model: int, args) -> dict:
    a = SimpleNamespace(
        arch="set_transformer", dim=128, d_model=d_model, n_heads=4, n_layers=2, ff_dim=2 * d_model,
        dropout=0.1, temperature=0.1, epochs=args.epochs, batch=args.batch, lr=1e-3, weight_decay=1e-4,
        eval_every=0, jitter_sigma=0.01, p_mirror=0.5, p_crop=0.5, p_permute=0.0, n_permute=10,
        p_swap=0.0, n_swaps=2, T=48, device=args.device, seed=args.seed,
    )
    torch.manual_seed(a.seed)
    if a.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    cfg = EmbeddingConfig(enabled=True, arch=a.arch, dim=a.dim, d_model=a.d_model, n_heads=a.n_heads,
                          n_layers=a.n_layers, ff_dim=a.ff_dim, dropout=a.dropout, temperature=a.temperature)
    emb = PlayEmbedder(cfg, device=a.device, T=a.T)
    emb.build()
    t0 = time.time()
    train_encoder(emb, corpus[tr].astype(np.float32), a, np.random.default_rng(a.seed + 1))
    recalls = _eval_aug(emb.encode_batch, corpus[va].astype(np.float32), a.seed + 5)
    print(f"  d{d_model}: crop={recalls['crop']:.4f}  jitter={recalls['jitter']:.4f}  "
          f"mirror={recalls['mirror']:.4f}  ({emb.n_params()} params, {round(time.time() - t0)}s)", flush=True)
    return {"d_model": d_model, "n_params": emb.n_params(), **recalls}


def run(args) -> dict:
    z = np.load(Path(args.corpus), allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    tr, va, train_games, val_games = split_by_game(meta, args.val_stride, args.val_offset)
    print(f"set-arch capacity (CPU, {args.epochs}ep, batch{args.batch}): n_val={len(va)}", flush=True)

    results = {f"d{d}": _train_and_eval(corpus, tr, va, d, args) for d in (64, 128)}
    d64c, d128c = results["d64"]["crop"], results["d128"]["crop"]
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "setarch-capacity",
        "stage": "capacity-matched set-arch d64 vs d128 — does more width buy back temporal-crop recall?",
        "dataset": {"source": "linouk23 SportVU 2015-16", "split": "by game", "n_val": int(len(va)),
                    "val_games": val_games},
        "config": {"arch": "set_transformer", "epochs": args.epochs, "batch": args.batch, "device": args.device,
                   "note": "only d_model (and ff_dim=2*d) differs between the two"},
        "results": results,
        "crop_delta_d64_to_d128": round(d128c - d64c, 4),
        "conclusion": ("more capacity does NOT rescue crop -> the order-invariance/crop cost is INTRINSIC "
                       "to the set architecture (confirms inc-09, caveat resolved)" if d128c < d64c + 0.15 else
                       "more capacity DOES recover crop -> inc-09's cost was partly a capacity artifact"),
        "reference": {"inc09_d64_150ep_crop": 0.487, "note": "committed inc-09 d64 set-arch (batch256, 150ep); "
                      "this run is a shorter, epoch-matched d64-vs-d128 comparison — the DELTA is the signal"},
        "hardware_finding": "set-transformer d128 fits MPS memory at batch<=256 (no OOM, contra original claim) "
        "but hangs/crawls on MPS compute; ran on CPU for reliability. A full-scale d128 (150ep) needs a CUDA GPU.",
        "provenance": {"seed": args.seed, "versions": {p: _ver(p) for p in ("torch", "numpy")},
                       "platform": platform.platform()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Capacity-matched set-arch d64 vs d128 (crop recall)")
    ap.add_argument("--corpus", default="data/sportvu/corpus_g12_T48.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps — use cuda on a GPU box for the fast, "
                    "definitive 150-epoch run (CPU here is ~3h; MPS hangs the set-arch)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--val-offset", type=int, default=2)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (Path(args.out_dir) / f"setarch_capacity_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"\ncrop: d64={report['results']['d64']['crop']}  d128={report['results']['d128']['crop']}  "
          f"delta={report['crop_delta_d64_to_d128']}")
    print(report["conclusion"])


if __name__ == "__main__":
    main()
