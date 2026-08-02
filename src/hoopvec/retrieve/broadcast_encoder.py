"""Broadcast-domain encoder — make the TRAINED encoder valid end-to-end, not just the floor.

The end2end result (floor r@1 0.80) had to use the hand-feature floor because the SportVU-trained transformer
is out-of-domain on IMAGE-coordinate broadcast tracks (top-down court vs broadcast-perspective plane). This
trains an encoder IN-DOMAIN: an augmentation-SSL encoder on the SportsMOT image-coordinate window tensors,
held out BY GAME (train on 3 games' GT windows, evaluate reconstructed-vs-GT retrieval on the 4th game's
windows — never trained on the test game). Augmentations target the reconstruction error that dominates
(inc-07: association) — jitter (localization) + order-perturbation (fragmentation / id-swap) — plus temporal
crop. Same architecture and objective as inc-06b; the only change is the DOMAIN it's trained on.

Honest scope: broadcast supervision here is tiny (a few hundred windows from a handful of games), so this is a
proof-of-concept + an honest measurement (does an in-domain trained encoder beat the coordinate-agnostic floor
on real reconstructed tracks?), not a production encoder. The floor 0.80 is the bar. Reported as-is.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..config import EmbeddingConfig
from ..ingest.frames import read_seqinfo
from .embed import PlayEmbedder
from .end2end import FRAMES_ROOT, GT_ROOT, TRACKERS_ROOT, _load_mot
from .reconstruct import frame_windows, tracks_to_tensor
from .run import features
from .train import _faiss_rankings, _metrics_rk, _ver, train_encoder


def _game(seq: str) -> str:
    return seq.split("_c")[0]                                    # v_00HRwkvvjtQ_c007 -> v_00HRwkvvjtQ


def _seq_windows(seq: str, tracker: str, T: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    info = read_seqinfo(FRAMES_ROOT / seq)
    w, h = info["imWidth"] or 1280, info["imHeight"] or 720
    gt = _load_mot(GT_ROOT / seq / "gt" / "gt.txt")
    recon = _load_mot(TRACKERS_ROOT / tracker / "data" / f"{seq}.txt")
    gts, recons = [], []
    for f0, f1 in frame_windows(gt, window):
        gts.append(tracks_to_tensor(gt, f0, f1, T, w, h))
        recons.append(tracks_to_tensor(recon, f0, f1, T, w, h))
    return (np.stack(gts) if gts else np.zeros((0, T, 11, 2)),
            np.stack(recons) if recons else np.zeros((0, T, 11, 2)))


def _args(device: str, seed: int, epochs: int) -> SimpleNamespace:
    """In-domain SSL recipe. p_mirror=0 (image space isn't court-symmetric); jitter + temporal crop +
    order-perturbation, so the encoder is robust to the reconstruction error that dominates (inc-07)."""
    return SimpleNamespace(
        arch="trajectory_transformer", dim=128, d_model=128, n_heads=4, n_layers=2, ff_dim=256, dropout=0.1,
        temperature=0.1, epochs=epochs, batch=64, lr=1e-3, weight_decay=1e-4, eval_every=0,
        jitter_sigma=0.02, p_mirror=0.0, p_crop=0.5, p_permute=0.3, n_permute=10, p_swap=0.5, n_swaps=2,
        T=48, device=device, seed=seed,
    )


def run(args) -> dict:
    seqs = sorted(p.name for p in GT_ROOT.iterdir() if (p / "gt" / "gt.txt").is_file())
    seqs = [s for s in seqs if (TRACKERS_ROOT / args.tracker / "data" / f"{s}.txt").is_file()]
    games = sorted({_game(s) for s in seqs})
    val_game = args.val_game or games[-1]
    train_seqs = [s for s in seqs if _game(s) != val_game]
    val_seqs = [s for s in seqs if _game(s) == val_game]

    train_gt = np.concatenate([_seq_windows(s, args.tracker, args.T, args.window)[0] for s in train_seqs])
    val_gt = np.concatenate([_seq_windows(s, args.tracker, args.T, args.window)[0] for s in val_seqs])
    val_recon = np.concatenate([_seq_windows(s, args.tracker, args.T, args.window)[1] for s in val_seqs])

    a = _args(args.device, args.seed, args.epochs)
    torch.manual_seed(a.seed)
    if a.device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(a.seed)
    cfg = EmbeddingConfig(enabled=True, arch=a.arch, dim=a.dim, d_model=a.d_model, n_heads=a.n_heads,
                          n_layers=a.n_layers, ff_dim=a.ff_dim, dropout=a.dropout, temperature=a.temperature)
    emb = PlayEmbedder(cfg, device=a.device, T=a.T)
    emb.build()
    _, secs = train_encoder(emb, train_gt.astype(np.float32), a, np.random.default_rng(a.seed + 3))

    # reconstructed-vs-GT retrieval on the held-out game: floor vs the in-domain trained encoder
    floor = _metrics_rk(_faiss_rankings(features(val_recon), features(val_gt)))
    genc, qenc = emb.encode_batch(val_gt), emb.encode_batch(val_recon)
    trained = _metrics_rk(_faiss_rankings(qenc, genc))
    trained_self = _metrics_rk(_faiss_rankings(genc, genc))     # sanity: distinct windows -> ~1.0

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "broadcast-domain-encoder",
        "stage": "in-domain SSL encoder on image-coordinate broadcast tracks, reconstructed-vs-GT on a held-out game",
        "dataset": {"source": "SportsMOT basketball-val", "tracker": args.tracker, "held_out_game": val_game,
                    "train_games": [g for g in games if g != val_game], "n_train_windows": int(len(train_gt)),
                    "n_val_windows": int(len(val_gt)), "chance_recall@1": round(1.0 / max(1, len(val_gt)), 4)},
        "method": {"query": "reconstructed window tensor", "gallery": "GT window tensors (held-out game)",
                   "encoder_train": "augmentation-SSL (jitter + crop + order-perturb, p_mirror=0) on TRAIN-game "
                   "GT windows; image coords", "retrieval": "FAISS IndexFlatIP (cosine)"},
        "results": {"floor_recon_vs_gt": floor, "trained_broadcast_recon_vs_gt": trained,
                    "trained_self_retrieval_sanity": trained_self},
        "provenance": {"seed": args.seed, "device": args.device, "epochs": args.epochs, "train_seconds": secs,
                       "versions": {p: _ver(p) for p in ("torch", "numpy", "faiss-cpu")},
                       "platform": platform.platform()},
        "notes": "Tiny broadcast supervision (a few hundred windows from 3 games) -> proof-of-concept + honest "
        "measurement, not a production encoder. Only change vs inc-06b is the training DOMAIN (image-coord "
        "broadcast tensors, not SportVU court). Floor is the bar; reported as-is.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + measure a broadcast-domain (image-coordinate) encoder")
    ap.add_argument("--tracker", default="bytetrack_ft")
    ap.add_argument("--val-game", default=None, help="held-out game prefix (default: last)")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / f"broadcast_encoder_{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    d, r = report["dataset"], report["results"]
    print(f"wrote {out}  | held-out game={d['held_out_game']} train_win={d['n_train_windows']} "
          f"val_win={d['n_val_windows']} (chance r@1={d['chance_recall@1']})")
    print(f"floor   recon-vs-GT: {r['floor_recon_vs_gt']}")
    print(f"trained recon-vs-GT: {r['trained_broadcast_recon_vs_gt']}  (self-sanity {r['trained_self_retrieval_sanity']['recall@1']})")


if __name__ == "__main__":
    main()
