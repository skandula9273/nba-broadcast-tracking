"""CLI: increment-07 — the reconstructed-vs-GT degradation study (the platform's headline finding).

Question: how much retrieval accuracy does broadcast reconstruction cost vs ground-truth tracks? With no
aligned broadcast+SportVU truth available, we run a CONTROLLED degradation — inject the per-stage error
budgets this project already measured (see degrade.py) into clean GT possessions and measure the recall@k
falloff. Setup reuses the whole harness: query = a degraded GT possession, gallery = the clean GT val
corpus, relevant = the source possession, retrieval = the FAISS index. Baseline (no degradation) = 1.0, so
every point below 1.0 is the recall the reconstruction costs. We report the TRAINED encoder and the
hand-feature FLOOR side by side (does learning degrade more gracefully?), one error source at a time, plus a
combined realistic operating point — and, separately and clearly labelled, a re-ID sensitivity (that stage
is not yet built/measured).
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..config import EmbeddingConfig
from .degrade import dropout_interp, id_swap, jitter_ft, permute_players, reconstruct
from .embed import PlayEmbedder
from .train import (
    _faiss_rankings,
    _metrics_rk,
    _ver,
    features,
    split_by_game,
    train_encoder,
)

# error-source sweeps: (name, perturb_fn, severity grid). Each grid[0] is the no-degradation anchor.
SWEEPS = [
    ("jitter_ft", jitter_ft, [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]),   # homography + detection localization
    ("dropout", dropout_interp, [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]),  # detection recall / fragmentation
    ("id_swap", id_swap, [0, 1, 2, 3, 4]),                        # tracking association (the sharp one)
]
REID_SWEEP = ("permute_players", permute_players, [0, 2, 4, 6, 10])  # sensitivity only — re-ID not measured
ENCODERS = ("trained", "floor")


def _prepare(args):
    """Load the corpus, split by game, train the encoder on the train split (reuses train.py)."""
    cache = Path(args.cache_dir) / f"corpus_g{args.n_games}_T{args.T}.npz"
    z = np.load(cache, allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    tr, va, train_games, val_games = split_by_game(meta, args.val_stride, args.val_offset)
    torch.manual_seed(args.seed)
    if args.device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(args.seed)
    cfg = EmbeddingConfig(
        enabled=True, dim=args.dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ff_dim=args.ff_dim, dropout=args.dropout, temperature=args.temperature,
    )
    emb = PlayEmbedder(cfg, device=args.device, T=args.T)
    emb.build()
    _, secs = train_encoder(emb, corpus[tr], args, np.random.default_rng(args.seed + 1))
    return emb, corpus[va], train_games, val_games, secs


def _point(perturb, sev, val_corpus, encoders, galleries, seed_seq) -> dict:
    """Degrade every val possession identically, then retrieve with each encoder over its clean gallery."""
    rng = np.random.default_rng(seed_seq)
    q = np.stack([perturb(val_corpus[i], rng, sev) for i in range(len(val_corpus))])
    return {name: _metrics_rk(_faiss_rankings(fn(q), galleries[name])) for name, fn in encoders.items()}


def run(args) -> dict:
    emb, val_corpus, train_games, val_games, secs = _prepare(args)
    n_val = len(val_corpus)
    encoders = {"trained": emb.encode_batch, "floor": features}
    galleries = {name: fn(val_corpus) for name, fn in encoders.items()}

    def ident(P, rng, _):
        return P

    baseline = _point(ident, None, val_corpus, encoders, galleries, [args.seed, 0])

    sweeps = {}
    for si, (name, fn, grid) in enumerate(SWEEPS, start=1):
        sweeps[name] = [
            {"severity": sev, **_point(fn, sev, val_corpus, encoders, galleries, [args.seed, si, gi])}
            for gi, sev in enumerate(grid)
        ]

    def recon(P, rng, _):
        return reconstruct(P, rng, args.re_sigma_ft, args.re_drop, args.re_swaps)

    combined = _point(recon, None, val_corpus, encoders, galleries, [args.seed, 90])

    rname, rfn, rgrid = REID_SWEEP
    reid = [
        {"n_wrong": sev, **_point(rfn, sev, val_corpus, encoders, galleries, [args.seed, 91, gi])}
        for gi, sev in enumerate(rgrid)
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "07-degradation-study",
        "stage": "reconstructed-vs-GT retrieval degradation (controlled)",
        "dataset": {
            "source": "linouk23 SportVU 2015-16", "n_games": args.n_games, "T": args.T,
            "split": "by game (held-out val)", "train_games": train_games, "val_games": val_games,
            "n_val": n_val,
        },
        "method": {
            "query": "a DEGRADED clean GT possession", "gallery": "the clean GT val corpus",
            "relevant": "the source possession", "retrieval": "FAISS IndexFlatIP (cosine)",
            "encoders": ["trained trajectory transformer", "hand-feature floor"],
            "baseline_no_degradation_is_1.0": "query == gallery item, so recall@1 = 1.0; drops are the cost",
        },
        "measured_budgets": {
            "homography_reproj_px": {"kp_sigma_1px": 0.40, "kp_sigma_3px": 2.20, "kp_sigma_5px": 4.69,
                                     "unregistered_baseline_px": 877.0, "source": "inc-05"},
            "detection": {"DetA": 0.707, "LocA": 0.841, "Frag": 1847, "source": "inc-04 (fine-tuned)"},
            "tracking": {"AssA": 0.317, "IDSW": 955, "frames": 12557, "source": "inc-04 (fine-tuned)"},
            "mapping_notes": "jitter_ft <- homography(~2-5px @ broadcast scale ~0.15-0.5ft) + detection "
            "localization; dropout <- DetA/Frag; id_swap count <- IDSW/frames ~0.076/frame ~2-4 per 48-frame "
            "possession (AssA 0.317 => association is the weak stage). Order-of-magnitude anchoring; the "
            "sweep carries the finding, not any single derived number.",
        },
        "realistic_budget": {"sigma_ft": args.re_sigma_ft, "drop_rate": args.re_drop, "n_swaps": args.re_swaps},
        "results": {
            "baseline_no_degradation": baseline,
            "sweeps": sweeps,
            "combined_realistic": combined,
            "sensitivity_reid_pending": {
                "permute_players": reid,
                "note": "re-ID / player-identity stage is NOT built or measured — a real tracker emits "
                "arbitrary track order and the encoder needs canonical order. Shown as sensitivity only, no "
                "operating point; expected to dominate, which motivates building+measuring re-ID next.",
            },
        },
        "provenance": {
            "seed": args.seed, "device": args.device, "train_seconds": secs,
            "versions": {p: _ver(p) for p in ("torch", "numpy", "faiss-cpu")},
            "platform": platform.platform(),
        },
        "notes": "Controlled degradation: no aligned broadcast+SportVU truth exists, so GT tracks are "
        "perturbed with measured per-stage error budgets. Reported as-is.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstructed-vs-GT degradation study (increment-07)")
    ap.add_argument("--n-games", type=int, default=12)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--cache-dir", default="data/sportvu")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--val-offset", type=int, default=2)
    # realistic combined operating point (anchored to the measured budgets)
    ap.add_argument("--re-sigma-ft", type=float, default=0.5)
    ap.add_argument("--re-drop", type=float, default=0.1)
    ap.add_argument("--re-swaps", type=int, default=2)
    # encoder arch + training (defaults match the committed inc-06b model)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--jitter-sigma", type=float, default=0.01)
    ap.add_argument("--p-mirror", type=float, default=0.5)
    ap.add_argument("--p-crop", type=float, default=0.5)
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"degradation_{stamp}.json").write_text(json.dumps(report, indent=2))

    r = report["results"]
    print(f"\nWrote degradation_{stamp}.json | n_val={report['dataset']['n_val']}")
    print(f"baseline (no degradation): trained r@1={r['baseline_no_degradation']['trained']['recall@1']} "
          f"floor r@1={r['baseline_no_degradation']['floor']['recall@1']}")
    for name, pts in r["sweeps"].items():
        print(f"\n{name}  (severity: trained r@1 | floor r@1)")
        for p in pts:
            print(f"  {p['severity']:>5}: {p['trained']['recall@1']:.3f} | {p['floor']['recall@1']:.3f}")
    c = r["combined_realistic"]
    print(f"\ncombined realistic {report['realistic_budget']}: "
          f"trained r@1={c['trained']['recall@1']} floor r@1={c['floor']['recall@1']}")
    print("\nreID sensitivity (pending measurement) — permute_players (n_wrong: trained r@1 | floor r@1):")
    for p in r["sensitivity_reid_pending"]["permute_players"]:
        print(f"  {p['n_wrong']:>2}: {p['trained']['recall@1']:.3f} | {p['floor']['recall@1']:.3f}")


if __name__ == "__main__":
    main()
