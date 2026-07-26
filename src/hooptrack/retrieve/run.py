"""CLI: the retrieval recall@k FLOOR — hand-feature baseline, augmentation-SSL eval. (increment-06)

Simpler-first (the contract): before the trained encoder, a **hand-feature** retrieval baseline sets the
recall@k floor the encoder must beat. Feature = the flattened, court-normalized possession trajectory,
L2-normalized; retrieval = cosine nearest-neighbour. Eval = the augmentation-SSL scheme: query with an
augmented possession (jitter / temporal-crop / court-mirror), the 'relevant' gallery item is the original.

Reuses the tested `eval/metrics.py` recall@k / MRR. Committed to eval_results/. The trained trajectory
transformer (next step) must beat these numbers — especially on mirror/crop, where a raw-trajectory
feature is weak by construction.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ..eval.metrics import mean_recall_at_k, mean_reciprocal_rank
from .possessions import augment, build_corpus

AUGS = ["jitter", "crop", "mirror"]
KS = [1, 5, 10]


def features(batch: np.ndarray) -> np.ndarray:
    """Hand feature: flattened normalized trajectory, L2-normalized rows (cosine-ready)."""
    x = batch.reshape(len(batch), -1)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _rankings(sim: np.ndarray):
    """Per query i: (gallery indices sorted by similarity desc, relevant={i})."""
    order = np.argsort(-sim, axis=1)
    return [(list(order[i]), {i}) for i in range(len(sim))]


def _load_or_build(cfg) -> tuple[np.ndarray, list[dict]]:
    cache = Path(cfg["cache_dir"]) / f"corpus_g{cfg['n_games']}_T{cfg['T']}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["corpus"], list(z["meta"])
    corpus, meta = build_corpus(cfg["n_games"], cfg["T"], cfg["max_possessions"], cache_dir=cfg["cache_dir"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, corpus=corpus, meta=np.array(meta, dtype=object))
    return corpus, meta


def run(n_games: int, T: int, max_possessions: int, cache_dir: str, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    corpus, meta = _load_or_build(
        {"n_games": n_games, "T": T, "max_possessions": max_possessions, "cache_dir": cache_dir}
    )
    n = len(corpus)
    gallery = features(corpus)

    def eval_aug(kind: str) -> dict:
        q = features(np.stack([augment(corpus[i], rng, kind) for i in range(n)]))
        sim = q @ gallery.T
        rk = _rankings(sim)
        out = {f"recall@{k}": round(mean_recall_at_k(rk, k), 4) for k in KS}
        out["MRR"] = round(mean_reciprocal_rank(rk), 4)
        return out

    per_aug = {a: eval_aug(a) for a in AUGS}
    overall = {
        f"recall@{k}": round(float(np.mean([per_aug[a][f"recall@{k}"] for a in AUGS])), 4) for k in KS
    }
    overall["MRR"] = round(float(np.mean([per_aug[a]["MRR"] for a in AUGS])), 4)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "06-embedding-core",
        "stage": "retrieval — hand-feature baseline (recall@k floor)",
        "dataset": {"source": "linouk23 SportVU 2015-16", "n_games": n_games,
                    "n_possessions": n, "T": T, "entities": 11},
        "eval": {"scheme": "augmentation-SSL: query=augmented possession, relevant=original",
                 "augmentations": AUGS, "ks": KS},
        "baseline": "flattened court-normalized trajectory + cosine NN (hand feature, no learning)",
        "results": {
            "per_augmentation": per_aug,
            "overall": overall,
            "random_baseline": {f"recall@{k}": round(k / n, 5) for k in KS},
        },
        "provenance": {"seed": seed, "court_ft": [94, 50],
                       "versions": {p: (_ver(p)) for p in ("numpy", "py7zr")},
                       "platform": platform.platform()},
        "notes": "Floor to beat. Raw-trajectory features are strong on jitter, weaker on temporal crop, "
        "and weak on court-mirror by construction — the trained trajectory transformer must add that "
        "invariance. Random baseline shown for scale.",
    }


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval recall@k floor (hand-feature baseline)")
    ap.add_argument("--n-games", type=int, default=6)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--max-possessions", type=int, default=1500)
    ap.add_argument("--cache-dir", default="data/sportvu")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()
    report = run(args.n_games, args.T, args.max_possessions, args.cache_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"retrieval_floor_{stamp}.json").write_text(json.dumps(report, indent=2))
    r = report["results"]
    print(f"Wrote retrieval_floor_{stamp}.json  | n_possessions={report['dataset']['n_possessions']}")
    print(f"  overall recall@1={r['overall']['recall@1']} recall@5={r['overall']['recall@5']} "
          f"MRR={r['overall']['MRR']}  (random recall@1={r['random_baseline']['recall@1']})")
    print("  per-aug: " + " | ".join(f"{a}: r@1={r['per_augmentation'][a]['recall@1']}" for a in AUGS))


if __name__ == "__main__":
    main()
