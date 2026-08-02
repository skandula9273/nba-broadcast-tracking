"""CLI: semantic transfer probe for the play-embedding encoder.

WHY. The committed retrieval eval (run.py / train.py) uses augmentation-SSL positives: the "relevant" item
for query i is corpus item i itself, under a known augmentation family. That measures instance-level
INVARIANCE, not play CONTENT. This probe asks a different question: does the encoder put possessions of the
same *coarse play type* near each other? Labels are derived from the SportVU (T, 11, 2) tensors themselves —
no external annotation (there are no play-type labels in this repo). Thresholds live in configs/semantic_probe.yaml
(levers). Scored on the SAME held-out val split, for THREE encoders: the hand-feature floor, the trained
trajectory transformer, and random.

Metric. `precision@5` = fraction of a query's top-5 nearest neighbours that share its bucket (relevant set =
same-bucket possessions, |relevant| > 1 — NOT the query's own index). We also report the literal recall@5
(|top5 ∩ relevant| / |relevant|), which is ~5/bucket-size ≈ 0.01 and has no meaningful random baseline — hence
precision@5 is the headline (its random baseline is the bucket prevalence). Reported exactly as it comes out.

NOTE ON THE TRAINED ENCODER: it is not persisted anywhere, so — like the degradation study (study.py) — it is
reproduced in-process with the COMMITTED inc-06b recipe (trajectory_transformer, unchanged). No new/tuned model.
"""
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from ..config import EmbeddingConfig
from .embed import PlayEmbedder
from .run import features
from .train import split_by_game, train_encoder


# ---- coarse labels from the (T, 11, 2) tensor (entity 0 = ball; coord 0 = along-court, coord 1 = across) ----

def label_transition(P: np.ndarray, thr: dict) -> str:
    """Transition vs halfcourt by how far the ball advances along-court early in the possession."""
    c = thr["transition"]
    ball = P[:, 0, :]
    advance = abs(float(ball[c["advance_frames"] - 1, 0] - ball[0, 0]))
    return "transition" if advance > c["advance_thresh"] else "halfcourt"


def label_side(P: np.ndarray, thr: dict) -> str:
    """Initiation side (left / middle / right) by mean ball across-court position at entry."""
    c = thr["initiation_side"]
    y = float(P[: c["entry_frames"], 0, 1].mean())
    if y < c["left_max"]:
        return "left"
    if y > c["right_min"]:
        return "right"
    return "middle"


def label_handler(P: np.ndarray, thr: dict) -> str:
    """Low / high count of nearest-player-to-ball changes over the possession (ball-handler turnover)."""
    c = thr["handler_change"]
    ball = P[:, 0, :]
    players = P[:, 1:, :]
    d = np.linalg.norm(players - ball[:, None, :], axis=2)   # (T, 10) player-to-ball distance
    nearest = d.argmin(axis=1)                               # (T,) nearest player per frame
    changes = int((nearest[1:] != nearest[:-1]).sum())
    return "handler_low" if changes <= c["count_thresh"] else "handler_high"


SCHEMES = {
    "transition": (label_transition, ["transition", "halfcourt"]),
    "initiation_side": (label_side, ["left", "middle", "right"]),
    "handler_change": (label_handler, ["handler_low", "handler_high"]),
}


def label_corpus(corpus: np.ndarray, fn, thr: dict) -> np.ndarray:
    return np.array([fn(corpus[i], thr) for i in range(len(corpus))], dtype=object)


def merge_small(labels: np.ndarray, min_frac: float) -> tuple[np.ndarray, dict]:
    """Merge any bucket below min_frac of the corpus into the largest remaining bucket (requirement 2)."""
    labels = labels.copy()
    n = len(labels)
    merges: dict[str, str] = {}
    while len(set(labels)) > 1:
        counts = Counter(labels)
        small = min(counts, key=lambda b: counts[b])
        if counts[small] >= min_frac * n:
            break
        big = max((b for b in counts if b != small), key=lambda b: counts[b])
        labels[labels == small] = big
        merges[small] = big
    return labels, merges


def relevant_sets(labels: np.ndarray) -> list[set]:
    """For each i: the set of OTHER indices sharing its bucket. Semantic (same bucket), NOT the index {i}."""
    labels = np.asarray(labels)
    out = []
    for i in range(len(labels)):
        s = set(np.where(labels == labels[i])[0].tolist())
        s.discard(i)
        out.append(s)
    return out


def precision_recall_at_k(emb: np.ndarray, labels: np.ndarray, k: int = 5):
    """Per-query precision@k (fraction of top-k sharing the bucket) and literal recall@k (|top-k ∩ relevant| /
    |relevant|). Self is excluded from retrieval; relevant = same-bucket others."""
    labels = np.asarray(labels)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1e30)                             # exclude self
    topk = np.argsort(-sim, axis=1)[:, :k]
    same = labels[topk] == labels[:, None]                  # (N, k) is neighbour same bucket?
    prec = same.mean(axis=1)
    rel_size = np.array([int((labels == labels[i]).sum()) - 1 for i in range(len(labels))])
    recall = np.where(rel_size > 0, same.sum(axis=1) / np.maximum(rel_size, 1), 0.0)
    return prec, recall


def _committed_encoder_args(device: str, seed: int, epochs: int) -> SimpleNamespace:
    """The COMMITTED inc-06b training recipe (unchanged) — reproduces the trained trajectory transformer."""
    return SimpleNamespace(
        arch="trajectory_transformer", dim=128, d_model=128, n_heads=4, n_layers=2, ff_dim=256, dropout=0.1,
        temperature=0.1, epochs=epochs, batch=512, lr=1e-3, weight_decay=1e-4, eval_every=0,
        jitter_sigma=0.01, p_mirror=0.5, p_crop=0.5, p_permute=0.0, n_permute=10, p_swap=0.0, n_swaps=2,
        T=48, device=device, seed=seed,
    )


def _trained_val_embeddings(train_corpus, val_corpus, a) -> np.ndarray:
    torch.manual_seed(a.seed)
    if a.device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(a.seed)
    cfg = EmbeddingConfig(
        enabled=True, arch=a.arch, dim=a.dim, d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers,
        ff_dim=a.ff_dim, dropout=a.dropout, temperature=a.temperature,
    )
    emb = PlayEmbedder(cfg, device=a.device, T=a.T)
    emb.build()
    train_encoder(emb, train_corpus, a, np.random.default_rng(a.seed + 1))
    return emb.encode_batch(val_corpus)


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _stats(prec, recall) -> dict:
    return {"precision@5": round(float(prec.mean()), 4), "recall@5_literal": round(float(recall.mean()), 4)}


def run(args) -> dict:
    thr = yaml.safe_load(Path(args.config).read_text())
    z = np.load(Path(args.corpus), allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    tr, va, train_games, val_games = split_by_game(meta, args.val_stride, args.val_offset)
    val_corpus = corpus[va]
    n_val = len(val_corpus)

    # THREE encoders' val embeddings (all L2-normalized -> cosine); random is seeded.
    rng = np.random.default_rng(args.seed)
    rand = rng.standard_normal((n_val, 128)).astype(np.float32)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    encoders = {
        "floor": features(val_corpus),
        "trained": _trained_val_embeddings(corpus[tr], val_corpus, _committed_encoder_args(args.device, args.seed, args.epochs)),
        "random": rand,
    }

    schemes_out = {}
    for name, (fn, order) in SCHEMES.items():
        corpus_labels = label_corpus(corpus, fn, thr)                 # label the WHOLE corpus
        raw_dist = {b: int(c) for b, c in Counter(corpus_labels).items()}
        merged_labels, merges = merge_small(corpus_labels, thr["min_bucket_frac"])
        merged_dist = {b: int(c) for b, c in Counter(merged_labels).items()}
        val_labels = merged_labels[va]
        buckets = sorted(set(val_labels.tolist()))
        degenerate = len(buckets) < 2

        scores, per_bucket = {}, {}
        if not degenerate:
            for enc, emb in encoders.items():
                prec, rec = precision_recall_at_k(emb, val_labels, k=5)
                scores[enc] = _stats(prec, rec)
                per_bucket.setdefault("precision@5", {})
                for bkt in buckets:
                    m = val_labels == bkt
                    per_bucket["precision@5"].setdefault(bkt, {})[enc] = round(float(prec[m].mean()), 4)

        schemes_out[name] = {
            "definition": fn.__doc__.strip(),
            "thresholds": thr[name],
            "corpus_distribution_raw": raw_dist,
            "merged_into": merges or None,
            "corpus_distribution_used": merged_dist,
            "val_bucket_counts": {b: int((val_labels == b).sum()) for b in buckets},
            "degenerate_single_bucket": degenerate,
            "overall": scores,
            "per_bucket_precision@5": per_bucket.get("precision@5", {}),
            "random_baseline_note": "random precision@5 ~= bucket prevalence; a useful encoder beats it",
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "retrieval-semantic-probe",
        "stage": "semantic transfer probe — does the augmentation-SSL encoder capture play content?",
        "purpose": "The committed retrieval eval's positive is the query's OWN index (instance-level "
        "invariance). This scores retrieval where relevant = same COARSE-PLAY-TYPE bucket (derived from the "
        "trajectory, |relevant|>1), i.e. whether the encoder captures content, not just augmentation-invariance.",
        "metric": "precision@5 = fraction of a query's top-5 nearest neighbours sharing its bucket (self "
        "excluded). recall@5_literal = |top5 ∩ relevant|/|relevant| (~5/bucket-size; shown for completeness, "
        "not the headline — precision@5's random baseline is the bucket prevalence).",
        "dataset": {
            "source": "linouk23 SportVU 2015-16", "corpus_size": int(len(corpus)),
            "split": "by game (held-out val)", "n_train": int(len(tr)), "n_val": n_val,
            "val_games": val_games,
        },
        "encoders": {
            "floor": "hand-feature: flattened court-normalized trajectory + cosine (run.py features)",
            "trained": "committed inc-06b trajectory transformer, reproduced in-process (no persisted "
            "checkpoint; standard recipe, not tuned)",
            "random": "seeded random unit vectors (baseline ~= bucket prevalence)",
        },
        "thresholds_config": thr,
        "schemes": schemes_out,
        "provenance": {
            "seed": args.seed, "device": args.device, "encoder_epochs": args.epochs,
            "versions": {p: _ver(p) for p in ("torch", "numpy")}, "platform": platform.platform(),
        },
        "notes": "Coarse labels derived from the tensors themselves — a weak content proxy, not designed "
        "set-plays. NOTE: initiation_side (left/right) is a court-mirror-sensitive dimension the trained "
        "encoder is explicitly INVARIANT to (inc-06b), so it is expected to underperform the floor there — a "
        "real consequence of the invariance, reported per-bucket. Thresholds picked on definition, run once, "
        "reported as-is.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic transfer probe for the play-embedding encoder")
    ap.add_argument("--config", default="configs/semantic_probe.yaml")
    ap.add_argument("--corpus", default="data/sportvu/corpus_g12_T48.npz")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--val-offset", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)   # committed inc-06b encoder recipe
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"semantic_probe_{stamp}.json").write_text(json.dumps(report, indent=2))

    print(f"\nWrote semantic_probe_{stamp}.json  | corpus={report['dataset']['corpus_size']} "
          f"n_val={report['dataset']['n_val']}")
    for name, s in report["schemes"].items():
        print(f"\n[{name}]  corpus dist: {s['corpus_distribution_used']}"
              + (f"  (merged {s['merged_into']})" if s["merged_into"] else ""))
        if s["degenerate_single_bucket"]:
            print("  DEGENERATE — single bucket after merge; no precision computed.")
            continue
        o = s["overall"]
        print(f"  precision@5:  floor={o['floor']['precision@5']}  trained={o['trained']['precision@5']}  "
              f"random={o['random']['precision@5']}")


if __name__ == "__main__":
    main()
