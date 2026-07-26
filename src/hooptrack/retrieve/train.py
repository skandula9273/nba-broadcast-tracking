"""CLI: increment-06b — the TRAINED compact trajectory transformer (contrastive InfoNCE).

The centerpiece's learned step: beat the hand-feature recall@k FLOOR (increment-06a), and specifically
drag court-mirror off the floor (r@1 ~0.001) by learning mirror-invariance. Mechanism: contrastive
InfoNCE over jitter/crop/mirror augmentations — a positive pair is two augmented views of one possession,
so views that differ by a court-mirror are pulled together.

Honest, one-variable comparison: the encoder trains ONLY on train-split games and is evaluated on
held-out val-split games it never saw; the hand-feature floor is recomputed on the *identical* val
gallery with the *identical* augmented queries. So "trained beats floor" isolates the feature method.

Follows run.py's argparse pattern (the retrieval sub-project's convention; the arch knobs live in the
EmbeddingConfig schema — rule #6). MPS caution (increment-04): tiny model, AMP off, run a 1-epoch probe
(`--epochs 1`) first; a non-finite loss raises loudly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from ..config import EmbeddingConfig
from ..eval.metrics import mean_recall_at_k, mean_reciprocal_rank
from .embed import PlayEmbedder, info_nce_loss
from .possessions import augment, augment_view
from .run import AUGS, KS, _rankings, features

METRIC_KEYS = [f"recall@{k}" for k in KS] + ["MRR"]


def _metrics_rk(rankings) -> dict:
    out = {f"recall@{k}": round(mean_recall_at_k(rankings, k), 4) for k in KS}
    out["MRR"] = round(mean_reciprocal_rank(rankings), 4)
    return out


def _aggregate(per_aug: dict) -> dict:
    return {m: round(float(np.mean([per_aug[a][m] for a in AUGS])), 4) for m in METRIC_KEYS}


def _bruteforce_rankings(qemb: np.ndarray, gemb: np.ndarray):
    """Per query i: (gallery ids sorted by cosine desc, relevant={i}) via q @ gallery.T."""
    return _rankings(qemb @ gemb.T)


def _faiss_rankings(qemb: np.ndarray, gemb: np.ndarray):
    """Same rankings, but through the real FAISS index (the retrieval identity)."""
    from .index import VectorIndex

    idx = VectorIndex(dim=gemb.shape[1], backend="faiss").add(np.arange(len(gemb)), gemb)
    ids, _ = idx.search(qemb, len(gemb))            # full ranking, so recall@k reads off the top
    return [(list(ids[i]), {i}) for i in range(len(qemb))]


def _eval(ranker, gemb: np.ndarray, qembs: dict) -> dict:
    per = {a: _metrics_rk(ranker(qembs[a], gemb)) for a in AUGS}
    per["overall"] = _aggregate(per)
    return per


def eval_on_val(encode_fn, val_orig: np.ndarray, aug_queries: dict, ranker=_bruteforce_rankings) -> dict:
    """Retrieval on the held-out val set: gallery = encode(originals), query = encode(augmented)."""
    gemb = encode_fn(val_orig)
    qembs = {a: encode_fn(aug_queries[a]) for a in AUGS}
    return _eval(ranker, gemb, qembs)


def verify_index(gemb: np.ndarray, qembs: dict, bruteforce: dict) -> dict:
    """Wiring check: a real FAISS index over the SAME embeddings must reproduce the brute-force recall@k
    exactly (else swapping the index in would silently change the committed numbers). Raises on mismatch."""
    # torch + faiss-cpu each link their own libomp on macOS -> OMP Error #15 on the second import.
    # setdefault (never overrides the user) is safe here: torch compute is done before faiss loads, and
    # the assertion below would catch any OMP-induced corruption rather than let it pass silently.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import faiss  # noqa: F401
    except ImportError:
        return {"backend": "faiss", "status": "skipped: faiss-cpu not installed"}
    via_index = _eval(_faiss_rankings, gemb, qembs)
    if via_index != bruteforce:
        raise RuntimeError(
            f"FAISS index does NOT reproduce brute-force retrieval — index wiring bug. "
            f"index overall={via_index['overall']} vs brute-force={bruteforce['overall']}"
        )
    return {
        "backend": "faiss-IndexFlatIP (exact; cosine on L2-normalized)",
        "faiss_version": _ver("faiss-cpu"),
        "reproduces_bruteforce": True,
        "overall": via_index["overall"],
    }


def split_by_game(meta, val_stride: int, val_offset: int):
    """Split possessions by GAME (no event overlap across splits -> no leakage). Deterministic."""
    games = sorted({m["game"] for m in meta})
    val_games = [g for i, g in enumerate(games) if i % val_stride == val_offset]
    train_games = [g for g in games if g not in val_games]
    tr = np.array([i for i, m in enumerate(meta) if m["game"] in train_games], dtype=int)
    va = np.array([i for i, m in enumerate(meta) if m["game"] in val_games], dtype=int)
    return tr, va, train_games, val_games


def train_encoder(embedder: PlayEmbedder, train_corpus: np.ndarray, args, rng, eval_hook=None):
    model = embedder.model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    n = len(train_corpus)
    history: list[float] = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        ep_losses = []
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:  # InfoNCE needs >=2 for negatives
                continue
            v1 = np.stack([augment_view(train_corpus[j], rng, args.jitter_sigma, args.p_mirror, args.p_crop)
                           for j in idx])
            v2 = np.stack([augment_view(train_corpus[j], rng, args.jitter_sigma, args.p_mirror, args.p_crop)
                           for j in idx])
            x1 = torch.as_tensor(v1, dtype=torch.float32, device=embedder.device)
            x2 = torch.as_tensor(v2, dtype=torch.float32, device=embedder.device)
            loss = info_nce_loss(model(x1), model(x2), args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        sched.step()
        mean_loss = float(np.mean(ep_losses))
        history.append(round(mean_loss, 4))
        if not math.isfinite(mean_loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {ep} — MPS precision guard (increment-04): AMP off, "
                "lower lr/batch or switch --device cpu."
            )
        if args.eval_every and (ep % args.eval_every == 0 or ep == args.epochs - 1):
            extra = eval_hook() if eval_hook else ""
            print(f"  epoch {ep:3d}  loss {mean_loss:.4f}  lr {sched.get_last_lr()[0]:.2e}{extra}", flush=True)
    return history, round(time.time() - t0, 1)


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def run(args) -> dict:
    cache = Path(args.cache_dir) / f"corpus_g{args.n_games}_T{args.T}.npz"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} not found — build it first (see Makefile: make retrieve-corpus).")
    z = np.load(cache, allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])

    # internal-consistency check (the increment-06b data-contamination guard): possessions must be ~unique
    sig = np.round(corpus.reshape(len(corpus), -1).sum(1), 3)
    unique_ratio = round(len(set(sig.tolist())) / len(corpus), 4)
    if unique_ratio < 0.9:
        raise RuntimeError(
            f"Corpus only {unique_ratio:.1%} unique — duplicated games (see _game_json mtime bug). Rebuild."
        )

    tr, va, train_games, val_games = split_by_game(meta, args.val_stride, args.val_offset)
    train_corpus, val_corpus = corpus[tr], corpus[va]
    n_val = len(val_corpus)

    # eval queries: built ONCE with a dedicated rng -> floor and trained see identical augmented queries
    rng_eval = np.random.default_rng(args.seed)
    aug_queries = {a: np.stack([augment(val_corpus[i], rng_eval, a) for i in range(n_val)]) for a in AUGS}

    floor = eval_on_val(features, val_corpus, aug_queries)

    torch.manual_seed(args.seed)
    if args.device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(args.seed)
    cfg = EmbeddingConfig(
        enabled=True, dim=args.dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ff_dim=args.ff_dim, dropout=args.dropout, temperature=args.temperature,
    )
    embedder = PlayEmbedder(cfg, device=args.device, T=args.T)
    embedder.build()

    def hook():
        p = eval_on_val(embedder.encode_batch, val_corpus, aug_queries)
        return f"  [val mirror r@1 {p['mirror']['recall@1']:.3f} | overall r@1 {p['overall']['recall@1']:.3f}]"

    rng_train = np.random.default_rng(args.seed + 1)
    history, secs = train_encoder(embedder, train_corpus, args, rng_train, eval_hook=hook)

    # Encode the val set ONCE, then score both brute-force and (as a wiring check) through the FAISS index
    # over the SAME embeddings — any difference would be the index alone.
    gemb = embedder.encode_batch(val_corpus)
    qembs = {a: embedder.encode_batch(aug_queries[a]) for a in AUGS}
    trained = _eval(_bruteforce_rankings, gemb, qembs)
    index_block = verify_index(gemb, qembs, trained)

    delta = {m: round(trained["overall"][m] - floor["overall"][m], 4) for m in METRIC_KEYS}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "06b-embedding-core-trained",
        "stage": "retrieval — trained compact trajectory transformer (contrastive InfoNCE)",
        "dataset": {
            "source": "linouk23 SportVU 2015-16", "n_games": args.n_games, "T": args.T, "entities": 11,
            "n_possessions": len(corpus), "unique_ratio": unique_ratio,
            "split": "by game (no event overlap across splits)",
            "train_games": train_games, "val_games": val_games,
            "n_train": int(len(train_corpus)), "n_val": n_val,
        },
        "eval": {
            "scheme": "augmentation-SSL: query=augmented possession, relevant=original; held-out val games",
            "augmentations": AUGS, "ks": KS,
            "comparison": "floor and trained scored on the IDENTICAL val gallery + identical augmented "
                          "queries (one variable: hand-feature vs learned encoder)",
        },
        "model": {
            "arch": "trajectory_transformer", "dim": args.dim, "d_model": args.d_model,
            "n_heads": args.n_heads, "n_layers": args.n_layers, "ff_dim": args.ff_dim,
            "dropout": args.dropout, "n_params": embedder.n_params(),
            "objective": "InfoNCE/NT-Xent", "temperature": args.temperature,
        },
        "training": {
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "weight_decay": args.weight_decay,
            "aug": {"jitter_sigma": args.jitter_sigma, "p_mirror": args.p_mirror, "p_crop": args.p_crop},
            "device": args.device, "seconds": secs,
            "loss_first": history[0], "loss_last": history[-1],
        },
        "results": {
            "floor_on_val": floor,
            "trained_on_val": trained,
            "index": index_block,
            "delta_overall": delta,
            "historical_floor_full_corpus_06a": {
                "recall@1": 0.4097, "recall@5": 0.6395, "MRR": 0.5238,
                "note": "committed 06a floor was on a latently DUPLICATED 6-game corpus (the _game_json "
                        "mtime bug fixed here); floor_on_val is the honest, clean baseline for this run",
            },
            "random_baseline": {f"recall@{k}": round(k / n_val, 5) for k in KS},
        },
        "provenance": {
            "seed": args.seed, "court_ft": [94, 50],
            "versions": {p: _ver(p) for p in ("torch", "numpy", "py7zr", "faiss-cpu")},
            "platform": platform.platform(),
        },
        "notes": "Trained encoder must beat the floor overall and specifically drag court-mirror off the "
                 "floor (learned mirror-invariance via mirror-augmented positive pairs). Reported as-is.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Trained trajectory transformer — recall@k (increment-06b)")
    ap.add_argument("--n-games", type=int, default=12)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--cache-dir", default="data/sportvu")
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--tag", default="")   # ablation label -> filename + JSON (e.g. 'nomirror')
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=13)
    # split
    ap.add_argument("--val-stride", type=int, default=3)   # every 3rd game (sorted) -> val
    ap.add_argument("--val-offset", type=int, default=2)
    # arch
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    # objective / training
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=20)  # 0 = only final
    # augmentation
    ap.add_argument("--jitter-sigma", type=float, default=0.01)
    ap.add_argument("--p-mirror", type=float, default=0.5)
    ap.add_argument("--p-crop", type=float, default=0.5)
    args = ap.parse_args()

    report = run(args)
    report["tag"] = args.tag or None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"retrieval_trained_{args.tag + '_' if args.tag else ''}{stamp}.json"
    (out / name).write_text(json.dumps(report, indent=2))

    r = report["results"]
    f, t, d = r["floor_on_val"], r["trained_on_val"], r["delta_overall"]
    print(f"\nWrote {name}  | n_train={report['dataset']['n_train']} "
          f"n_val={report['dataset']['n_val']}  params={report['model']['n_params']}")
    print(f"{'aug':<9}{'floor r@1':>11}{'trained r@1':>13}")
    for a in AUGS + ["overall"]:
        print(f"{a:<9}{f[a]['recall@1']:>11.4f}{t[a]['recall@1']:>13.4f}")
    print(f"overall: floor r@1={f['overall']['recall@1']}  trained r@1={t['overall']['recall@1']}  "
          f"(delta {d['recall@1']:+.4f})")
    idx = r["index"]
    print(f"FAISS index: {idx.get('backend')}  reproduces brute-force: "
          f"{idx.get('reproduces_bruteforce', idx.get('status'))}")


if __name__ == "__main__":
    main()
