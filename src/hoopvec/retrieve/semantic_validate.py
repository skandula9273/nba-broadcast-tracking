"""Validate that semantic play retrieval is ACHIEVABLE — the positive complement to `semantic_probe`.

The probe showed the augmentation-SSL encoder sits at ~random on play-type buckets (it learns aug-invariance,
not content). This asks the follow-up: can ANY encoder do semantic retrieval on these tensors? Train the SAME
architecture with a SUPERVISED contrastive objective (SupCon) on a real play-type axis — transition vs
halfcourt, i.e. fast break vs set offense — and measure held-out-GAME semantic precision@5 against the floor,
the SSL encoder, and random. If supervised >> the rest on UNSEEN games, semantic retrieval is achievable and
the eval is sensitive: the SSL *objective* was the limitation, not the task or the metric.

Honest caveat: the bucket is a *derived* proxy (ball advance early in the possession), not an annotated
set-play. So a supervised win shows the encoder can be steered to a semantic axis and GENERALIZE it to held-out
games — not that it discovers real plays. Trained by-game split (train games' labels only), scored on val games.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import EmbeddingConfig
from .embed import PlayEmbedder
from .possessions import augment_view
from .run import features
from .semantic_probe import (
    SCHEMES,
    _committed_encoder_args,
    _trained_val_embeddings,
    _ver,
    label_corpus,
    merge_small,
    precision_recall_at_k,
)
from .train import split_by_game


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020), out-of-N form. `z` is L2-normalized (B, d); positives
    for row i are the other rows sharing its label. Pulls same-play-type together, pushes others apart."""
    B = z.shape[0]
    sim = z @ z.T / temperature
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)                       # exclude self from the denominator
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)       # log-softmax over non-self
    pos_count = pos.sum(1)
    per_row = -(logp * pos).sum(1) / pos_count.clamp(min=1)       # mean log-prob over a row's positives
    valid = pos_count > 0                                         # a row needs >=1 positive in the batch
    return per_row[valid].mean()


def train_supervised(embedder: PlayEmbedder, train_corpus: np.ndarray, labels_int: np.ndarray, args,
                     rng: np.random.Generator) -> float:
    """SupCon-train the encoder on `labels_int` (per train possession). Same aug views as the SSL recipe, so
    the only changed variable vs SSL is the OBJECTIVE (SupCon on play-type, not InfoNCE on aug-identity)."""
    model = embedder.model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    n = len(train_corpus)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:
                continue
            v = np.stack([augment_view(train_corpus[j], rng, args.jitter_sigma, args.p_mirror, args.p_crop)
                          for j in idx])
            x = torch.as_tensor(v, dtype=torch.float32, device=embedder.device)
            y = torch.as_tensor(labels_int[idx], dtype=torch.long, device=embedder.device)
            loss = supcon_loss(model(x), y, args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not math.isfinite(float(loss)):
                raise RuntimeError(f"Non-finite SupCon loss at epoch {ep} (MPS guard: AMP off, lower lr/batch).")
        sched.step()
    return round(time.time() - t0, 1)


def _supervised_val_embeddings(train_corpus, val_corpus, train_labels_int, a) -> np.ndarray:
    torch.manual_seed(a.seed)
    if a.device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(a.seed)
    cfg = EmbeddingConfig(enabled=True, arch=a.arch, dim=a.dim, d_model=a.d_model, n_heads=a.n_heads,
                          n_layers=a.n_layers, ff_dim=a.ff_dim, dropout=a.dropout, temperature=a.temperature)
    emb = PlayEmbedder(cfg, device=a.device, T=a.T)
    emb.build()
    train_supervised(emb, train_corpus, train_labels_int, a, np.random.default_rng(a.seed + 2))
    return emb.encode_batch(val_corpus)


def run(args) -> dict:
    thr = yaml.safe_load(Path(args.config).read_text())
    z = np.load(Path(args.corpus), allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    tr, va, train_games, val_games = split_by_game(meta, args.val_stride, args.val_offset)

    fn, _order = SCHEMES[args.scheme]
    labels_all, merges = merge_small(label_corpus(corpus, fn, thr), thr["min_bucket_frac"])
    buckets = sorted(set(labels_all.tolist()))
    to_int = {b: i for i, b in enumerate(buckets)}
    labels_int = np.array([to_int[b] for b in labels_all])
    val_labels = labels_all[va]

    a = _committed_encoder_args(args.device, args.seed, args.epochs)
    encoders = {
        "floor": features(corpus[va]),
        "ssl": _trained_val_embeddings(corpus[tr], corpus[va], a),           # inc-06b InfoNCE (aug-invariance)
        "supervised": _supervised_val_embeddings(corpus[tr], corpus[va], labels_int[tr], a),  # SupCon on play-type
    }
    rng = np.random.default_rng(args.seed)
    rand = rng.standard_normal((len(va), 128)).astype(np.float32)
    encoders["random"] = rand / np.linalg.norm(rand, axis=1, keepdims=True)

    scores = {}
    for enc, emb in encoders.items():
        prec, _ = precision_recall_at_k(emb, val_labels, k=5)
        scores[enc] = round(float(prec.mean()), 4)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "semantic-retrieval-validation",
        "stage": "can an encoder trained FOR semantics do semantic play retrieval? (positive complement to the probe)",
        "scheme": args.scheme, "scheme_definition": fn.__doc__.strip(),
        "dataset": {"source": "linouk23 SportVU 2015-16", "split": "by game (held-out val)",
                    "n_train": int(len(tr)), "n_val": int(len(va)), "val_games": val_games,
                    "val_bucket_counts": {b: int((val_labels == b).sum()) for b in buckets}},
        "metric": "semantic precision@5 (fraction of top-5 neighbours sharing the play-type bucket; self "
        "excluded). Random baseline ~= bucket prevalence.",
        "results_precision@5": scores,
        "encoders": {
            "floor": "flattened normalized trajectory (has raw ball position -> some transition signal)",
            "ssl": "inc-06b InfoNCE on augmentation-identity (learns aug-invariance, not content)",
            "supervised": "SAME architecture, SupCon on the play-type axis (only the objective changed)",
            "random": "seeded unit vectors (~= bucket prevalence)",
        },
        "provenance": {"seed": args.seed, "device": args.device, "epochs": args.epochs,
                       "versions": {p: _ver(p) for p in ("torch", "numpy")}, "platform": platform.platform()},
        "notes": "Derived bucket, not an annotated set-play: a supervised win shows the encoder can be STEERED "
        "to a semantic axis and generalize it to HELD-OUT GAMES (train/val split by game), i.e. semantic "
        "retrieval is achievable and the metric is sensitive -- the SSL objective was the limitation, not the "
        "task. Only variable changed vs SSL is the objective (same arch, aug, split, seed).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate semantic retrieval is achievable (supervised encoder)")
    ap.add_argument("--config", default="configs/semantic_probe.yaml")
    ap.add_argument("--corpus", default="data/sportvu/corpus_g12_T48.npz")
    ap.add_argument("--scheme", default="transition", choices=list(SCHEMES))
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--val-offset", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / f"semantic_validate_{args.scheme}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}  | scheme={args.scheme} n_val={report['dataset']['n_val']}")
    print(f"semantic precision@5: {report['results_precision@5']}")


if __name__ == "__main__":
    main()
