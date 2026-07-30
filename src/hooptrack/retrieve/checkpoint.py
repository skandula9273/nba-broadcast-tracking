"""Checkpoint save/load for the play encoder — so retrieval and degradation describe ONE model instance.

Before this, `train.py` and `study.py` each trained from scratch and discarded the model, so the committed
retrieval numbers and the committed degradation numbers came from DIFFERENT random inits of the same recipe.
A checkpoint stores the trained weights PLUS the config that built them, so:
  - the encoder reloads correctly even after `configs/*.yaml` drifts (we rebuild from the checkpoint's OWN
    config, never the current file), and
  - any eval that consumed a checkpoint is traceable to the exact weights (path + corpus fingerprint + git sha).

Weights live under `weights/retrieve/` (gitignored, like the detector weights); provenance travels in the
committed eval JSON.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..config import EmbeddingConfig
from .embed import PlayEmbedder


def corpus_fingerprint(corpus: np.ndarray) -> str:
    """Stable content hash of the training corpus — identifies the exact data the checkpoint was trained on."""
    return hashlib.sha1(np.ascontiguousarray(corpus, dtype=np.float32).tobytes()).hexdigest()[:16]


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def save_checkpoint(embedder: PlayEmbedder, path: str | Path, *, seed: int, corpus_fp: str,
                    final_loss: float | None, val_metrics: dict | None, extra: dict | None = None) -> str:
    """Save {state_dict, config, arch, T, seed, corpus fingerprint, git sha, final loss, val metrics}."""
    assert embedder.model is not None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in embedder.model.state_dict().items()},
        "config": embedder.cfg.model_dump(),        # the config that BUILT this model (reload uses this)
        "arch": embedder.cfg.arch, "T": embedder.T, "seed": seed,
        "corpus_fingerprint": corpus_fp, "git_sha": git_sha(),
        "final_loss": final_loss, "val_metrics": val_metrics,
        **(extra or {}),
    }, path)
    return str(path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[PlayEmbedder, dict]:
    """Reconstruct the encoder from the checkpoint's OWN config (not the current config file) + load weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = EmbeddingConfig(**ckpt["config"])          # config drift can't break an old checkpoint
    emb = PlayEmbedder(cfg, device=device, T=ckpt["T"])
    emb.build()
    emb.model.load_state_dict(ckpt["state_dict"])
    emb.model.eval()
    return emb, ckpt
