"""THE CENTERPIECE — the trained play/possession embedding model.

Turns a possession (all players + ball trajectories over a window) into a vector, so similar plays land
near each other. This is where the real ML depth + your retrieval identity live. Depth goes HERE, not in
the perception pipeline.

Design (config: `embedding.*`):
  - arch:      trajectory_transformer (a temporal transformer over the possession's tracks) | baller2vec
  - objective: contrastive (augment a possession -> positive; other possessions -> negatives)
             | denoising (play2vec-style) | masked (masked-trajectory prediction)
  - output:    a fixed-size embedding (embedding.dim)

Prior art to build on: play2vec, SimPlay, HoopTransformer (SSL on NBA trajectories), TrajSV, UniTE.
Eval: retrieval recall@k / MRR against a held-out similar-play set (see eval/metrics.py).

This is a PyTorch training job (GPU). Keep the training loop, loss, and ablations first-class + W&B-logged.
"""
from __future__ import annotations

from ..config import EmbeddingConfig


class PlayEmbedder:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self._model = None  # a torch.nn.Module built lazily

    def build(self):
        raise NotImplementedError(
            f"PlayEmbedder.build: construct the {self.cfg.arch} encoder (dim={self.cfg.dim}). "
            "A temporal transformer over per-frame player+ball features, or the baller2vec encoder."
        )

    def train(self, possessions) -> None:
        raise NotImplementedError(
            f"PlayEmbedder.train: self-supervised ({self.cfg.objective}) training loop. "
            "Contrastive: augment (temporal crop, court mirror, jitter) -> positives; other possessions -> "
            "negatives; InfoNCE loss. Log to W&B; ablate arch/objective one variable at a time."
        )

    def encode(self, possession) -> "list[float]":
        raise NotImplementedError("PlayEmbedder.encode: possession -> embedding vector.")
