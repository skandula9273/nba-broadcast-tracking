"""THE CENTERPIECE — the trained play/possession embedding model. (increment-06b)

Turns a possession (all players + ball trajectories over a window) into a vector, so similar plays land
near each other. This is where the real ML depth + the retrieval identity live — depth goes HERE, not in
the perception pipeline.

Design (config: `embedding.*`):
  - arch:      trajectory_transformer — a compact temporal transformer over the possession's tracks.
               Each of T timesteps is ONE token: the flattened (ball + 10 players) x (x, y) = 22-dim
               court-normalized positions. Tokens -> d_model, + sinusoidal positional encoding over time,
               pre-LN TransformerEncoder, temporal mean-pool, projection head -> L2-normalized embedding.
  - objective: contrastive InfoNCE / NT-Xent over jitter/crop/mirror augmentations (see train.py). A
               positive pair is two augmented views of one possession; every other view is a negative.
  - output:    a fixed-size L2-normalized embedding (embedding.dim), cosine-ready for FAISS.

Small BY DESIGN (increment-04 MPS caution: tiny model, AMP off, 1-epoch probe). Prior art: play2vec,
HoopTransformer (SSL on NBA trajectories), TrajSV. Eval: retrieval recall@k / MRR (eval/metrics.py),
measured against the hand-feature floor on an identical held-out val set.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..config import EmbeddingConfig

N_ENTITIES = 11   # ball + 10 players
COORDS = 2        # (x, y), court-normalized to [0, 1]


def _sinusoidal_pe(T: int, d: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding (no params), shape (T, d)."""
    pe = torch.zeros(T, d)
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TrajectoryTransformer(nn.Module):
    """Compact temporal transformer: possession (B, T, 11, 2) -> L2-normalized embedding (B, dim)."""

    def __init__(
        self, dim: int = 128, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
        ff_dim: int = 256, dropout: float = 0.1, T: int = 48,
    ) -> None:
        super().__init__()
        self.T = T
        self.in_proj = nn.Linear(N_ENTITIES * COORDS, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff_dim, dropout, activation="gelu",
            batch_first=True, norm_first=True,   # pre-LN: stabler on small batches / MPS (AMP off, inc-04)
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, dim))
        self.register_buffer("pe", _sinusoidal_pe(T, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 11, 2). Center coords to [-0.5, 0.5] so a court-mirror is a coordinate sign flip.
        B, T = x.shape[0], x.shape[1]
        h = x.reshape(B, T, -1) - 0.5                      # (B, T, 22)
        h = self.in_proj(h) + self.pe[:T].to(h.dtype)      # (B, T, d_model)
        h = self.encoder(h)
        h = self.norm(h).mean(dim=1)                       # temporal mean-pool -> (B, d_model)
        z = self.head(h)                                   # (B, dim)
        return nn.functional.normalize(z, dim=1)           # cosine-ready


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent (SimCLR). z1, z2: (B, dim) L2-normalized views of the SAME B possessions.

    For each of the 2B views, its one positive is the matching view of the same possession; all other
    2B-2 views are negatives. Self-similarity is masked out. Cross-entropy over the similarity rows.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                         # (2B, dim)
    sim = (z @ z.T) / temperature                          # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))                      # a view is never its own positive
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return nn.functional.cross_entropy(sim, targets)


class PlayEmbedder:
    """Wraps the encoder: build the model, and encode possessions -> vectors. Training loop is in train.py
    (orchestration — augmentation, InfoNCE, eval — mirrors run.py owning the floor)."""

    def __init__(self, cfg: EmbeddingConfig, device: str = "mps", T: int = 48) -> None:
        self.cfg = cfg
        self.device = device
        self.T = T
        self.model: TrajectoryTransformer | None = None

    def build(self) -> TrajectoryTransformer:
        self.model = TrajectoryTransformer(
            dim=self.cfg.dim, d_model=self.cfg.d_model, n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers, ff_dim=self.cfg.ff_dim, dropout=self.cfg.dropout, T=self.T,
        ).to(self.device)
        return self.model

    def n_params(self) -> int:
        assert self.model is not None
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode_batch(self, arr: np.ndarray, batch: int = 512) -> np.ndarray:
        """(N, T, 11, 2) numpy possessions -> (N, dim) L2-normalized numpy embeddings."""
        assert self.model is not None
        self.model.eval()
        out = []
        for i in range(0, len(arr), batch):
            x = torch.as_tensor(arr[i:i + batch], dtype=torch.float32, device=self.device)
            out.append(self.model(x).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.cfg.dim), np.float32)

    def encode(self, possession) -> "list[float]":
        return self.encode_batch(np.asarray(possession)[None])[0].tolist()
