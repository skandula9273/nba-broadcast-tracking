"""Vector index for play/possession embeddings — FAISS backend. (increment-06b)

Thin, swappable wrapper (config: `retrieval.index`) around a FAISS index so the retrieval core can turn
the trained encoder's embeddings into fast nearest-neighbour search.

Design choice — **exact flat inner-product** (`IndexFlatIP`, simpler-first, rule #7): the encoder emits
**L2-normalized** vectors, so inner product == cosine, and an exact flat index reproduces the brute-force
`q @ gallery.T` retrieval the recall@k numbers were computed with — **bit-for-bit** (see the wiring check in
train.py / tests). That equivalence is the point: swapping in the index must not change the measured recall.
An approximate index (IVF/HNSW: recall-vs-latency) is a later *measured* ablation, not a prerequisite.
`IndexIDMap` lets us attach arbitrary possession ids and get them back from search.
"""
from __future__ import annotations

import os

import numpy as np


class VectorIndex:
    def __init__(self, dim: int, backend: str = "faiss", metric: str = "ip") -> None:
        self.dim = dim
        self.backend = backend
        self.metric = metric            # "ip" = inner product; on L2-normalized vectors this IS cosine
        self._index = None              # built lazily so importing this module needs no faiss

    def _build(self) -> None:
        if self.backend != "faiss":
            raise NotImplementedError(f"backend {self.backend!r} — only 'faiss' is implemented (pgvector TODO).")
        if self.metric != "ip":
            raise ValueError(f"metric {self.metric!r}: only 'ip' (cosine on L2-normalized vectors) is wired.")
        # torch + faiss-cpu both link libomp on macOS. KMP_DUPLICATE_LIB_OK lets them coexist, but that
        # alone still SEGFAULTs when both run OpenMP compute (torch MPS + faiss) — verified. Pinning faiss
        # to a single OpenMP thread avoids the collision (negligible cost at this index size, and correct
        # for the serving path too, where one process encodes with torch then searches with faiss).
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import faiss

        faiss.omp_set_num_threads(1)
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    def add(self, ids, vectors) -> "VectorIndex":
        """Add L2-normalized `vectors` (N, dim) under int64 `ids` (N,). Callable repeatedly (incremental)."""
        if self._index is None:
            self._build()
        v = np.ascontiguousarray(vectors, dtype="float32")
        i = np.ascontiguousarray(ids, dtype="int64")
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"vectors must be (N, {self.dim}); got {v.shape}")
        if i.shape != (len(v),):
            raise ValueError(f"ids must be ({len(v)},) to match vectors; got {i.shape}")
        self._index.add_with_ids(v, i)
        return self

    def search(self, query, top_k: int):
        """Nearest neighbours by cosine. `query` is (dim,) or (N, dim). Returns (ids, scores):
        for a single query, 1-D arrays of length k; for a batch, (N, k) arrays. ids are the ones you added."""
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("index is empty — add() vectors before search().")
        q = np.ascontiguousarray(query, dtype="float32")
        single = q.ndim == 1
        if single:
            q = q[None]
        if q.shape[1] != self.dim:
            raise ValueError(f"query must have dim {self.dim}; got {q.shape}")
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(q, k)          # faiss returns (D, I)
        return (ids[0], scores[0]) if single else (ids, scores)

    def __len__(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)
