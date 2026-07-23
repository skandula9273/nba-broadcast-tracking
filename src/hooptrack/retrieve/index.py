"""Vector index wrapper for play/possession embeddings: FAISS or pgvector. (V1)

Thin interface so the retrieval core can swap backends via config (retrieval.index).
"""
from __future__ import annotations

from collections.abc import Sequence


class VectorIndex:
    def __init__(self, dim: int, backend: str = "faiss") -> None:
        self.dim = dim
        self.backend = backend

    def add(self, ids: Sequence, vectors) -> None:
        raise NotImplementedError("index.add: wire FAISS (faiss.IndexFlatIP) or pgvector.")

    def search(self, query, top_k: int):
        raise NotImplementedError("index.search: return top_k ids for a query vector.")
