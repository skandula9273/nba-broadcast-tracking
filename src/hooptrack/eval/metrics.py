"""Retrieval metrics for the play-embedding core — pure logic, no deps, unit-tested.

recall@k / hit-rate@k / MRR. These are the V1 retrieval metrics and they compute today. Tracking metrics
(HOTA/MOTA/IDF1) come from TrackEval on MOT-format outputs (see trackeval_adapter.py), and detection mAP
from the detector's own eval — those need the CV stack + data, so they live behind the pipeline.
"""
from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence, relevant: set, k: int) -> float:
    """Fraction of the relevant items present in the top-k retrieved. Empty relevant -> 0.0."""
    if not relevant:
        return 0.0
    topk = set(list(retrieved)[:k])
    return sum(1 for r in relevant if r in topk) / len(relevant)


def hit_rate_at_k(retrieved: Sequence, relevant: set, k: int) -> float:
    """1.0 if any relevant item is in the top-k, else 0.0."""
    if not relevant:
        return 0.0
    topk = list(retrieved)[:k]
    return 1.0 if any(r in relevant for r in topk) else 0.0


def reciprocal_rank(retrieved: Sequence, relevant: set) -> float:
    """1 / rank (1-indexed) of the first relevant item; 0.0 if none appears."""
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def mean_recall_at_k(rankings: Sequence[tuple[Sequence, set]], k: int) -> float:
    if not rankings:
        return 0.0
    return sum(recall_at_k(ret, rel, k) for ret, rel in rankings) / len(rankings)


def mean_reciprocal_rank(rankings: Sequence[tuple[Sequence, set]]) -> float:
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(ret, rel) for ret, rel in rankings) / len(rankings)
