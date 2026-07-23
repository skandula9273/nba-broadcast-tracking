"""Retrieval metric tests — hand-checked expected values. These pass today (pure logic, no CV stack)."""

from hooptrack.eval.metrics import (
    hit_rate_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k([1, 2, 3, 4], {3, 5}, 3) == 0.5   # top3={1,2,3}; hits {3} of 2 relevant
    assert recall_at_k([5, 1, 2], {5}, 1) == 1.0
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0       # empty relevant -> 0.0


def test_hit_rate_at_k():
    assert hit_rate_at_k([1, 2, 3], {3}, 3) == 1.0
    assert hit_rate_at_k([1, 2, 3], {3}, 2) == 0.0       # 3 is at rank 3, not in top-2
    assert hit_rate_at_k([1, 2, 3], {9}, 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([9, 3, 7], {3}) == 0.5
    assert reciprocal_rank([3, 1], {3}) == 1.0
    assert reciprocal_rank([9, 8, 7], {3}) == 0.0


def test_means():
    rankings = [([1, 2, 3], {3}), ([9, 8, 7], {1})]
    assert mean_recall_at_k(rankings, 3) == (1.0 + 0.0) / 2
    assert mean_reciprocal_rank(rankings) == ((1 / 3) + 0.0) / 2
