"""Semantic-probe label/relevant-set tests (numpy logic; torch-gated because the module imports torch)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")

from hoopvec.retrieve.semantic_probe import (  # noqa: E402
    SCHEMES,
    label_handler,
    label_side,
    label_transition,
    merge_small,
    precision_recall_at_k,
    relevant_sets,
)

THR = {  # mirrors configs/semantic_probe.yaml
    "min_bucket_frac": 0.10,
    "transition": {"advance_frames": 12, "advance_thresh": 0.15},
    "initiation_side": {"entry_frames": 6, "left_max": 0.34, "right_min": 0.66},
    "handler_change": {"count_thresh": 8},
}


def _corpus(n=60, seed=0):
    return np.random.default_rng(seed).random((n, 48, 11, 2)).astype(np.float32)


def test_label_functions_deterministic_under_fixed_seed():
    P = _corpus(1, seed=7)[0]
    for fn in (label_transition, label_side, label_handler):
        assert fn(P, THR) == fn(P, THR)  # pure functions — same possession -> same label every call


def test_buckets_partition_the_corpus():
    C = _corpus(60)
    for _name, (fn, order) in SCHEMES.items():
        labels = [fn(C[i], THR) for i in range(len(C))]
        assert len(labels) == len(C)                       # exactly one label per possession
        assert all(v in order for v in labels)             # every label is a declared bucket
        idx_by_bucket = {b: {i for i, v in enumerate(labels) if v == b} for b in set(labels)}
        assert set().union(*idx_by_bucket.values()) == set(range(len(C)))   # cover
        assert sum(len(s) for s in idx_by_bucket.values()) == len(C)         # disjoint


def test_relevant_set_is_not_index_identity():
    labels = np.array([0, 0, 0, 1, 1, 1, 1])   # every bucket has >= 3 members
    for i, s in enumerate(relevant_sets(labels)):
        assert i not in s          # self excluded
        assert s != {i}            # NOT the query's own index (the committed eval's positive)
        assert len(s) > 1          # a real same-bucket relevant set


def test_merge_small_removes_degenerate_bucket():
    labels = np.array(["a"] * 95 + ["b"] * 5)   # 'b' is 5% < 10%
    merged, merges = merge_small(labels, 0.10)
    assert merges == {"b": "a"} and set(merged.tolist()) == {"a"}


def test_precision_at_5_rewards_clustering():
    labels = np.array([0] * 20 + [1] * 20)
    emb = np.zeros((40, 4), np.float32)
    emb[:20, 0] = 1.0
    emb[20:, 1] = 1.0                           # two orthogonal clusters, one per bucket
    prec, _ = precision_recall_at_k(emb, labels, k=5)
    assert prec.mean() > 0.9                    # neighbours are same-bucket -> high precision
