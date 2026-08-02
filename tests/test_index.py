"""FAISS VectorIndex tests (increment-06b). faiss-gated so the suite stays CI-safe.

The core guarantee: an exact flat inner-product index over L2-normalized vectors reproduces brute-force
cosine top-k retrieval bit-for-bit — so swapping the index in never changes the committed recall@k."""

import numpy as np
import pytest

pytest.importorskip("faiss")

from hoopvec.retrieve.index import VectorIndex  # noqa: E402


def _normed(n, d, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_flat_ip_reproduces_bruteforce_topk():
    X = _normed(50, 16)
    idx = VectorIndex(16).add(np.arange(50), X)
    q = X[:10]
    ids, scores = idx.search(q, 5)
    bf = np.argsort(-(q @ X.T), axis=1)[:, :5]
    assert np.array_equal(ids, bf)                     # same top-5, same order — the wiring check
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)   # each vector retrieves itself first, cosine 1


def test_arbitrary_ids_roundtrip():
    X = _normed(8, 4)
    ids_in = np.array([100, 7, 42, 9, 300, 5, 11, 88])
    idx = VectorIndex(4).add(ids_in, X)
    ids, _ = idx.search(X[3], 1)                        # 1-D query -> 1-D result
    assert int(ids[0]) == 9                             # row 3's attached id, not its position


def test_len_and_incremental_add():
    idx = VectorIndex(4)
    assert len(idx) == 0
    idx.add(np.arange(3), _normed(3, 4, 1))
    idx.add(np.arange(3, 7), _normed(4, 4, 2))
    assert len(idx) == 7


def test_single_vector_query_shapes():
    X = _normed(6, 4)
    idx = VectorIndex(4).add(np.arange(6), X)
    ids, scores = idx.search(X[2], 3)
    assert ids.shape == (3,) and scores.shape == (3,)
    assert int(ids[0]) == 2


def test_mirror_structured_still_exact():
    # exactness holds regardless of geometry (e.g. sign-flip 'mirror'-like structure)
    X = _normed(30, 12, 3)
    Xm = X.copy()
    Xm[:, ::2] *= -1
    allv = np.vstack([X, Xm]).astype("float32")
    idx = VectorIndex(12).add(np.arange(60), allv)
    ids, _ = idx.search(X[:5], 3)
    bf = np.argsort(-(X[:5] @ allv.T), axis=1)[:, :3]
    assert np.array_equal(ids, bf)


def test_search_empty_raises():
    with pytest.raises(RuntimeError):
        VectorIndex(4).search(_normed(1, 4)[0], 1)
