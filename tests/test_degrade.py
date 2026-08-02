"""Reconstruction-error model tests (increment-07) — numpy only, CI-safe."""

import numpy as np

from hoopvec.retrieve.degrade import (
    dropout_interp,
    id_swap,
    jitter_ft,
    permute_players,
    reconstruct,
)


def _poss(T=16, mid=False):
    rng = np.random.default_rng(0)
    P = rng.random((T, 11, 2))
    return 0.3 + 0.4 * P if mid else P  # mid: keep away from [0,1] edges to avoid clip in magnitude checks


def _perframe_pointset(P):
    # the multiset of player coordinates per frame (invariant under pure relabeling: swap / permute)
    return np.sort(P[:, 1:].reshape(P.shape[0], -1), axis=1)


def test_all_preserve_shape_and_range():
    P = _poss()
    rng = np.random.default_rng(1)
    for out in (jitter_ft(P, rng, 2.0), dropout_interp(P, rng, 0.2),
                id_swap(P, rng, 2), permute_players(P, rng, 6),
                reconstruct(P, rng, 1.0, 0.1, 2)):
        assert out.shape == P.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_jitter_zero_is_identity_and_monotone():
    P = _poss(mid=True)
    assert np.array_equal(jitter_ft(P, np.random.default_rng(0), 0.0), P)
    dev1 = np.abs(jitter_ft(P, np.random.default_rng(2), 0.5) - P).mean()
    dev2 = np.abs(jitter_ft(P, np.random.default_rng(2), 4.0) - P).mean()
    assert 0 < dev1 < dev2  # more feet of noise -> more deviation


def test_dropout_zero_identity_and_changes_frames():
    P = _poss()
    assert np.array_equal(dropout_interp(P, np.random.default_rng(0), 0.0), P)
    out = dropout_interp(P, np.random.default_rng(3), 0.25)
    assert not np.array_equal(out, P)  # some frames were blanked+interpolated


def test_id_swap_is_relabeling():
    P = _poss()
    out = id_swap(P, np.random.default_rng(4), 2)
    assert np.array_equal(out[:, 0], P[:, 0])                    # ball untouched
    assert np.allclose(_perframe_pointset(out), _perframe_pointset(P))  # same points, relabeled
    assert not np.array_equal(out[:, 1:], P[:, 1:])             # slots actually changed


def test_permute_players_is_relabeling():
    P = _poss()
    assert np.array_equal(permute_players(P, np.random.default_rng(0), 1), P)  # <2 -> no-op
    out = permute_players(P, np.random.default_rng(5), 10)
    assert np.array_equal(out[:, 0], P[:, 0])
    assert np.allclose(_perframe_pointset(out), _perframe_pointset(P))          # whole-possession relabel


def test_deterministic_under_seed():
    P = _poss()
    a = reconstruct(P, np.random.default_rng(7), 1.0, 0.1, 2)
    b = reconstruct(P, np.random.default_rng(7), 1.0, 0.1, 2)
    assert np.array_equal(a, b)
