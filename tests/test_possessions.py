"""Possession augmentation + feature tests — numpy only (download/py7zr are lazy), so CI-safe."""

import numpy as np
from pytest import approx

from hooptrack.retrieve.possessions import augment, jitter, mirror, order_perturb, temporal_crop
from hooptrack.retrieve.run import features


def _poss(T=8):
    rng = np.random.default_rng(0)
    return rng.random((T, 11, 2))


def test_mirror_flips_the_right_axis():
    p = _poss()
    assert mirror(p, "length")[..., 0] == approx(1 - p[..., 0])
    assert mirror(p, "length")[..., 1] == approx(p[..., 1])  # width unchanged
    assert mirror(p, "width")[..., 1] == approx(1 - p[..., 1])


def test_crop_and_jitter_preserve_shape_and_range():
    p = _poss()
    rng = np.random.default_rng(1)
    c = temporal_crop(p, rng)
    j = jitter(p, rng)
    assert c.shape == p.shape and j.shape == p.shape
    assert j.min() >= 0 and j.max() <= 1
    assert np.abs(j - p).max() < 0.2  # jitter is small


def test_augment_dispatch():
    p = _poss()
    rng = np.random.default_rng(2)
    for kind in ("jitter", "crop", "mirror"):
        assert augment(p, rng, kind).shape == p.shape


def test_order_perturb_off_is_identity_and_relabels_when_on():
    p = _poss()
    rng = np.random.default_rng(3)
    assert np.array_equal(order_perturb(p, rng, p_swap=0.0, p_permute=0.0), p)  # default off -> no-op
    out = order_perturb(p, np.random.default_rng(4), p_permute=1.0, n_permute=10)
    assert np.array_equal(out[:, 0], p[:, 0])                                   # ball untouched
    per = lambda x: np.sort(x[:, 1:].reshape(x.shape[0], -1), axis=1)           # noqa: E731
    assert np.allclose(per(out), per(p))                                        # relabel: same points
    assert not np.array_equal(out[:, 1:], p[:, 1:])                             # order actually changed


def test_features_are_l2_normalized():
    batch = _poss(6)[None].repeat(4, axis=0)  # (4, 6, 11, 2)
    f = features(batch)
    assert f.shape == (4, 6 * 11 * 2)
    assert np.linalg.norm(f, axis=1) == approx(np.ones(4))
