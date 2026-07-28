"""Appearance re-ID clustering + builder tests — numpy only (boxmot/OSNet is lazy), so CI-safe."""

from types import SimpleNamespace

import numpy as np

from hooptrack.config import ReidConfig, TrackConfig
from hooptrack.reid.identify import ReIDIdentifier, agglomerate, build_reid


def _l2rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_agglomerate_finds_two_appearance_groups():
    a = _l2rows(np.array([[1, 0, 0.02], [1, 0.02, 0], [1, 0.01, 0.01]]))   # group near [1,0,0]
    b = _l2rows(np.array([[0, 1, 0.02], [0.01, 1, 0], [0, 1, 0.01]]))       # group near [0,1,0]
    labels = agglomerate(np.vstack([a, b]), 0.9)
    assert len(set(labels.tolist())) == 2
    assert len(set(labels[:3].tolist())) == 1 and len(set(labels[3:].tolist())) == 1
    assert labels[0] != labels[3]


def test_agglomerate_all_merge_or_all_separate():
    E = _l2rows(np.random.default_rng(0).standard_normal((5, 8)))
    assert len(set(agglomerate(E, -1.1).tolist())) == 1   # threshold below -1 -> everything merges
    assert len(set(agglomerate(E, 1.01).tolist())) == 5   # threshold above 1 -> all singletons


def test_build_reid_off_returns_none_on_builds_lazy():
    off = SimpleNamespace(reid=ReidConfig(enabled=False), track=TrackConfig())
    assert build_reid(off) is None
    on = SimpleNamespace(reid=ReidConfig(enabled=True, sim_threshold=0.6), track=TrackConfig())
    r = build_reid(on)
    assert isinstance(r, ReIDIdentifier)
    assert r.sim_threshold == 0.6 and r._model is None    # OSNet not loaded until first identify()
