"""Appearance re-ID clustering + builder tests — numpy only (boxmot/OSNet is lazy), so CI-safe."""

from types import SimpleNamespace

import numpy as np

from hoopvec.config import ReidConfig, TrackConfig
from hoopvec.pipeline import Track
from hoopvec.reid.identify import ReIDIdentifier, agglomerate, build_reid
from hoopvec.reid.jersey import majority_number


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


def test_jersey_majority_vote():
    assert majority_number(["32", "32", "30", "32"], 2) == "32"   # 3 votes for 32 >= 2
    assert majority_number(["5", "7"], 2) is None                 # top has 1 vote < 2 -> abstain
    assert majority_number([], 2) is None


def test_build_reid_carries_jersey_flag():
    on = SimpleNamespace(reid=ReidConfig(enabled=True, jersey_ocr=True, jersey_min_votes=3), track=TrackConfig())
    r = build_reid(on)
    assert r.jersey_ocr is True and r.jersey_min_votes == 3


def test_build_reid_carries_stitch_config():
    on = SimpleNamespace(reid=ReidConfig(enabled=True, stitch=True, stitch_gap=45, stitch_dist=1.5),
                         track=TrackConfig())
    r = build_reid(on)
    assert r.stitch is True and r.stitch_gap == 45 and r.stitch_dist == 1.5


def test_identify_stitch_pools_fragments_into_one_player(monkeypatch):
    # two temporally-disjoint, spatially-adjacent fragments = the same player split by the tracker.
    tracks = ([Track(1, f, "athlete", (90, 0, 110, 40)) for f in range(1, 6)]
              + [Track(2, f, "athlete", (95, 0, 115, 40)) for f in range(8, 13)])

    # mock OSNet: one ORTHOGONAL embedding per (canonical) track id -> without stitching, ids 1 and 2 would be
    # distinct appearance clusters; with stitching they collapse to one canonical id -> one identity.
    def fake_emb(self, cts, frames):
        cids = sorted({t.track_id for t in cts})
        return {cid: np.eye(len(cids))[i] for i, cid in enumerate(cids)}

    monkeypatch.setattr(ReIDIdentifier, "track_embeddings", fake_emb)
    out = ReIDIdentifier(stitch=True, jersey_ocr=False).identify(tracks, frames=[])
    pids = {t.track_id: t.player_id for t in out}
    assert set(pids) == {1, 2}                       # raw tracker ids preserved on the output tracks
    assert pids[1] == pids[2] is not None            # but pooled into ONE player_id by stitching
