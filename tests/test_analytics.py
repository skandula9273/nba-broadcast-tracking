"""Analytics tests — numpy only, synthetic tracks, CI-safe."""

from hooptrack.analytics.possessions import (
    analytics,
    player_points,
    segment_phases,
    spacing,
)
from hooptrack.pipeline import Track, TrackResult


def _res(tracks):
    return TrackResult(tracks=tracks, meta={})


def test_player_points_uses_court_xy_else_foot():
    t1 = Track(1, 1, "player", (0, 0, 10, 20), court_xy=(100.0, 200.0))
    t2 = Track(2, 1, "player", (0, 0, 10, 20))            # no court_xy -> foot point (5, 20)
    assert player_points(_res([t1, t2]))[1] == [(100.0, 200.0), (5.0, 20.0)]


def test_spacing_mean_pairwise_in_court_units():
    ts = [Track(1, 1, "player", (0, 0, 0, 0), court_xy=(0.0, 0.0)),
          Track(2, 1, "player", (0, 0, 0, 0), court_xy=(0.0, 10.0))]
    s = spacing(_res(ts))
    assert s["coords"] == "court_cm" and s["mean_spread"] == 10.0 and s["n_frames"] == 1


def test_segment_phases_returns_labeled_runs():
    ts = []
    for f in range(1, 11):
        ts += [Track(1, f, "player", (0, 0, 0, 0), court_xy=(f * 10.0, 0.0)),
               Track(2, f, "player", (0, 0, 0, 0), court_xy=(f * 10.0, 50.0))]
    runs = segment_phases(_res(ts), min_run=1)
    assert runs and all(r["phase"] in ("transition", "halfcourt") and r["start"] <= r["end"] for r in runs)


def test_analytics_states_no_ball():
    a = analytics(_res([Track(1, 1, "player", (0, 0, 10, 20)), Track(2, 1, "player", (20, 0, 30, 20))]))
    assert "spacing" in a and "possessions" in a
    assert "ball not tracked" in a["possessions"]["note"]     # honest: no ball -> no true possessions
