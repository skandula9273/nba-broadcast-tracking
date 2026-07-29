"""Analytics tests — numpy only, synthetic tracks, CI-safe."""

from hooptrack.analytics.possessions import (
    analytics,
    ball_handler_timeline,
    ball_possession,
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


def test_ball_handler_is_nearest_player():
    # player 1 box-center at (5,10); player 2 at (105,10); ball at (100,10) -> handler = player 2
    ts = [Track(1, 1, "player", (0, 0, 10, 20)), Track(2, 1, "player", (100, 0, 110, 20))]
    assert ball_handler_timeline(_res(ts), {1: (100.0, 10.0, 0.6)}) == {1: 2}


def test_ball_possession_segments_by_handler_and_reports_coverage():
    ts = []
    for f in range(1, 7):                                      # p1 holds frames 1-3, p2 holds frames 4-6
        ts += [Track(1, f, "player", (0, 0, 10, 20)), Track(2, f, "player", (100, 0, 110, 20))]
    ball = {f: ((5.0 if f <= 3 else 105.0), 10.0, 0.6) for f in range(1, 7)}
    p = ball_possession(_res(ts), ball, min_run=1)
    assert p["ball_coverage"] == 1.0 and p["n_handler_changes"] == 1
    assert [(r["handler"], r["start"], r["end"]) for r in p["possessions"]] == [(1, 1, 3), (2, 4, 6)]


def test_analytics_uses_ball_when_supplied():
    ts = [Track(1, 1, "player", (0, 0, 10, 20)), Track(2, 1, "player", (100, 0, 110, 20))]
    a = analytics(_res(ts), ball_by_frame={1: (5.0, 10.0, 0.7)})
    assert "ball" in a["possessions"]["source"] and a["possessions"]["ball_frames"] == 1
