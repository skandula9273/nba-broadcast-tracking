"""TrackResult -> (T,11,2) converter (increment-10): the re-ID-substitute ordering + tensor build, CI-safe."""

import numpy as np

from hoopvec.pipeline import Track, TrackResult
from hoopvec.retrieve.from_tracks import order_tracks, track_result_to_tensor


def _res(tracks):
    return TrackResult(tracks=tracks, meta={})


def test_order_by_length_then_first_x():
    tracks = ([Track(2, f, "athlete", (10, 0, 20, 20)) for f in (1, 2, 3)]      # len 3, first cx 15
              + [Track(1, f, "athlete", (0, 0, 10, 20)) for f in (1, 2)]         # len 2, first cx 5
              + [Track(3, f, "athlete", (100, 0, 110, 20)) for f in (1, 2, 3)])  # len 3, first cx 105
    # length desc puts the two len-3 tracks first; the tie breaks on first-frame x asc (15 < 105)
    assert order_tracks(_res(tracks), 1, 3, 10) == [2, 3, 1]


def test_tensor_shape_ball_zero_and_court_projection():
    H = np.eye(3)                                              # identity -> court_ft == pixel, then /94,/50
    tracks = [Track(1, f, "athlete", (0, 0, 18.8, 50)) for f in range(1, 5)]   # foot (9.4, 50)
    ten = track_result_to_tensor(_res(tracks), H, 1, 4, T=4, n_players=10)
    assert ten.shape == (4, 11, 2)
    assert np.allclose(ten[:, 0, :], 0.0)                     # entity 0 (ball) never filled
    assert np.allclose(ten[:, 1, 0], 9.4 / 94.0) and np.allclose(ten[:, 1, 1], 1.0)


def test_keeps_only_n_players_longest():
    tracks = []
    for tid in range(1, 13):                                  # 12 tracks; keep 10
        tracks += [Track(tid, f, "athlete", (tid * 10.0, 0, tid * 10 + 8, 20)) for f in range(1, 6)]
    ten = track_result_to_tensor(_res(tracks), np.eye(3), 1, 5, T=4, n_players=10)
    assert sum(1 for k in range(1, 11) if ten[:, k].any()) == 10
