"""tracks_to_tensor adapter — pipeline Track output -> (T,11,2) tensor, numpy only, CI-safe."""

import numpy as np

from hooptrack.pipeline import Track
from hooptrack.retrieve.reconstruct import frame_windows, tracks_to_tensor


def _track(tid, frame, cx, cy):
    return Track(track_id=tid, frame=frame, cls="athlete", xyxy=(cx - 10, cy - 20, cx + 10, cy))


def test_shape_and_ball_slot_zero():
    tracks = [_track(1, f, 100, 200) for f in range(1, 11)]
    ten = tracks_to_tensor(tracks, 1, 10, T=8, width=1000, height=500)
    assert ten.shape == (8, 11, 2)
    assert np.allclose(ten[:, 0, :], 0.0)                 # entity 0 (ball) never filled -> zero


def test_image_normalization_uses_foot_point():
    # one player centered at x=500 (foot y=400) in a 1000x400 frame -> normalized (0.5, 1.0) in slot 1
    tracks = [_track(1, f, 500, 400) for f in range(1, 6)]
    ten = tracks_to_tensor(tracks, 1, 5, T=4, width=1000, height=400)
    assert np.allclose(ten[:, 1, 0], 0.5) and np.allclose(ten[:, 1, 1], 1.0)


def test_canonical_left_to_right_slot_order():
    # two players: id 2 on the right (x=800), id 1 on the left (x=200) -> left fills slot 1, right slot 2
    tracks = ([_track(2, f, 800, 300) for f in range(1, 6)]
              + [_track(1, f, 200, 300) for f in range(1, 6)])
    ten = tracks_to_tensor(tracks, 1, 5, T=4, width=1000, height=600)
    assert ten[0, 1, 0] < ten[0, 2, 0]                    # slot 1 (left) x < slot 2 (right) x


def test_keeps_only_n_players_most_present():
    tracks = ([_track(1, f, 100, 200) for f in range(1, 11)]     # present 10 frames
              + [_track(2, f, 300, 200) for f in range(1, 11)]   # present 10 frames
              + [_track(3, f, 500, 200) for f in range(1, 3)])   # present 2 frames -> dropped when n=2
    ten = tracks_to_tensor(tracks, 1, 10, T=6, width=1000, height=400, n_players=2)
    filled = [k for k in range(1, 11) if ten[:, k].any()]
    assert len(filled) == 2                               # only the 2 most-present ids


def test_game_split_groups_clips_by_game():
    from hooptrack.retrieve.broadcast_encoder import _game
    assert _game("v_00HRwkvvjtQ_c007") == "v_00HRwkvvjtQ"
    assert _game("v_5ekaksddqrc_c003") == "v_5ekaksddqrc"      # trailing 'rc' not confused with the _c split
    assert _game("v_4r8QL_wglzQ_c001") == "v_4r8QL_wglzQ"


def test_frame_windows_non_overlapping():
    tracks = [_track(1, f, 100, 200) for f in range(1, 101)]
    wins = frame_windows(tracks, window=48)
    assert wins[0] == (1, 48) and wins[1] == (49, 96)
    assert all(b - a + 1 == 48 for a, b in wins)
