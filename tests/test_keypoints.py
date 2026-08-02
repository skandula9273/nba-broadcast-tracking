"""Court-keypoint front-end harness tests — numpy + opencv (the solver) only, so CI-safe."""

import numpy as np
from pytest import approx

from hoopvec.homography.court import reprojection_error
from hoopvec.homography.keypoints import (
    CANONICAL,
    _court_grid,
    evaluate,
    gt_keypoints,
    solve_from_keypoints,
)

# a court->image homography that maps the 2800x1500 court well inside a 1624x1234 image (all keypoints visible)
H = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0]])


def test_canonical_is_seven_unambiguous_points():
    assert len(CANONICAL) == 7
    assert CANONICAL["corner_bl"] == (0.0, 0.0) and CANONICAL["center"] == (1400.0, 750.0)


def test_gt_keypoints_project_and_solve_roundtrips():
    kp = gt_keypoints(H, 1624, 1234)
    assert len(kp) == 7                                   # all canonical points land in-frame
    H_solved = solve_from_keypoints(kp)                   # exact keypoints -> recover H
    assert reprojection_error(H_solved, H, _court_grid()) == approx(0.0, abs=1e-3)


def test_gt_keypoints_filter_out_of_frame():
    assert len(gt_keypoints(H, 10, 10)) < 7               # tiny image -> most keypoints fall outside


def test_solve_needs_four_points():
    assert solve_from_keypoints({"center": (1.0, 2.0)}) is None   # <4 -> unsolvable


def test_evaluate_floor_runs():
    insts = []
    for dx in (0.0, 20.0):
        Hx = H.copy()
        Hx[0, 2] = dx                                     # slightly shifted camera per instant
        insts.append({"H_gt": Hx, "w": 1624, "h": 1234, "image": None,
                      "gt_keypoints": gt_keypoints(Hx, 1624, 1234)})
    errs = evaluate(None, insts)                          # detector=None -> trivial global-mean floor
    assert len(errs) == 2 and all(e >= 0 for e in errs)
