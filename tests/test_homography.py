"""Homography machinery tests — numpy + opencv only (core deps), so they run in CI."""

from types import SimpleNamespace

import numpy as np
from pytest import approx

from hooptrack.config import HomographyConfig
from hooptrack.homography.court import (
    CourtHomography,
    build_homography,
    fixed_register,
    foot_point,
    homography_from_calibration,
    image_to_court,
    load_court2img,
    reprojection_error,
    solve_homography,
)
from hooptrack.pipeline import Track


def test_homography_from_calibration_identity():
    # K=I, R=I, T=[0,0,1] -> H = [r1 r2 T] = identity; court (x,y) maps to image (x,y)
    calib = {"KK": [1, 0, 0, 0, 1, 0, 0, 0, 1], "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "T": [0, 0, 1]}
    H = homography_from_calibration(calib)
    assert H == approx(np.eye(3))


def test_solve_and_reprojection_roundtrip():
    # a non-trivial synthetic court->image homography
    H_gt = np.array([[1.2, 0.1, 30.0], [0.05, 1.1, 20.0], [1e-4, 2e-4, 1.0]])
    court = np.array([[0, 0], [1400, 0], [2800, 0], [0, 1500], [1400, 750], [2800, 1500]], float)
    hom = np.hstack([court, np.ones((len(court), 1))])
    img = (H_gt @ hom.T).T
    img = img[:, :2] / img[:, 2:3]
    H_solved, mask = solve_homography(court, img)
    assert reprojection_error(H_solved, H_gt, court) == approx(0.0, abs=1e-3)
    assert reprojection_error(H_gt, H_gt, court) == 0.0
    # image->court recovers the court point
    back = image_to_court(img[4], H_gt)[0]
    assert back == approx(court[4], abs=1e-6)


def test_foot_point():
    assert foot_point((10, 20, 50, 80)) == (30.0, 80.0)  # bottom-center


def test_stage_projects_foot_point_to_court():
    H = np.array([[1.2, 0.1, 30.0], [0.05, 1.1, 20.0], [1e-4, 2e-4, 1.0]])   # court->image
    stage = CourtHomography(fixed_register(H))
    box = (100.0, 200.0, 140.0, 300.0)
    (out,) = stage.project([Track(track_id=1, frame=1, cls="player", xyxy=box)], frames=None)
    assert out.court_xy == approx(tuple(image_to_court(foot_point(box), H)[0]))


def test_stage_keeps_null_when_registration_fails():
    stage = CourtHomography(lambda _f, _frames: None)   # front-end can't register this frame -> honest null
    (out,) = stage.project([Track(track_id=1, frame=1, cls="player", xyxy=(0.0, 0.0, 10.0, 10.0))], None)
    assert out.court_xy is None


def test_build_homography_none_by_default_stage_when_calibrated(tmp_path):
    assert build_homography(SimpleNamespace(homography=HomographyConfig(enabled=False))) is None
    p = tmp_path / "H.npy"
    np.save(p, np.eye(3))
    on = SimpleNamespace(homography=HomographyConfig(enabled=True, calibration=str(p)))
    assert isinstance(build_homography(on), CourtHomography)


def test_load_court2img_from_calibration_json(tmp_path):
    import json

    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"calibration": {"KK": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                              "R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "T": [0, 0, 1]}}))
    assert load_court2img(p) == approx(np.eye(3))
