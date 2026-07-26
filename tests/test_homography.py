"""Homography machinery tests — numpy + opencv only (core deps), so they run in CI."""

import numpy as np
from pytest import approx

from hooptrack.homography.court import (
    foot_point,
    homography_from_calibration,
    image_to_court,
    reprojection_error,
    solve_homography,
)


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
