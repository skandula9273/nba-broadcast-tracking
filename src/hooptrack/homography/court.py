"""Court registration -> homography -> image pixels to real court coordinates. (V1, increment-05)

Classical (no-training) baseline, per the simpler-first rule. This module is the homography *machinery*:
turn image<->court point correspondences into a homography (OpenCV DLT + RANSAC), project image points
(player feet) to court coordinates (fills `Track.court_xy` -> the top-down "moving dots"), and measure
**reprojection error** against ground-truth camera calibration.

Ground truth: DeepSportradar provides per-image camera calibration (K, R, T). The court is the plane z=0,
so the GT court->image homography is `H = K [r1 r2 T]` (validated: it projects court points into the FOV).

The remaining piece is the *registration front-end* — detecting court keypoints in an unlabelled image to
supply the correspondences. A classical line/keypoint detector is the baseline; a learned detector
(KaliCalib) is the measured upgrade if the classical front-end underperforms (deferred, not gold-plated).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..pipeline import Track


def homography_from_calibration(calib: dict) -> np.ndarray:
    """GT court(z=0, cm) -> image homography from a DeepSportradar calibration dict: H = K [r1 r2 T]."""
    K = np.asarray(calib["KK"], float).reshape(3, 3)
    R = np.asarray(calib["R"], float).reshape(3, 3)
    T = np.asarray(calib["T"], float).reshape(3, 1)
    H = K @ np.hstack([R[:, 0:1], R[:, 1:2], T])
    return H / H[2, 2]


def solve_homography(court_pts, img_pts, ransac_thresh: float = 5.0):
    """Court<->image correspondences -> homography (OpenCV DLT + RANSAC). Returns (H_court2img, inlier_mask)."""
    import cv2

    court = np.asarray(court_pts, float).reshape(-1, 1, 2)
    img = np.asarray(img_pts, float).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(court, img, cv2.RANSAC, ransac_thresh)
    return H, mask


def _apply(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ hom.T).T
    return proj[:, :2] / proj[:, 2:3]


def image_to_court(img_xy, H_court2img: np.ndarray) -> np.ndarray:
    """Project image points (e.g. player feet) to court coordinates via the inverse homography."""
    return _apply(np.linalg.inv(H_court2img), img_xy)


def reprojection_error(H_pred: np.ndarray, H_gt: np.ndarray, court_pts) -> float:
    """Mean pixel distance between court points projected into the image by the predicted vs GT homography."""
    return float(np.mean(np.linalg.norm(_apply(H_pred, court_pts) - _apply(H_gt, court_pts), axis=1)))


def foot_point(xyxy) -> tuple[float, float]:
    """Bottom-center of a track box — the point that sits on the court plane."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, float(y2))


class CourtHomography:
    """Fills `Track.court_xy` by projecting each track's foot point to court coords.

    `register` is the front-end: a callable `(frame_idx, frames) -> H_court2img | None`. When it can't
    register a frame (returns None), that frame's tracks keep `court_xy = None` — honest about registration
    failures rather than emitting a bogus position (the known-failure-mode rule).
    """

    def __init__(self, register) -> None:
        self.register = register

    def project(self, tracks: list[Track], frames) -> list[Track]:
        H_cache: dict[int, np.ndarray | None] = {}
        for t in tracks:
            if t.frame not in H_cache:
                H_cache[t.frame] = self.register(t.frame, frames)
            H = H_cache[t.frame]
            if H is None:
                continue
            cx, cy = image_to_court(foot_point(t.xyxy), H)[0]
            t.court_xy = (float(cx), float(cy))
        return tracks


# ---- registration front-ends (the "calibration source" for the stage) ----
# The keypoint front-end (detect court lines in an unlabelled frame) stays deferred. What IS supported is a
# provided calibration: a FIXED court->image homography for the clip. Fine for a static camera; for a moving
# broadcast camera it's only an approximation, so it's opt-in via config, not a default.

def fixed_register(H_court2img):
    """A registration front-end that returns the same court->image homography for every frame."""
    H = np.asarray(H_court2img, float).reshape(3, 3)
    return lambda frame_idx, frames: H


def load_court2img(path: str | Path) -> np.ndarray:
    """Load a court->image homography: a DeepSportradar calibration JSON (`{"calibration": {KK,R,T}}` or the
    calibration dict itself), a `.npy` 3x3, or a whitespace-text 3x3."""
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        return homography_from_calibration(d.get("calibration", d))
    if path.suffix == ".npy":
        return np.load(path).reshape(3, 3)
    return np.loadtxt(path).reshape(3, 3)


def build_homography(cfg):
    """Build the `CourtHomography` pipeline stage from config, or None. Registration source, in priority:
    a trained court-keypoint detector (`homography.keypoint_weights`, per-frame), else a provided fixed
    calibration (`homography.calibration`). Disabled or neither -> None (tracks honestly keep court_xy=None)."""
    hc = cfg.homography
    if not hc.enabled:
        return None
    if getattr(hc, "keypoint_weights", None):
        from .keypoint_net import learned_register
        return CourtHomography(learned_register(hc.keypoint_weights, cfg.detect.device))
    if getattr(hc, "calibration", None):
        return CourtHomography(fixed_register(load_court2img(hc.calibration)))
    return None
