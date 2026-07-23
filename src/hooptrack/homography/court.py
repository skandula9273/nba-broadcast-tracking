"""Court registration -> homography -> image pixels to real court coordinates. (V1)

Plan: KaliCalib-style keypoint estimation (encoder-decoder) or a SegFormer camera-param estimator,
then OpenCV DLT/RANSAC for the homography. Fills Track.court_xy.
"""
from __future__ import annotations

from ..pipeline import Track


class CourtHomography:
    def project(self, tracks: list[Track], frames) -> list[Track]:
        raise NotImplementedError("homography: implement KaliCalib keypoints + OpenCV DLT. See design-doc.")
