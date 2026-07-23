"""Tracking stage: detections -> consistent track IDs across frames.

Config lever (`track.method`): bytetrack (V0 baseline) | botsort | deepeiou. Honest note from SportsMOT:
Kalman-only motion models underperform on sports (fast, variable motion); motion+appearance fusion wins —
so ByteTrack is the baseline and BoT-SORT/Deep-EIoU are the measured upgrades. Implements `Tracker`.

ByteTrack is **motion-only**: it associates by IoU + a Kalman motion model and never looks at pixels.
boxmot still requires an image array for shape validation, so we pass a blank canvas sized to the
sequence — honest, since no pixel content is used. The later BoT-SORT ablation will pass real frames.
API verified against boxmot 22 (rule #1): `boxmot.trackers.bbox.bytetrack.ByteTrack`, `update(dets, img)`
takes Nx6 `[x1,y1,x2,y2,conf,cls]` and returns a `TrackResults` array with `.id/.xyxy/.conf`.
"""
from __future__ import annotations

import numpy as np

from ..config import TrackConfig
from ..pipeline import Detection, Track


class ByteTrackTracker:
    """ByteTrack baseline (via boxmot). Motion-only association — the V0 floor."""

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg

    def track(self, detections: list[Detection], frames) -> list[Track]:
        # lazy import so the package imports without the CV stack installed
        from boxmot.trackers.bbox.bytetrack import ByteTrack

        cfg = self.cfg
        height = int(getattr(frames, "height", 720) or 720)
        width = int(getattr(frames, "width", 1280) or 1280)
        fps = int(round(getattr(frames, "fps", 30) or 30))

        # Fresh tracker per sequence => IDs never leak across sequences.
        bt = ByteTrack(
            min_conf=cfg.min_conf,
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            frame_rate=fps,
        )
        blank = np.zeros((height, width, 3), dtype=np.uint8)

        by_frame: dict[int, list[Detection]] = {}
        for d in detections:
            by_frame.setdefault(d.frame, []).append(d)

        try:
            n_frames = len(frames)
        except TypeError:
            n_frames = 0
        n_frames = max(n_frames, max(by_frame) if by_frame else 0)

        tracks: list[Track] = []
        for fidx in range(1, n_frames + 1):
            fr = by_frame.get(fidx, [])
            if fr:
                arr = np.array([[*d.xyxy, d.conf, 0.0] for d in fr], dtype=float)
            else:
                arr = np.empty((0, 6), dtype=float)  # still step the motion model on empty frames
            res = bt.update(arr, img=blank)
            for tid, box, conf in zip(res.id, res.xyxy, res.conf):
                x1, y1, x2, y2 = box
                tracks.append(
                    Track(
                        track_id=int(tid),
                        frame=fidx,
                        cls="player",
                        xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        score=float(conf),
                    )
                )
        return tracks


def build_tracker(cfg: TrackConfig):
    if cfg.method == "bytetrack":
        return ByteTrackTracker(cfg)
    raise NotImplementedError(f"tracker '{cfg.method}' not wired yet (bytetrack | botsort | deepeiou).")
