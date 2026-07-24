"""Tracking stage: detections -> consistent track IDs across frames.

Config lever (`track.method`): bytetrack (V0 baseline) | botsort | deepeiou. Honest note from SportsMOT:
Kalman-only motion models underperform on sports (fast, variable motion); motion+appearance fusion wins —
so ByteTrack is the baseline and BoT-SORT is the first measured upgrade. Both implement `Tracker` and share
one online loop (`_boxmot_tracks`): group detections by frame, feed them in order, one `update` per frame.

- **ByteTrack** is motion-only (Kalman + IoU). It never looks at pixels, so a blank canvas satisfies
  boxmot's shape check — honest, since no pixel content is used.
- **BoT-SORT** adds appearance (OSNet ReID) + camera-motion compensation, so it needs the **real frames**;
  we imread each frame's image lazily. It uses boxmot's tuned `botsort.yaml` defaults (fixed by the pinned
  boxmot version); only the ReID weight + device are ours.

APIs verified against boxmot 22 (rule #1): `boxmot.trackers.bbox.bytetrack.ByteTrack`,
`boxmot.trackers.registry.create_tracker('botsort', ...)`, `update(dets, img)` takes Nx6
`[x1,y1,x2,y2,conf,cls]` and returns a `TrackResults` with `.id/.xyxy/.conf`.
"""
from __future__ import annotations

import numpy as np

from ..config import TrackConfig
from ..pipeline import Detection, Track


def _boxmot_tracks(tracker, detections: list[Detection], frames, use_pixels: bool) -> list[Track]:
    """Shared online loop over a boxmot tracker. Fresh trackers are built per call (per-sequence ID isolation)."""
    height = int(getattr(frames, "height", 720) or 720)
    width = int(getattr(frames, "width", 1280) or 1280)

    by_frame: dict[int, list[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame, []).append(d)
    try:
        n_frames = len(frames)
    except TypeError:
        n_frames = 0
    n_frames = max(n_frames, max(by_frame) if by_frame else 0)

    if use_pixels:
        import cv2

        frame_paths = {i: p for i, p in frames}  # appearance/CMC need the actual image
    blank = None if use_pixels else np.zeros((height, width, 3), dtype=np.uint8)

    tracks: list[Track] = []
    for fidx in range(1, n_frames + 1):
        if use_pixels:
            img = cv2.imread(str(frame_paths[fidx])) if fidx in frame_paths else None
            if img is None:
                img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            img = blank
        fr = by_frame.get(fidx, [])
        arr = (
            np.array([[*d.xyxy, d.conf, 0.0] for d in fr], dtype=float)
            if fr
            else np.empty((0, 6), dtype=float)  # still step motion/CMC on empty frames
        )
        res = tracker.update(arr, img)
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


class ByteTrackTracker:
    """ByteTrack baseline (via boxmot). Motion-only association — the V0 floor."""

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg

    def params(self) -> dict:
        c = self.cfg
        return {
            "method": "bytetrack",
            "min_conf": c.min_conf,
            "track_thresh": c.track_thresh,
            "match_thresh": c.match_thresh,
            "track_buffer": c.track_buffer,
            "with_reid": False,
        }

    def track(self, detections: list[Detection], frames) -> list[Track]:
        from boxmot.trackers.bbox.bytetrack import ByteTrack

        c = self.cfg
        fps = int(round(getattr(frames, "fps", 30) or 30))
        bt = ByteTrack(
            min_conf=c.min_conf,
            track_thresh=c.track_thresh,
            match_thresh=c.match_thresh,
            track_buffer=c.track_buffer,
            frame_rate=fps,
        )
        return _boxmot_tracks(bt, detections, frames, use_pixels=False)


class BotSortTracker:
    """BoT-SORT (via boxmot): motion + appearance (OSNet ReID) + camera-motion compensation.

    The first measured tracking upgrade over ByteTrack — attacks the ID-switches that motion-only tracking
    can't hold through fast basketball motion. Uses boxmot's tuned botsort config (params fixed by the
    pinned boxmot version); needs real frame pixels.
    """

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg

    def params(self) -> dict:
        return {
            "method": "botsort",
            "reid_weights": self.cfg.reid_weights,
            "device": self.cfg.device,
            "with_reid": True,
            "use_cmc": True,
            "config": "boxmot default botsort.yaml (tuned; thresholds fixed by pinned boxmot version)",
        }

    def track(self, detections: list[Detection], frames) -> list[Track]:
        from pathlib import Path

        from boxmot.trackers.registry import create_tracker

        c = self.cfg
        fps = int(round(getattr(frames, "fps", 30) or 30))
        tracker = create_tracker(
            "botsort",
            reid_weights=Path(c.reid_weights),
            device=c.device,
            half=False,
            tracker_kwargs={"frame_rate": fps},
        )
        return _boxmot_tracks(tracker, detections, frames, use_pixels=True)


def build_tracker(cfg: TrackConfig):
    if cfg.method == "bytetrack":
        return ByteTrackTracker(cfg)
    if cfg.method == "botsort":
        return BotSortTracker(cfg)
    raise NotImplementedError(f"tracker '{cfg.method}' not wired yet (bytetrack | botsort | deepeiou).")
