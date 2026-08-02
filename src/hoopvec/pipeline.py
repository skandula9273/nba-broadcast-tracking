"""The one shared pipeline path. Both the API (serve) and the eval harness call this, so committed
numbers describe the deployed system (the SEC honesty rule).

Stages are injected as Protocols so each is independently testable and swappable via config — one
variable at a time. Stage implementations live in their own modules and are wired here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .config import Config


@dataclass
class Detection:
    frame: int
    cls: str
    xyxy: tuple[float, float, float, float]
    conf: float


@dataclass
class Track:
    track_id: int
    frame: int
    cls: str
    xyxy: tuple[float, float, float, float]
    score: float = 1.0                            # tracker/detector confidence -> MOT conf column
    court_xy: tuple[float, float] | None = None   # filled by homography (V1)
    player_id: str | None = None                  # filled by re-ID (V1)


@dataclass
class TrackResult:
    tracks: list[Track] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Detector(Protocol):
    def detect(self, frames) -> list[Detection]: ...


@runtime_checkable
class Tracker(Protocol):
    # `frames` mirrors the Homography/ReID protocols: motion-only trackers (ByteTrack) ignore it,
    # but appearance trackers (BoT-SORT/Deep-EIoU, the V1 ablation) need the pixels for embeddings.
    def track(self, detections: list[Detection], frames) -> list[Track]: ...


@runtime_checkable
class Homography(Protocol):
    def project(self, tracks: list[Track], frames) -> list[Track]: ...


@runtime_checkable
class ReID(Protocol):
    def identify(self, tracks: list[Track], frames) -> list[Track]: ...


@dataclass
class Pipeline:
    cfg: Config
    detector: Detector
    tracker: Tracker
    homography: Homography | None = None
    reid: ReID | None = None

    def run(self, frames) -> TrackResult:
        """broadcast frames -> tracks (in image coords; court coords + identity added in V1). Records per-stage
        wall-clock in meta['timings'] (cheap; used by the serving observability layer and any latency report)."""
        import time

        timings: dict[str, float] = {}
        t = time.perf_counter()
        dets = self.detector.detect(frames)
        timings["detect_s"] = round(time.perf_counter() - t, 4)
        t = time.perf_counter()
        tracks = self.tracker.track(dets, frames)
        timings["track_s"] = round(time.perf_counter() - t, 4)
        if self.cfg.homography.enabled and self.homography is not None:
            t = time.perf_counter()
            tracks = self.homography.project(tracks, frames)
            timings["homography_s"] = round(time.perf_counter() - t, 4)
        if self.cfg.reid.enabled and self.reid is not None:
            t = time.perf_counter()
            tracks = self.reid.identify(tracks, frames)
            timings["reid_s"] = round(time.perf_counter() - t, 4)
        return TrackResult(tracks=tracks, meta={"n_ids": len({t.track_id for t in tracks}),
                                                "n_dets": len(dets), "timings": timings})
