"""FastAPI service. `POST /track` runs the ONE shared pipeline (detect -> track) — the same `Pipeline` the
eval harness calls — and returns image-coordinate tracks. The `source` may be a **video clip** (decoded to
frames via `ingest.extract_frames`) or a prepared **MOT sequence dir**.

Honest scope: only **detect -> track** is wired. Homography and re-ID are disabled (the pipeline is built
with `homography=None, reid=None`), so each track's `court_xy` and `player_id` are null — these are 2D
image-box tracks, NOT top-down "moving dots". `/health` is a liveness check; `/metrics` returns rolling
serving observability (per-stage latency percentiles, throughput, a detections/frame drift signal — V2).

Detector via env `HOOPVEC_CONFIG` (default `configs/v0_finetuned_640.yaml` -> the fine-tuned athlete detector
at imgsz 640, the measured Pareto-optimal operating point: peak mAP 0.987 / HOTA 0.525 / ~25 fps, beating the
old imgsz-1280 default on both accuracy and speed). Needs the local fine-tuned weights (`weights/finetuned/
best.pt`, gitignored); set `HOOPVEC_CONFIG=configs/v0.yaml` for a weights-free COCO yolov8m fallback (the
former `HOOPTRACK_CONFIG` is still honoured for back-compat). The
heavy stack (YOLO/boxmot) is imported lazily on the first /track call, so `/health` stays cheap.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .observability import Metrics

app = FastAPI(title="hoopvec")
_PIPELINE = None  # (Config, Pipeline) built once on the first /track call
_METRICS = Metrics()  # rolling serving metrics fed by /track, exposed at /metrics (V2 observability)


class TrackRequest(BaseModel):
    source: str                       # a video file to decode, OR a MOT sequence dir (img1/ + seqinfo.ini)
    max_frames: int | None = None     # optional cap (keeps a demo call fast)
    every: int = 1                    # video only: keep every Nth decoded frame


def _pipeline():
    """Build the shared detect->track pipeline once (same construction as track/run.py)."""
    global _PIPELINE
    if _PIPELINE is None:
        from ..config import load_config
        from ..detect.detector import build_detector
        from ..homography.court import build_homography
        from ..pipeline import Pipeline
        from ..reid.identify import build_reid
        from ..track.tracker import build_tracker

        # Config via env: HOOPVEC_CONFIG (the deprecated HOOPTRACK_CONFIG is still read for back-compat).
        explicit = os.environ.get("HOOPVEC_CONFIG") or os.environ.get("HOOPTRACK_CONFIG")
        cfg = load_config(explicit or "configs/v0_finetuned_640.yaml")
        # The Pareto-optimal default uses the fine-tuned weights (gitignored, local-only). On a fresh checkout
        # they're absent — fall back to the weights-free COCO config with a clear warning rather than crashing
        # on the first /track. An explicit config env var is always honoured as-is (no silent override).
        if explicit is None and cfg.detect.weights and not Path(cfg.detect.weights).exists():
            print(f"[serve] fine-tuned weights {cfg.detect.weights!r} not found -> falling back to configs/v0.yaml "
                  "(COCO yolov8m). Set HOOPVEC_CONFIG to override.", flush=True)
            cfg = load_config("configs/v0.yaml")
        _PIPELINE = (cfg, Pipeline(cfg=cfg, detector=build_detector(cfg.detect),
                                   tracker=build_tracker(cfg.track), homography=build_homography(cfg),
                                   reid=build_reid(cfg)))
    return _PIPELINE


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Rolling serving observability: per-stage latency percentiles, throughput, detections/frame baseline."""
    return _METRICS.summary()


@app.post("/track")
def track(req: TrackRequest) -> dict:
    import tempfile

    from ..ingest.frames import extract_frames, load_mot_sequence

    src = Path(req.source)
    if src.is_file():                                   # a video clip -> decode to frames on disk
        seq = extract_frames(src, tempfile.mkdtemp(prefix="hoopvec_"), every=req.every,
                             max_frames=req.max_frames)
        source_kind = "video (decoded)"
    elif (src / "img1").is_dir() or (src / "seqinfo.ini").is_file():
        seq = load_mot_sequence(src, max_frames=req.max_frames)
        source_kind = "mot_sequence"
    else:
        raise HTTPException(404, f"'{src}' is neither a video file nor a MOT sequence dir (img1/ + seqinfo.ini)")
    if len(seq) == 0:
        raise HTTPException(400, f"no frames from '{src}'")

    from ..analytics.possessions import analytics

    cfg, pipe = _pipeline()
    result = pipe.run(seq)   # THE shared pipeline path; homography/re-ID disabled in config -> detect->track only

    ball_by_frame = None     # optional COCO sports-ball pass -> true possessions where the ball is visible
    if cfg.detect.ball:
        from ..detect.ball import BallDetector
        ball_by_frame = BallDetector(device=cfg.detect.device, conf=cfg.detect.ball_conf,
                                     imgsz=cfg.detect.imgsz).detect(seq)

    timings = result.meta.get("timings", {})   # per-stage wall-clock -> observability (/metrics) + drift verdict
    drift = _METRICS.record(len(seq), timings, len(result.tracks), result.meta.get("n_dets", 0))
    return {
        "sequence": seq.name,
        "source": source_kind,
        "frames": len(seq),
        "image_size": {"width": seq.width, "height": seq.height},
        "pipeline": {
            "detector": cfg.detect.model, "weights": cfg.detect.weights or "yolov8m.pt (COCO)",
            "tracker": cfg.track.method,
            "homography_enabled": cfg.homography.enabled, "reid_enabled": cfg.reid.enabled,
        },
        "timings_s": timings, "drift": drift,
        "n_ids": result.meta.get("n_ids"),
        "n_tracks": len(result.tracks),
        "analytics": analytics(result, ball_by_frame),   # true possessions if ball on; else spacing + phases
        "coords": "image xyxy — court_xy & player_id are null (homography/re-ID not wired)",
        "tracks": [
            {
                "frame": t.frame, "track_id": t.track_id, "cls": t.cls,
                "xyxy": [round(float(v), 1) for v in t.xyxy], "score": round(float(t.score), 4),
                "court_xy": t.court_xy, "player_id": t.player_id,
            }
            for t in result.tracks
        ],
    }
