"""FastAPI service. `POST /track` runs the ONE shared pipeline (detect -> track) — the same `Pipeline` the
eval harness calls — and returns image-coordinate tracks. The `source` may be a **video clip** (decoded to
frames via `ingest.extract_frames`) or a prepared **MOT sequence dir**.

Honest scope: only **detect -> track** is wired. Homography and re-ID are disabled (the pipeline is built
with `homography=None, reid=None`), so each track's `court_xy` and `player_id` are null — these are 2D
image-box tracks, NOT top-down "moving dots". `/health` is a liveness check.

Detector via env `HOOPTRACK_CONFIG` (default `configs/v0.yaml` -> COCO yolov8m, auto-downloaded); point it at
`configs/v0_finetuned.yaml` for the fine-tuned athlete detector. The heavy stack (YOLO/boxmot) is imported
lazily on the first /track call, so `/health` stays cheap.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="hooptrack")
_PIPELINE = None  # (Config, Pipeline) built once on the first /track call


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
        from ..track.tracker import build_tracker

        cfg = load_config(os.environ.get("HOOPTRACK_CONFIG", "configs/v0.yaml"))
        _PIPELINE = (cfg, Pipeline(cfg=cfg, detector=build_detector(cfg.detect),
                                   tracker=build_tracker(cfg.track), homography=build_homography(cfg)))
    return _PIPELINE


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/track")
def track(req: TrackRequest) -> dict:
    import tempfile

    from ..ingest.frames import extract_frames, load_mot_sequence

    src = Path(req.source)
    if src.is_file():                                   # a video clip -> decode to frames on disk
        seq = extract_frames(src, tempfile.mkdtemp(prefix="hooptrack_"), every=req.every,
                             max_frames=req.max_frames)
        source_kind = "video (decoded)"
    elif (src / "img1").is_dir() or (src / "seqinfo.ini").is_file():
        seq = load_mot_sequence(src, max_frames=req.max_frames)
        source_kind = "mot_sequence"
    else:
        raise HTTPException(404, f"'{src}' is neither a video file nor a MOT sequence dir (img1/ + seqinfo.ini)")
    if len(seq) == 0:
        raise HTTPException(400, f"no frames from '{src}'")

    cfg, pipe = _pipeline()
    result = pipe.run(seq)   # THE shared pipeline path; homography/re-ID disabled in config -> detect->track only
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
        "n_ids": result.meta.get("n_ids"),
        "n_tracks": len(result.tracks),
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
