"""FastAPI service. `POST /track` runs the ONE shared pipeline (detect -> track) — the same `Pipeline` the
eval harness calls — on a prepared MOT sequence directory, and returns image-coordinate tracks.

Honest scope: only **detect -> track** is wired. Homography and re-ID are disabled (the pipeline is built
with `homography=None, reid=None`), so each track's `court_xy` and `player_id` are null — these are 2D
image-box tracks, NOT top-down "moving dots". Video-clip -> frames (`ingest.extract_frames`) is still a stub,
so the input is a **frames-on-disk MOT sequence** (`<seq>/img1/` + `seqinfo.ini`), not a raw broadcast clip.
`/health` is a liveness check.

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
    sequence_dir: str                 # a MOT sequence directory: <seq>/ with img1/ + seqinfo.ini
    max_frames: int | None = None     # optional cap (keeps a demo call fast)


def _pipeline():
    """Build the shared detect->track pipeline once (same construction as track/run.py)."""
    global _PIPELINE
    if _PIPELINE is None:
        from ..config import load_config
        from ..detect.detector import build_detector
        from ..pipeline import Pipeline
        from ..track.tracker import build_tracker

        cfg = load_config(os.environ.get("HOOPTRACK_CONFIG", "configs/v0.yaml"))
        _PIPELINE = (cfg, Pipeline(cfg=cfg, detector=build_detector(cfg.detect), tracker=build_tracker(cfg.track)))
    return _PIPELINE


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/track")
def track(req: TrackRequest) -> dict:
    from ..ingest.frames import load_mot_sequence

    seq_dir = Path(req.sequence_dir)
    if not (seq_dir / "img1").is_dir() and not (seq_dir / "seqinfo.ini").is_file():
        raise HTTPException(404, f"no MOT sequence at '{seq_dir}' (need <seq>/img1/ + seqinfo.ini)")
    seq = load_mot_sequence(seq_dir, max_frames=req.max_frames)
    if len(seq) == 0:
        raise HTTPException(400, f"no frames found under '{seq_dir}'")

    cfg, pipe = _pipeline()
    result = pipe.run(seq)   # THE shared pipeline path; homography/re-ID disabled in config -> detect->track only
    return {
        "sequence": seq.name,
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
