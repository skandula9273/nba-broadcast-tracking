"""Serving latency baseline — the missing operating point for the V2 'serving optimization + Pareto frontier'.

Times the REAL deployed perception path (detect -> track — the same `Pipeline` stages `serve /track` runs) on
real frames, so there's a concrete before-number to optimize against. Reports per-stage total / ms-per-frame /
throughput (fps) on the actual device, after a warmup pass (so model-load + first-call MPS graph compilation
don't pollute the steady-state number). Committed timestamped JSON.

Honest scope: this is **stage-level throughput** (detector batches internally, so there are no per-frame
percentiles here), on `detect -> track` only. Homography/re-ID are off by default in the deployed path; the
re-ID jersey-OCR stage is the known bottleneck (easyocr, ~minutes per clip — see the jersey runs), timed
separately. The retrieval/encode stage is numpy+FAISS (sub-millisecond) and not the serving cost.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

from ..config import load_config
from ..detect.detector import build_detector
from ..ingest.frames import load_mot_sequence
from ..track.tracker import build_tracker


def _ver(pkg: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not-installed"


def _stage(total_s: float, n: int) -> dict:
    return {"total_s": round(total_s, 3), "ms_per_frame": round(1000 * total_s / max(1, n), 2),
            "fps": round(n / total_s, 2) if total_s > 0 else None}


def run(args) -> dict:
    cfg = load_config(args.config)
    seq = load_mot_sequence(Path(args.seq_dir), max_frames=args.frames)
    n = len(seq)
    detector, tracker = build_detector(cfg.detect), build_tracker(cfg.track)

    warm = load_mot_sequence(Path(args.seq_dir), max_frames=args.warmup)   # warm model load + MPS compile
    detector.detect(warm)

    t0 = time.perf_counter()
    dets = detector.detect(seq)
    detect_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    tracks = tracker.track(dets, seq)
    track_s = time.perf_counter() - t1

    stages = {"detect": {**_stage(detect_s, n), "n_dets": len(dets)}, "track": _stage(track_s, n)}
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "serving-latency-baseline",
        "stage": "serving latency baseline (detect -> track, the deployed serve path)",
        "config": {"detector": cfg.detect.model, "weights": cfg.detect.weights or "yolov8m.pt (COCO)",
                   "device": cfg.detect.device, "batch": cfg.detect.batch, "imgsz": cfg.detect.imgsz,
                   "tracker": cfg.track.method},
        "sequence": {"name": seq.name, "n_frames": n, "width": seq.width, "height": seq.height,
                     "warmup_frames": args.warmup},
        "stages": stages,
        "end_to_end": {**_stage(detect_s + track_s, n), "n_tracks": len(tracks)},
        "provenance": {"versions": {p: _ver(p) for p in ("torch", "ultralytics", "boxmot")},
                       "platform": platform.platform()},
        "notes": "Stage-level throughput (detector batches -> no per-frame percentiles). Warmup pass excludes "
        "model load / first-call MPS compile. Homography/re-ID off (deployed default); jersey-OCR re-ID is the "
        "known bottleneck (easyocr, minutes/clip) measured separately. Encode/FAISS is sub-ms, not the cost.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Serving latency baseline for the detect->track path")
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--seq-dir", default="data/sportsmot/val/v_00HRwkvvjtQ_c007")
    ap.add_argument("--frames", type=int, default=200, help="frames to time (steady state)")
    ap.add_argument("--warmup", type=int, default=8, help="frames for the untimed warmup pass")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / f"serving_latency_{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    s, e = report["stages"], report["end_to_end"]
    print(f"wrote {out}  | device={report['config']['device']} n={report['sequence']['n_frames']}")
    print(f"detect: {s['detect']['ms_per_frame']} ms/frame ({s['detect']['fps']} fps)")
    print(f"track:  {s['track']['ms_per_frame']} ms/frame ({s['track']['fps']} fps)")
    print(f"detect->track end to end: {e['ms_per_frame']} ms/frame ({e['fps']} fps)")


if __name__ == "__main__":
    main()
