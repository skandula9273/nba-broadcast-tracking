"""Ball-detection coverage harness — reproducible, committed measurement of COCO 'sports ball' on broadcast.

Coverage = fraction of sampled frames with a confident ball detection (there's no ball GT in SportsMOT, so this
is coverage, not accuracy — same honest framing as jersey OCR). Reports coverage at a few confidence thresholds
so the operating point is a choice, not a hidden default. `make ball-eval`.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..ingest.frames import load_mot_sequence
from .ball import BallDetector

FRAMES_ROOT = Path("data/sportsmot/val")
GT_ROOT = Path("data/sportsmot/trackeval/gt/SportsMOT-basketball-val")


def _subsample(seq, every: int):
    return [(fidx, path) for i, (fidx, path) in enumerate(seq) if i % every == 0]


def run(args) -> dict:
    seqs = sorted(p.name for p in GT_ROOT.iterdir() if (p / "gt" / "gt.txt").is_file())
    if args.limit_seqs:
        seqs = seqs[: args.limit_seqs]
    det = BallDetector(device=args.device, conf=args.min_conf, imgsz=args.imgsz)

    thresholds = [0.25, 0.35, 0.5]
    n_frames = 0
    confs: list[float] = []
    per_seq = {}
    for s in seqs:
        frames = _subsample(load_mot_sequence(FRAMES_ROOT / s), args.every)
        balls = det.detect(frames)
        c = [conf for _, _, conf in balls.values()]
        n_frames += len(frames)
        confs.extend(c)
        per_seq[s] = {"n_frames": len(frames), "coverage@0.25": round(len(c) / max(1, len(frames)), 3)}

    confs_arr = np.array(confs) if confs else np.zeros(0)
    coverage = {f"coverage@{t}": round(float((confs_arr >= t).sum()) / max(1, n_frames), 3) for t in thresholds}
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "ball-detection-coverage",
        "stage": "COCO 'sports ball' (class 32) coverage on SportsMOT broadcast — no training",
        "basis": "coverage = fraction of sampled frames with a ball >= threshold; NOT accuracy (no ball GT); "
                 "the ball is the missing piece for true possessions/shots",
        "dataset": {"source": "SportsMOT basketball-val", "n_seqs": len(seqs), "sampled_every": args.every,
                    "n_frames_sampled": n_frames, "imgsz": args.imgsz},
        "coverage": coverage,
        "conf_distribution": {"mean": round(float(confs_arr.mean()), 3) if len(confs_arr) else None,
                              "p50": round(float(np.median(confs_arr)), 3) if len(confs_arr) else None,
                              "max": round(float(confs_arr.max()), 3) if len(confs_arr) else None},
        "per_seq": per_seq,
        "notes": "COCO yolov8m sports-ball, no fine-tuning. Sparse but real signal -> enables ball-handler "
        "attribution where the ball is visible (analytics.ball_possession); precise shot detection needs a "
        "basketball-specific detector + annotations. Reported as-is.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ball-detection coverage on SportsMOT (COCO sports ball)")
    ap.add_argument("--every", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--limit-seqs", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=0.2, help="detector conf floor (thresholds swept above it)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (Path(args.out_dir) / f"ball_coverage_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote ball_coverage_{stamp}.json | n_frames={report['dataset']['n_frames_sampled']}")
    print(f"coverage: {report['coverage']}")
    print(f"conf: {report['conf_distribution']}")


if __name__ == "__main__":
    main()
