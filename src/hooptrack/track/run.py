"""CLI: run tracking -> MOT-format outputs (for TrackEval HOTA) + a run_stats.json provenance file.

Drives the ONE shared pipeline (detect -> track) per sequence, so `make track` produces exactly the
tracks the eval harness scores and the deployed API would produce. Outputs land in the TrackEval trackers
tree that `ingest/fetch.py` created:
    data/sportsmot/trackeval/trackers/<benchmark>-<eval_split>/<method>/data/<seq>.txt
    data/sportsmot/trackeval/trackers/<benchmark>-<eval_split>/<method>/run_stats.json
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from ..detect.detector import DEFAULT_WEIGHTS, build_detector
from ..eval.trackeval_adapter import write_mot
from ..ingest.frames import load_mot_sequence
from ..pipeline import Pipeline
from ..track.tracker import build_tracker


def _versions() -> dict:
    out = {}
    for pkg in ("torch", "torchvision", "ultralytics", "boxmot", "trackeval", "numpy", "opencv-python"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def tracker_output_dir(cfg: Config) -> Path:
    gt_set = f"{cfg.eval.benchmark}-{cfg.eval.eval_split}"
    return Path(cfg.eval.data_dir) / "trackeval" / "trackers" / gt_set / cfg.track.method


def _sequences(cfg: Config) -> list[str]:
    """Read the seqmap the data step wrote (authoritative: exactly the GT we prepared). Skips the header."""
    gt_set = f"{cfg.eval.benchmark}-{cfg.eval.eval_split}"
    seqmap = Path(cfg.eval.data_dir) / "trackeval" / "gt" / "seqmaps" / f"{gt_set}.txt"
    seqs = [s for s in seqmap.read_text().split() if s and s != "name"]
    if cfg.eval.max_sequences is not None:
        seqs = seqs[: cfg.eval.max_sequences]
    return seqs


def run(cfg: Config) -> dict:
    _seed_everything(cfg.seed)
    split_dir = Path(cfg.eval.data_dir) / cfg.eval.mot_split
    out_dir = tracker_output_dir(cfg)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(cfg.detect)
    tracker = build_tracker(cfg.track)
    pipe = Pipeline(cfg=cfg, detector=detector, tracker=tracker)

    seqs = _sequences(cfg)
    seq_stats = []
    for i, seq in enumerate(seqs, 1):
        sequence = load_mot_sequence(split_dir / seq, max_frames=cfg.eval.max_frames)
        t0 = time.time()
        result = pipe.run(sequence)
        dt = time.time() - t0
        n_lines = write_mot(result.tracks, data_dir / f"{seq}.txt")
        n_frames = len(sequence)
        stats = {
            "name": seq,
            "frames": n_frames,
            "track_rows": n_lines,
            "n_ids": result.meta.get("n_ids"),
            "seconds": round(dt, 2),
            "fps": round(n_frames / dt, 2) if dt > 0 else None,
        }
        seq_stats.append(stats)
        print(f"[{i}/{len(seqs)}] {seq}: {n_frames} frames, {stats['n_ids']} ids, "
              f"{n_lines} rows, {stats['fps']} fps")

    total_frames = sum(s["frames"] for s in seq_stats)
    total_seconds = sum(s["seconds"] for s in seq_stats)
    run_stats = {
        "gt_set": f"{cfg.eval.benchmark}-{cfg.eval.eval_split}",
        "tracker": cfg.track.method,
        "seed": cfg.seed,
        "detector": {
            "model": cfg.detect.model,
            "weights": cfg.detect.weights or DEFAULT_WEIGHTS,
            "conf": cfg.detect.conf,
            "iou": cfg.detect.iou,
            "imgsz": cfg.detect.imgsz,
            "device": cfg.detect.device,
            "person_class": cfg.detect.person_class,
            "fine_tuned": False,
            "note": "COCO-pretrained person class as athlete proxy — not fine-tuned on SportsMOT.",
        },
        "tracker_params": {
            "min_conf": cfg.track.min_conf,
            "track_thresh": cfg.track.track_thresh,
            "match_thresh": cfg.track.match_thresh,
            "track_buffer": cfg.track.track_buffer,
        },
        "caps": {"max_sequences": cfg.eval.max_sequences, "max_frames": cfg.eval.max_frames},
        "versions": _versions(),
        "device_info": {"platform": platform.platform(), "machine": platform.machine()},
        "sequences": seq_stats,
        "totals": {
            "frames": total_frames,
            "seconds": round(total_seconds, 2),
            "fps": round(total_frames / total_seconds, 2) if total_seconds > 0 else None,
        },
    }
    (out_dir / "run_stats.json").write_text(json.dumps(run_stats, indent=2))
    print(f"\nWrote {len(seqs)} tracker files + run_stats.json -> {out_dir}")
    print(f"Totals: {total_frames} frames in {total_seconds:.1f}s ({run_stats['totals']['fps']} fps)")
    return run_stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Run tracking -> MOT outputs for TrackEval")
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
