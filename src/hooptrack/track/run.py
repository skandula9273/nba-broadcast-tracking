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

from ..config import Config, DetectConfig, load_config
from ..detect.detector import DEFAULT_WEIGHTS, CachingDetector, build_detector, detection_cache_dir
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


def tracker_name(cfg: Config) -> str:
    """Output/eval subdir name — distinguishes ablation variants (e.g. botsort_noreid) from track.method."""
    return cfg.eval.tracker_name or cfg.track.method


def tracker_output_dir(cfg: Config) -> Path:
    gt_set = f"{cfg.eval.benchmark}-{cfg.eval.eval_split}"
    return Path(cfg.eval.data_dir) / "trackeval" / "trackers" / gt_set / tracker_name(cfg)


def _sequences(cfg: Config) -> list[str]:
    """Read the seqmap the data step wrote (authoritative: exactly the GT we prepared). Skips the header."""
    gt_set = f"{cfg.eval.benchmark}-{cfg.eval.eval_split}"
    seqmap = Path(cfg.eval.data_dir) / "trackeval" / "gt" / "seqmaps" / f"{gt_set}.txt"
    seqs = [s for s in seqmap.read_text().split() if s and s != "name"]
    if cfg.eval.max_sequences is not None:
        seqs = seqs[: cfg.eval.max_sequences]
    return seqs


def _detector_provenance(dc: DetectConfig) -> dict:
    """Detector provenance DERIVED from the actual detect config — never hand-set, so it cannot go stale.

    `fine_tuned` is true iff the resolved weights are not the COCO-pretrained default (yolov8m.pt); the note
    is generated from that same fact. When `detect.weights` is null the default resolves to COCO; any other
    path is a non-default (fine-tuned) checkpoint. This is the single source of truth for the detector regime
    (the eval caveats read it back), so config and provenance can't disagree the way they did pre-2026-07-28.
    """
    weights = dc.weights or DEFAULT_WEIGHTS
    coco_default = weights == DEFAULT_WEIGHTS
    note = (
        f"COCO-pretrained {DEFAULT_WEIGHTS}, person class as athlete proxy — NOT fine-tuned on SportsMOT; "
        "referees/bench/crowd it calls 'person' are false positives vs athlete-only GT (depresses DetA)."
        if coco_default else
        f"non-default (fine-tuned) weights '{weights}' — an athlete-class detector, not the COCO person proxy."
    )
    return {
        "model": dc.model,
        "weights": weights,
        "coco_default_weights": coco_default,
        "fine_tuned": not coco_default,
        "conf": dc.conf,
        "iou": dc.iou,
        "imgsz": dc.imgsz,
        "device": dc.device,
        "person_class": dc.person_class,
        "note": note,
    }


def run(cfg: Config) -> dict:
    _seed_everything(cfg.seed)
    split_dir = Path(cfg.eval.data_dir) / cfg.eval.mot_split
    out_dir = tracker_output_dir(cfg)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(cfg.detect)
    if cfg.eval.cache_detections:
        detector = CachingDetector(detector, detection_cache_dir(cfg))
    tracker = build_tracker(cfg.track)
    from ..homography.court import build_homography

    pipe = Pipeline(cfg=cfg, detector=detector, tracker=tracker, homography=build_homography(cfg))

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
        "tracker": tracker_name(cfg),
        "cache_detections": cfg.eval.cache_detections,
        "seed": cfg.seed,
        "detector": _detector_provenance(cfg.detect),
        "tracker_params": tracker.params(),
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
