"""CLI: run detection standalone over the prepared sequences and report counts/fps.

A light sanity entry for `make detect` (detection-only, no tracker). Full detection mAP needs GT box
matching and is added as its own measured step; this just confirms the detector runs and how many
`person` boxes it emits per sequence.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..config import load_config
from ..detect.detector import build_detector
from ..ingest.frames import load_mot_sequence


def main() -> None:
    ap = argparse.ArgumentParser(description="Run detection (standalone)")
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    gt_set = f"{cfg.eval.benchmark}-{cfg.eval.eval_split}"
    seqmap = Path(cfg.eval.data_dir) / "trackeval" / "gt" / "seqmaps" / f"{gt_set}.txt"
    seqs = [s for s in seqmap.read_text().split() if s and s != "name"]
    if cfg.eval.max_sequences is not None:
        seqs = seqs[: cfg.eval.max_sequences]

    detector = build_detector(cfg.detect)
    split_dir = Path(cfg.eval.data_dir) / cfg.eval.mot_split
    for seq in seqs:
        sequence = load_mot_sequence(split_dir / seq, max_frames=cfg.eval.max_frames)
        t0 = time.time()
        dets = detector.detect(sequence)
        dt = time.time() - t0
        n = len(sequence)
        print(f"{seq}: {len(dets)} person boxes over {n} frames "
              f"({len(dets)/n:.1f}/frame, {n/dt:.1f} fps)")


if __name__ == "__main__":
    main()
