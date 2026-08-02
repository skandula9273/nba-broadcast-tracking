"""CLI: detection mAP eval on the fine-tuned YOLO weights -> a committed timestamped artifact. (increment-04)

The 0.987 mAP@50 existed only as a training-run console figure (every `detection_mAP` in the tracking eval
JSONs is null). This archives it: run Ultralytics `YOLO.val()` on the basketball-val YOLO dataset and write a
real `eval_results/detection_*.json`. It does NOT fine-tune — it validates the existing `weights/finetuned/
best.pt` against the val labels that model was selected on.

Two facts recorded IN the artifact so they travel with the number:
  - **Single class 'athlete'** (`finetune.py names: {0: 'athlete'}`) — the BALL is NOT detected. (The
    design doc's "player/ball detection" was wrong.)
  - **best.pt was model-SELECTED on this same basketball-val split** — the 15 sequences HOTA is scored on —
    so the number is mildly optimistic; a held-out SportsMOT *test* number (GT withheld behind Codalab)
    would be the rigorous version.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..config import load_config

INFERENCE_IMGSZ = 1280  # the deployed tracking pipeline infers at 1280; val here matches the fine-tune imgsz


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def run(weights: str, dataset_yaml: str, imgsz: int, device: str, seed: int = 13) -> dict:
    from ultralytics import YOLO

    n_val_images = len(list((Path(dataset_yaml).parent / "images" / "val").glob("*")))
    model = YOLO(weights)
    project = str(Path(dataset_yaml).parent / "_val_runs")
    res = model.val(
        data=dataset_yaml, imgsz=imgsz, split="val", device=device,
        verbose=False, plots=False, save_json=False, project=project, name="detection_eval", exist_ok=True,
    )
    b = res.box
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "04-detector-finetune",
        "stage": "detection mAP — fine-tuned yolov8m, single class 'athlete'",
        "dataset": {
            "name": "SportsMOT basketball-val (YOLO labels from MOT gt, single class 'athlete')",
            "yaml": dataset_yaml, "split": "val", "n_images": n_val_images, "classes": ["athlete"],
        },
        "results": {
            "detection_mAP50": round(float(b.map50), 4),
            "detection_mAP50_95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4),
            "recall": round(float(b.mr), 4),
        },
        "model": {
            "weights": weights, "arch": "yolov8m (fine-tuned from COCO)",
            "classes": {"0": "athlete"}, "ball_detected": False, "imgsz_val": imgsz,
            "imgsz_inference": INFERENCE_IMGSZ,
        },
        "caveats": [
            "Single class 'athlete' — the BALL is NOT detected (finetune.py names={0:'athlete'}); the design "
            "doc's 'player/ball detection' label was wrong.",
            "OPTIMISTIC: best.pt was model-SELECTED by this same basketball-val mAP — the 15 sequences HOTA "
            "is scored on. The weights only ever saw basketball-train (no frame leakage), but a held-out "
            "SportsMOT test number (GT withheld behind Codalab) is the rigorous version.",
            f"Validated at imgsz {imgsz} (the fine-tune imgsz that produced the reported figure); the "
            f"deployed tracking pipeline infers at imgsz {INFERENCE_IMGSZ} — a documented train/infer mismatch.",
            "Val labels are subsampled (every 5th frame; consecutive 25fps frames are near-duplicates), "
            "matching how the model was trained/selected.",
        ],
        "provenance": {
            "seed": seed, "device": device,
            "versions": {p: _ver(p) for p in ("ultralytics", "torch", "numpy")},
            "platform": platform.platform(),
        },
        "notes": "Archives the increment-04 detection figure as a real artifact (it was previously a console "
        "figure only; detection_mAP is null in every tracking eval JSON). Does not fine-tune — validates the "
        "committed weights against the val labels they were selected on.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Detection mAP eval on fine-tuned YOLO weights (increment-04)")
    ap.add_argument("--config", default="configs/v0_finetuned.yaml")
    ap.add_argument("--dataset", default="data/sportsmot/_yolo_ft/sportsmot_basketball.yaml")
    ap.add_argument("--imgsz", type=int, default=640)  # the fine-tune imgsz that produced the reported mAP
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()
    cfg = load_config(args.config)

    report = run(cfg.detect.weights, args.dataset, args.imgsz, cfg.detect.device, seed=cfg.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"detection_{stamp}.json").write_text(json.dumps(report, indent=2))
    r = report["results"]
    print(f"Wrote detection_{stamp}.json  | n_images={report['dataset']['n_images']} (class: athlete; ball NOT detected)")
    print(f"  mAP@50={r['detection_mAP50']}  mAP@50-95={r['detection_mAP50_95']}  P={r['precision']}  R={r['recall']}")


if __name__ == "__main__":
    main()
