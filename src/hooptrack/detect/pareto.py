"""Detector accuracy-latency Pareto — the real V2 first step (serving optimization frontier).

`serve-bench` gave ONE operating point (9.9 fps). This turns it into a measured **frontier**: sweep the deployed
detector's inference **image size** — a zero-retraining knob you can turn live — and measure (mAP@50, detection
fps, params) at each, so the accuracy-latency tradeoff is a curve, not a lever named in prose. The fine-tuned
yolov8m is held fixed; only `imgsz` varies (one variable). The model-SIZE axis (a fine-tuned yolov8n) is a
different knob that needs its own matched fine-tune — the documented next point, not faked here.
"""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..config import DetectConfig, load_config
from ..detect.detector import build_detector
from ..ingest.frames import load_mot_sequence

BENCH_SEQ = Path("data/sportsmot/val/v_00HRwkvvjtQ_c007")


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _map50(weights: str, dataset_yaml: str, imgsz: int, device: str, tmp: Path) -> float:
    from ultralytics import YOLO

    res = YOLO(weights).val(data=dataset_yaml, imgsz=imgsz, split="val", device=device, verbose=False,
                            plots=False, save_json=False, project=str(tmp), name=f"map_{imgsz}", exist_ok=True)
    return round(float(res.box.map50), 4)


def _fps(weights: str, imgsz: int, device: str, n_frames: int, warmup: int) -> tuple[float, float]:
    det = build_detector(DetectConfig(model="yolo", weights=weights, imgsz=imgsz, device=device, batch=16))
    det.detect(load_mot_sequence(BENCH_SEQ, max_frames=warmup))          # warm model load + MPS compile
    seq = load_mot_sequence(BENCH_SEQ, max_frames=n_frames)
    t0 = time.perf_counter()
    det.detect(seq)
    dt = time.perf_counter() - t0
    return round(len(seq) / dt, 2), round(1000 * dt / len(seq), 2)       # (fps, ms/frame)


def _frontier(points: list[dict]) -> None:
    """Mark each point on_frontier=True unless another point dominates it (>= mAP AND >= fps, strictly better once)."""
    for p in points:
        dominated = any(q is not p and q["mAP50"] >= p["mAP50"] and q["fps"] >= p["fps"]
                        and (q["mAP50"] > p["mAP50"] or q["fps"] > p["fps"]) for q in points)
        p["on_frontier"] = not dominated


def run(args) -> dict:
    cfg = load_config(args.config)
    weights = cfg.detect.weights
    n_params = None
    try:
        from ultralytics import YOLO
        n_params = sum(p.numel() for p in YOLO(weights).model.parameters())
    except Exception:
        pass
    weight_mb = round(Path(weights).stat().st_size / 1e6, 1) if Path(weights).exists() else None

    tmp = Path(tempfile.mkdtemp(prefix="pareto_"))
    points = []
    for imgsz in args.imgsz:
        fps, ms = _fps(weights, imgsz, cfg.detect.device, args.frames, args.warmup)
        points.append({"name": f"yolov8m_ft@{imgsz}", "imgsz": imgsz,
                       "mAP50": _map50(weights, args.dataset, imgsz, cfg.detect.device, tmp),
                       "fps": fps, "ms_per_frame": ms})
    _frontier(points)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "v2-detector-pareto",
        "stage": "detector accuracy-latency Pareto over inference imgsz (V2 serving-optimization first step)",
        "detector": {"weights": weights, "arch": "yolov8m (fine-tuned, class 'athlete')",
                     "n_params": n_params, "weight_mb": weight_mb, "device": cfg.detect.device,
                     "note": "one variable = inference imgsz; a zero-retraining live knob"},
        "bench": {"sequence": BENCH_SEQ.name, "frames_timed": args.frames, "warmup": args.warmup,
                  "mAP_split": "SportsMOT basketball-val (YOLO labels, class athlete)"},
        "points": points,
        "frontier": [p["name"] for p in points if p["on_frontier"]],
        "notes": "Accuracy-latency frontier via the DEPLOYABLE resolution knob (no retraining to move along it). "
        "Deployed default is imgsz 1280. The model-SIZE axis (a fine-tuned yolov8n) is a separate knob needing "
        "its own matched fine-tune — the documented next point, not faked here. mAP at imgsz != 640 measures the "
        "inference-resolution effect (the model was fine-tuned at 640).",
        "provenance": {"versions": {p: _ver(p) for p in ("ultralytics", "torch")}, "platform": platform.platform()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Detector accuracy-latency Pareto over inference imgsz")
    ap.add_argument("--config", default="configs/v0_finetuned.yaml")
    ap.add_argument("--dataset", default="data/sportsmot/_yolo_ft/sportsmot_basketball.yaml")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280, 960, 640, 480])
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (Path(args.out_dir) / f"detector_pareto_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote detector_pareto_{stamp}.json  ({report['detector']['weight_mb']} MB, {report['detector']['n_params']} params)")
    print(f"{'point':<18}{'mAP50':>8}{'fps':>8}{'ms/frame':>10}{'  frontier':>10}")
    for p in report["points"]:
        print(f"{p['name']:<18}{p['mAP50']:>8}{p['fps']:>8}{p['ms_per_frame']:>10}{'  *' if p['on_frontier'] else '':>10}")


if __name__ == "__main__":
    main()
