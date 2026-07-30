"""Inference-format optimization — the 'serving + inference optimization' V2 line, measured.

The detector Pareto (detect/pareto.py) swept the resolution and model-size knobs. This adds the **deployment
format** axis: export the fine-tuned detector to ONNX and CoreML and measure per-frame latency + mAP for each
against the PyTorch/MPS baseline, so 'ONNX/TensorRT/CoreML' becomes a number, not a checklist item. Timed at
**batch 1** (single-frame latency — the deployment-relevant metric; exported models are batch-1 by default), so
this is apples-to-apples and NOT comparable to pareto.py's batched-throughput fps. TensorRT is NVIDIA-only and
not available in this (Apple/MPS) environment — flagged, not faked.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..config import load_config
from ..ingest.frames import load_mot_sequence

BENCH_SEQ = Path("data/sportsmot/val/v_00HRwkvvjtQ_c007")


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _export(weights: str, fmt: str, imgsz: int) -> str | None:
    """Export `weights` to `fmt` (onnx|coreml) at imgsz; return the path, or None if export fails. Ultralytics
    returns a str for ONNX but a Path for CoreML — coerce to str so downstream string ops are consistent."""
    from ultralytics import YOLO
    try:
        return str(YOLO(weights).export(format=fmt, imgsz=imgsz, verbose=False))
    except Exception as e:
        print(f"  [{fmt}] export failed: {type(e).__name__}: {e}", flush=True)
        return None


def _size_mb(model_path: str) -> float:
    p = Path(model_path)
    if p.is_file():
        return round(p.stat().st_size / 1e6, 1)
    return round(sum(f.stat().st_size for f in p.rglob("*")) / 1e6, 1)   # .mlpackage is a directory


def _measure(model_path: str, imgsz: int, device: str, dataset: str, frames: int, warmup: int) -> dict:
    """Per-frame latency (batch 1) + full mAP@50 + size for one model artifact. Full mAP for EVERY format is the
    authoritative equivalence proof (a lossy export shows up as a mAP drop); a per-frame box spot-check looked
    tempting but postprocessing nuances make exported detection counts differ slightly even when mAP is
    identical, so it produced misleading mismatches — the val is the honest number."""
    from ultralytics import YOLO

    model_path = str(model_path)
    model = YOLO(model_path)
    is_pt = model_path.endswith(".pt")
    paths = [str(p) for _, p in load_mot_sequence(BENCH_SEQ, max_frames=frames + warmup)]
    kw = {"imgsz": imgsz, "verbose": False}
    if is_pt:
        kw["device"] = device                                # exported artifacts pick their own runtime/provider
    for p in paths[:warmup]:                                  # warm the runtime (graph build / provider init)
        model.predict(p, **kw)
    t0 = time.perf_counter()
    for p in paths[warmup:]:
        model.predict(p, **kw)
    n = len(paths) - warmup
    dt = time.perf_counter() - t0
    val = model.val(data=dataset, imgsz=imgsz, split="val", device=(device if is_pt else None),
                    verbose=False, plots=False, save_json=False)
    return {"ms_per_frame": round(1000 * dt / n, 2), "fps_batch1": round(n / dt, 2),
            "mAP50": round(float(val.box.map50), 4), "size_mb": _size_mb(model_path)}


def run(args) -> dict:
    cfg = load_config(args.config)
    weights = cfg.detect.weights
    device = cfg.detect.device
    imgsz = args.imgsz

    formats = {"pytorch": weights}
    for fmt in args.formats:
        path = _export(weights, fmt, imgsz)
        if path:
            formats[fmt] = path

    results = {}
    for name, path in formats.items():
        print(f"  measuring {name} ({path}) ...", flush=True)
        results[name] = {"artifact": path, **_measure(path, imgsz, device, args.dataset, args.frames, args.warmup)}

    base = results["pytorch"]["ms_per_frame"]
    for name, r in results.items():
        r["speedup_vs_pytorch"] = round(base / r["ms_per_frame"], 2) if r["ms_per_frame"] else None
    fastest = min(results, key=lambda k: results[k]["ms_per_frame"])

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "v2-inference-format-opt",
        "stage": "detector deployment-format latency/accuracy (PyTorch vs ONNX vs CoreML), batch 1",
        "detector": {"weights": weights, "arch": "yolov8m (fine-tuned)", "imgsz": imgsz, "device_pytorch": device},
        "bench": {"sequence": BENCH_SEQ.name, "frames_timed": args.frames, "warmup": args.warmup,
                  "batch": 1, "note": "single-frame latency; NOT comparable to pareto.py batched throughput"},
        "results": results,
        "fastest_format": fastest,
        "tensorrt": "not measured — NVIDIA-only; this environment is Apple/MPS (flagged, not faked)",
        "notes": "Full mAP measured for EVERY format (a lossy export would show as a mAP drop); equal mAP across "
        "formats == lossless export. The real signal is the latency delta: which runtime is fastest for "
        "single-frame serving on this hardware. onnxruntime uses its default (CPU) provider; CoreML the ANE/GPU.",
        "provenance": {"versions": {p: _ver(p) for p in ("ultralytics", "onnxruntime", "coremltools", "torch")},
                       "platform": platform.platform()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Detector inference-format optimization bench (ONNX/CoreML)")
    ap.add_argument("--config", default="configs/v0_finetuned_640.yaml")
    ap.add_argument("--dataset", default="data/sportsmot/_yolo_ft/sportsmot_basketball.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--formats", nargs="+", default=["onnx", "coreml"])
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (Path(args.out_dir) / f"inference_format_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote inference_format_{stamp}.json")
    print(f"{'format':<10}{'ms/frame':>10}{'fps@1':>8}{'speedup':>9}{'size_MB':>9}{'mAP50':>9}")
    for name, r in report["results"].items():
        print(f"{name:<10}{r['ms_per_frame']:>10}{r['fps_batch1']:>8}{r['speedup_vs_pytorch']:>9}"
              f"{r['size_mb']:>9}{r['mAP50']:>9}")
    print(f"fastest: {report['fastest_format']}")


if __name__ == "__main__":
    main()
