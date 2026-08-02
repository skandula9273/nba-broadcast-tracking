"""ONNX-runtime PROVIDER comparison — the fair Apple-Silicon follow-up to the inference-format bench.

`export_bench` found ONNX slower than PyTorch/MPS — but ultralytics ran onnxruntime on its DEFAULT (CPU)
provider. That's not a fair test on Apple hardware, which has a Neural Engine / GPU reachable via the
**CoreMLExecutionProvider**. This isolates the runtime: it times the RAW forward pass (no pre/post-processing)
on an identical fixed input through torch-MPS, onnxruntime-CPU, and onnxruntime-CoreML, so the only variable is
the execution provider. The question it answers honestly: does routing ONNX through Apple's CoreML EP beat MPS
PyTorch, overturning the format bench's negative — or confirm that MPS PyTorch is already the best on-device?
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return round(1000 * (time.perf_counter() - t0) / iters, 3)   # ms per forward pass


def _bench_ort(onnx_path: str, provider: str, x: np.ndarray, iters: int, warmup: int) -> dict | None:
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(onnx_path, providers=[provider])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if provider not in sess.get_providers():                     # onnxruntime silently falls back if unusable
        return {"error": f"provider {provider} not active (got {sess.get_providers()})"}
    name = sess.get_inputs()[0].name
    return {"ms_per_forward": _time(lambda: sess.run(None, {name: x}), iters, warmup),
            "active_providers": sess.get_providers()}


def _bench_torch_mps(weights: str, imgsz: int, x: np.ndarray, iters: int, warmup: int) -> dict:
    import torch
    from ultralytics import YOLO

    if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
        return {"error": "MPS not available"}
    model = YOLO(weights).model.to("mps").eval()
    xt = torch.from_numpy(x).to("mps")
    with torch.no_grad():
        return {"ms_per_forward": _time(lambda: model(xt), iters, warmup), "device": "mps"}


def run(args) -> dict:
    x = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    results = {
        "torch_mps": _bench_torch_mps(args.weights, args.imgsz, x, args.iters, args.warmup),
        "onnx_cpu": _bench_ort(args.onnx, "CPUExecutionProvider", x, args.iters, args.warmup),
        "onnx_coreml": _bench_ort(args.onnx, "CoreMLExecutionProvider", x, args.iters, args.warmup),
    }
    timed = {k: v["ms_per_forward"] for k, v in results.items() if "ms_per_forward" in v}
    fastest = min(timed, key=timed.get) if timed else None
    base = timed.get("torch_mps")
    for v in results.values():
        if "ms_per_forward" in v and base:
            v["speedup_vs_torch_mps"] = round(base / v["ms_per_forward"], 2)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "v2-onnx-provider-comparison",
        "stage": "raw forward-pass latency by execution provider (torch-MPS vs onnxruntime CPU vs CoreML EP)",
        "detector": {"weights": args.weights, "onnx": args.onnx, "imgsz": args.imgsz},
        "bench": {"iters": args.iters, "warmup": args.warmup, "input": list(x.shape),
                  "note": "RAW forward pass only (no pre/post-processing) -> isolates the runtime/provider; "
                  "NOT comparable to export_bench's full-predict ms (which includes NMS/postproc)"},
        "results": results,
        "fastest": fastest,
        "verdict": ("CoreML EP is fastest -> the format-bench negative is overturned for the forward pass"
                    if fastest == "onnx_coreml" else
                    "torch-MPS is fastest -> MPS PyTorch is already best on-device (format-bench negative holds)"
                    if fastest == "torch_mps" else f"fastest={fastest}"),
        "provenance": {"versions": {p: _ver(p) for p in ("onnxruntime", "coremltools", "torch")},
                       "platform": platform.platform()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX-runtime provider comparison (CPU vs CoreML vs torch-MPS)")
    ap.add_argument("--weights", default="weights/finetuned/best.pt")
    ap.add_argument("--onnx", default="weights/finetuned/best.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    if not Path(args.onnx).exists():
        raise SystemExit(f"{args.onnx} not found — run `make detect-export-bench` first to export the ONNX model.")
    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (Path(args.out_dir) / f"onnx_providers_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote onnx_providers_{stamp}.json  (raw forward pass, imgsz {args.imgsz})")
    for name, r in report["results"].items():
        ms = f"{r['ms_per_forward']} ms  ({r.get('speedup_vs_torch_mps', '-')}x)" if "ms_per_forward" in r \
            else r.get("error", "?")
        print(f"  {name:<12} {ms}")
    print(f"fastest: {report['fastest']}  -> {report['verdict']}")


if __name__ == "__main__":
    main()
