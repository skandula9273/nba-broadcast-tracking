"""CLI: homography reprojection-error eval on DeepSportradar calibration GT. (increment-05)

Honest scope. The homography stage has two parts: (1) a *solver* (correspondences -> H via DLT/RANSAC,
built + tested in court.py) and (2) a *registration front-end* (detect court keypoints in an unlabelled
image to supply correspondences). A robust classical front-end is a substantial, finicky CV project, and
homography is an enabler (don't gold-plate) — so this eval measures what we can honestly measure now:

  - **Stage accuracy vs keypoint quality:** simulate a keypoint detector by taking GT court keypoints
    (projected via the GT calibration) with Gaussian pixel noise sigma, solve H, and report the resulting
    court-registration reprojection error. This quantifies exactly how good a front-end must be for a given
    accuracy — the number the *deferred* front-end (classical or KaliCalib) has to hit.
  - **Trivial baseline:** a single global-mean homography for every image (the no-registration floor).

Both run on DeepSportradar GT calibration (728 instants / 15 arenas). Committed to eval_results/.
"""
from __future__ import annotations

import argparse
import glob
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .court import _apply, homography_from_calibration, reprojection_error, solve_homography

COURT_W, COURT_H = 2800.0, 1500.0  # FIBA court, cm (DeepSportradar court frame)
NOISE_SIGMAS = [0.0, 1.0, 3.0, 5.0, 10.0]


def _court_grid(nx: int = 15, ny: int = 8) -> np.ndarray:
    xs = np.linspace(0, COURT_W, nx)
    ys = np.linspace(0, COURT_H, ny)
    return np.array([[x, y] for x in xs for y in ys], float)


def _load_calibs(data_dir: Path) -> list[tuple[str, np.ndarray]]:
    out = []
    for f in sorted(glob.glob(str(data_dir / "**" / "*.json"), recursive=True)):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "calibration" in d and "KK" in d.get("calibration", {}):
            arena = Path(f).relative_to(data_dir).parts[0]
            out.append((arena, homography_from_calibration(d["calibration"])))
    return out


def _visible(H: np.ndarray, grid: np.ndarray, w: int = 1624, h: int = 1234) -> np.ndarray:
    uv = _apply(H, grid)
    m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return grid[m]


def run(data_dir: str, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    grid = _court_grid()
    calibs = _load_calibs(Path(data_dir))
    if not calibs:
        raise RuntimeError(f"no DeepSportradar calibration JSONs under {data_dir}")

    H_mean = np.mean([H / np.linalg.norm(H) for _, H in calibs], axis=0)  # trivial global-mean baseline

    noise_errs: dict[float, list[float]] = {s: [] for s in NOISE_SIGMAS}
    trivial_errs: list[float] = []
    used = 0
    for _, H_gt in calibs:
        pts = _visible(H_gt, grid)
        if len(pts) < 6:  # need enough visible correspondences to solve
            continue
        used += 1
        trivial_errs.append(reprojection_error(H_mean, H_gt, pts))
        img = _apply(H_gt, pts)
        for s in NOISE_SIGMAS:
            noisy = img + rng.normal(0, s, img.shape)
            H_hat, _ = solve_homography(pts, noisy)
            noise_errs[s].append(reprojection_error(H_hat, H_gt, pts) if H_hat is not None else np.nan)

    def stats(v):
        a = np.array(v, float)
        a = a[~np.isnan(a)]
        return {"mean_px": round(float(a.mean()), 3), "median_px": round(float(np.median(a)), 3)}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "05-homography",
        "stage": "homography (court registration)",
        "dataset": {"name": "DeepSportradar basketball-instants (camera calibration)",
                    "n_instants": len(calibs), "n_evaluated": used,
                    "arenas": sorted({a for a, _ in calibs})},
        "metric": "court-registration reprojection error (px), court->image over a visible court grid",
        "results": {
            "solver_vs_keypoint_noise_sigma_px": {str(s): stats(noise_errs[s]) for s in NOISE_SIGMAS},
            "trivial_baseline_global_mean_H": stats(trivial_errs),
        },
        "notes": "Solver (DLT+RANSAC) + reprojection metric are real and validated. The automatic court-"
        "keypoint front-end is the deferred component (classical detector or the learned KaliCalib); this "
        "eval quantifies the reprojection error it must beat and the accuracy it yields per keypoint sigma. "
        "GT = DeepSportradar per-image K/R/T (H = K[r1 r2 T]). Enabler stage — not gold-plated.",
        "provenance": {"seed": seed, "court_cm": [COURT_W, COURT_H], "grid": "15x8",
                       "versions": {p: (version(p) if _has(p) else None) for p in ("numpy", "opencv-python")},
                       "platform": platform.platform()},
    }


def _has(pkg: str) -> bool:
    try:
        version(pkg)
        return True
    except PackageNotFoundError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Homography reprojection-error eval (DeepSportradar)")
    ap.add_argument("--data-dir", default="data/deepsport")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()
    report = run(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"homography_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    r = report["results"]
    print(f"Wrote {path}")
    print(f"  trivial (global-mean H): {r['trivial_baseline_global_mean_H']['median_px']} px median")
    print(f"  solver @ sigma=3px keypoints: {r['solver_vs_keypoint_noise_sigma_px']['3.0']['median_px']} px median")


if __name__ == "__main__":
    main()
