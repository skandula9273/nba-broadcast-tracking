"""Court-keypoint registration FRONT-END — the deferred piece of the homography stage.

Increment-05 built the *solver* (correspondences -> H) and measured it vs keypoint noise; it deferred the
*front-end* that supplies correspondences from an unlabelled frame. This is that front-end, done
harness-before-modeling (rule #1): define a canonical set of court line-intersections, generate their GT
image locations from DeepSportradar calibration, and measure ANY keypoint detector by the reprojection error
of the homography it yields (reusing inc-05's metric). A trivial floor (predict the global-mean keypoint
locations) sets the bar; a learned KaliCalib-lite detector is the next step. A detector is any callable
`image -> {keypoint_name: (u, v)}`.
"""
from __future__ import annotations

import argparse
import glob
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .court import _apply, homography_from_calibration, reprojection_error, solve_homography

COURT_W, COURT_H = 2800.0, 1500.0   # DeepSportradar court frame (cm), from inc-05

# Canonical keypoints: unambiguous court line-intersections (court frame, cm). Corners + midline + center are
# certain from any court diagram (no guessing lane geometry); >=4 visible suffice to solve H.
CANONICAL: dict[str, tuple[float, float]] = {
    "corner_bl": (0.0, 0.0),           "corner_br": (COURT_W, 0.0),
    "corner_tr": (COURT_W, COURT_H),   "corner_tl": (0.0, COURT_H),
    "mid_bottom": (COURT_W / 2, 0.0),  "mid_top": (COURT_W / 2, COURT_H),
    "center": (COURT_W / 2, COURT_H / 2),
}


def gt_keypoints(h_court2img: np.ndarray, w: int, h: int) -> dict[str, tuple[float, float]]:
    """Project the canonical court keypoints into the image via GT H; keep those inside the frame."""
    out = {}
    for name, court_xy in CANONICAL.items():
        u, v = _apply(h_court2img, [court_xy])[0]
        if 0 <= u < w and 0 <= v < h:
            out[name] = (float(u), float(v))
    return out


def solve_from_keypoints(pred: dict[str, tuple[float, float]]) -> np.ndarray | None:
    """Named image keypoints + their known court coords -> H_court2img (needs >=4)."""
    names = [n for n in pred if n in CANONICAL]
    if len(names) < 4:
        return None
    H, _ = solve_homography([CANONICAL[n] for n in names], [pred[n] for n in names])
    return H


def _court_grid(nx: int = 15, ny: int = 8) -> np.ndarray:
    xs = np.linspace(0, COURT_W, nx)
    ys = np.linspace(0, COURT_H, ny)
    return np.array([[x, y] for x in xs for y in ys], float)


def _visible(H: np.ndarray, grid: np.ndarray, w: int, h: int) -> np.ndarray:
    uv = _apply(H, grid)
    m = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return grid[m]


def load_instants(data_dir: str | Path) -> list[dict]:
    """Each DeepSportradar instant with a calibration: {H_gt, w, h, image, gt_keypoints}."""
    out = []
    for f in sorted(glob.glob(str(Path(data_dir) / "**" / "*.json"), recursive=True)):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        calib = d.get("calibration")
        if not (isinstance(calib, dict) and "KK" in calib):
            continue
        H = homography_from_calibration(calib)
        w = int(calib.get("img_width", 1624))
        h = int(calib.get("img_height", 1234))
        img = next(iter(sorted(Path(f).parent.glob(Path(f).stem + "*.png"))), None)
        out.append({"H_gt": H, "w": w, "h": h, "image": str(img) if img else None,
                    "gt_keypoints": gt_keypoints(H, w, h)})
    return out


def evaluate(detector, instants: list[dict]) -> list[float]:
    """Reprojection error (px, over a visible court grid) of the H each detector prediction yields vs GT.
    `detector=None` is the trivial floor: predict the global-mean GT location of each visible keypoint."""
    grid = _court_grid()
    mean_kp = None
    if detector is None:
        acc: dict[str, list] = defaultdict(list)
        for it in instants:
            for n, uv in it["gt_keypoints"].items():
                acc[n].append(uv)
        mean_kp = {n: tuple(np.mean(v, axis=0)) for n, v in acc.items()}

    errs = []
    for it in instants:
        if detector is None:
            pred = {n: mean_kp[n] for n in it["gt_keypoints"] if n in mean_kp}
        else:
            import cv2

            img = cv2.imread(it["image"]) if it["image"] else None
            pred = detector(img) if img is not None else {}
        H_hat = solve_from_keypoints(pred)
        pts = _visible(it["H_gt"], grid, it["w"], it["h"])
        if H_hat is None or len(pts) < 6:
            continue
        errs.append(reprojection_error(H_hat, it["H_gt"], pts))
    return errs


def _stats(errs: list[float]) -> dict:
    a = np.asarray(errs, float)
    return {"median_px": round(float(np.median(a)), 2), "mean_px": round(float(a.mean()), 2), "n": len(a)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Court-keypoint front-end harness — trivial floor (no detector)")
    ap.add_argument("--data-dir", default="data/deepsport")
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    inst = load_instants(args.data_dir)
    cov = Counter(n for it in inst for n in it["gt_keypoints"])
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "court-keypoint-frontend",
        "stage": "homography registration front-end — harness + trivial floor (no detector yet)",
        "dataset": {"name": "DeepSportradar basketball-instants", "n_instants": len(inst),
                    "n_with_images": sum(it["image"] is not None for it in inst)},
        "canonical_keypoints": {n: list(xy) for n, xy in CANONICAL.items()},
        "metric": "court-registration reprojection error (px): detect keypoints -> solve H -> vs GT calibration",
        "results": {
            "trivial_floor_global_mean_keypoints_px": _stats(evaluate(None, inst)),
            "gt_keypoint_visibility_frac": {n: round(cov[n] / len(inst), 2) for n in CANONICAL},
            "target_from_inc05": "solver at ~3px keypoints -> ~2px registration; so a detector must reach "
            "few-px keypoint accuracy",
        },
        "provenance": {"court_cm": [COURT_W, COURT_H], "platform": platform.platform()},
        "notes": "Harness-before-modeling for the deferred front-end: a detector is any callable "
        "image->{keypoint:(u,v)}; measured by reprojection error. Floor = predict the global-mean keypoint "
        "locations. NEXT: a learned KaliCalib-lite keypoint detector, trained on these GT keypoints, must beat "
        "the floor. DeepSportradar is fixed arena cameras (not moving broadcast) — a stated dataset boundary.",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"court_keypoints_floor_{stamp}.json").write_text(json.dumps(report, indent=2))
    r = report["results"]["trivial_floor_global_mean_keypoints_px"]
    print(f"Wrote court_keypoints_floor_{stamp}.json | {len(inst)} instants")
    print(f"  trivial floor: median {r['median_px']}px, mean {r['mean_px']}px (n={r['n']})")


if __name__ == "__main__":
    main()
