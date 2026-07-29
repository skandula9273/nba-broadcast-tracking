"""Detection generalization — per-GAME mAP breakdown (is 0.987 uniform, or one easy game?).

The headline detection mAP (0.987) is an aggregate over basketball-val, model-SELECTED on that same split.
This can't produce a truly held-out cross-DATASET number here (SportsMOT test GT is behind Codalab, and
DeepSportradar uses world-coordinate player annotations + a full camera model, not image boxes — projecting
them correctly is error-prone, so a wrong number would be worse than none). The clean, correctly-computable
generalization signal is the PER-GAME spread: the 4 val games are distinct arenas / teams / broadcast styles,
so uniformly high mAP across them is evidence the detector generalizes across games rather than overfitting one
game's peculiarities; a wide spread would mean the headline is carried by an easy game. Reported as-is.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ..config import load_config


def _ver(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def _game(name: str) -> str:
    return name.split("_c")[0]                                   # v_00HRwkvvjtQ_c001__000001.jpg -> v_00HRwkvvjtQ


def run(weights: str, dataset_yaml: str, imgsz: int, device: str) -> dict:
    import yaml as yamllib
    from ultralytics import YOLO

    root = Path(dataset_yaml).parent
    img_dir, lbl_dir = (root / "images" / "val").resolve(), (root / "labels" / "val").resolve()  # abs symlink targets
    by_game: dict[str, list[str]] = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        by_game[_game(p.name)].append(p.name)
    n_val = sum(len(v) for v in by_game.values())

    model = YOLO(weights)
    tmp = Path(tempfile.mkdtemp(prefix="gen_eval_"))
    per_game = {}
    for g, names in sorted(by_game.items()):
        gi, gl = tmp / g / "images", tmp / g / "labels"    # per-game symlinked DIR (the mode Ultralytics likes)
        gi.mkdir(parents=True, exist_ok=True)
        gl.mkdir(parents=True, exist_ok=True)
        for name in names:
            os.symlink(img_dir / name, gi / name)
            lbl = name.rsplit(".", 1)[0] + ".txt"
            if (lbl_dir / lbl).exists():
                os.symlink(lbl_dir / lbl, gl / lbl)
        game_yaml = tmp / f"{g}.yaml"
        game_yaml.write_text(yamllib.safe_dump({"path": str(tmp / g), "train": "images", "val": "images",
                                                "names": {0: "athlete"}}))
        res = model.val(data=str(game_yaml), imgsz=imgsz, device=device, verbose=False, plots=False,
                        save_json=False, project=str(tmp), name=g, exist_ok=True)
        b = res.box
        per_game[g] = {"n_images": len(names), "mAP50": round(float(b.map50), 4),
                       "mAP50_95": round(float(b.map), 4), "precision": round(float(b.mp), 4),
                       "recall": round(float(b.mr), 4)}

    maps = np.array([v["mAP50"] for v in per_game.values()])
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "detection-generalization",
        "stage": "per-game detection mAP breakdown on basketball-val (cross-game generalization signal)",
        "dataset": {"name": "SportsMOT basketball-val (YOLO labels, class 'athlete')", "n_games": len(per_game),
                    "n_val_images": n_val},
        "per_game": per_game,
        "spread_mAP50": {"min": round(float(maps.min()), 4), "max": round(float(maps.max()), 4),
                         "mean": round(float(maps.mean()), 4), "std": round(float(maps.std()), 4)},
        "model": {"weights": weights, "classes": {"0": "athlete"}, "ball_detected": False, "imgsz_val": imgsz},
        "notes": "Per-game spread on the SELECTION split (val), so still mildly optimistic — but a LOW spread "
        "across 4 distinct games (arenas/teams/broadcast styles) is evidence the detector generalizes across "
        "games, not memorizes one. A truly held-out cross-DATASET number needs SportsMOT test (Codalab) or a "
        "box-annotated arena set (DeepSportradar is world-coord). Retrieval cross-game generalization IS "
        "measured elsewhere (semantic_validate held-out-game 0.942; broadcast_encoder held-out-game).",
        "provenance": {"device": device, "versions": {p: _ver(p) for p in ("ultralytics", "torch", "numpy")},
                       "platform": platform.platform()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-game detection mAP generalization breakdown")
    ap.add_argument("--config", default="configs/v0_finetuned.yaml")
    ap.add_argument("--dataset", default="data/sportsmot/_yolo_ft/sportsmot_basketball.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()
    cfg = load_config(args.config)

    report = run(cfg.detect.weights, args.dataset, args.imgsz, cfg.detect.device)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (Path(args.out_dir) / f"detection_generalization_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote detection_generalization_{stamp}.json")
    for g, v in report["per_game"].items():
        print(f"  {g}: mAP50={v['mAP50']}  P={v['precision']}  R={v['recall']}  (n={v['n_images']})")
    print(f"spread mAP50: {report['spread_mAP50']}")


if __name__ == "__main__":
    main()
