"""Fine-tune the YOLO detector on SportsMOT athletes. CLI for increment-04.

Why: the attribution study (increment-03) showed the tracking ceiling is the detector — COCO "person"
was never told what an athlete is (best DetA 0.44). This fine-tunes yolov8m from its COCO weights on
SportsMOT-basketball athlete boxes, so the detector learns the athlete class (dropping the crowd/bench
false positives that motion-only tracking can't). One variable downstream: swap the weights, re-run the
same ByteTrack pipeline, measure the DetA/HOTA lift vs the 0.301 floor.

Pipeline: download basketball-train -> MOT gt -> YOLO detection labels (single class 'athlete'),
frame-subsampled (consecutive 25fps frames are near-duplicates); a symlinked YOLO dataset (no pixel
copies); fine-tune; save best.pt. Inference imgsz stays 1280 (baseline) when re-running tracking; only
the training imgsz is set here. APIs confirmed against the installed ultralytics before the long run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

from ..config import load_config
from ..ingest.fetch import download_split_tar, download_splits_txt, extract_sequences
from ..ingest.frames import read_seqinfo

SPORT = "basketball"


def _basketball_seqs(cache: Path, mot_split: str) -> list[str]:
    splits = download_splits_txt(cache)
    sport = set((splits / f"{SPORT}.txt").read_text().split())
    split = set((splits / f"{mot_split}.txt").read_text().split())
    return sorted(sport & split)


def _write_yolo_labels(seq_dir: Path, split_label: str, subsample: int, ds_root: Path) -> int:
    """Convert one MOT sequence's gt.txt -> YOLO labels + symlinked images (every `subsample`-th frame)."""
    info = read_seqinfo(seq_dir)
    w, h = info["imWidth"], info["imHeight"]
    img_dir = seq_dir / info["imDir"]
    ext = info["imExt"]
    # group GT boxes by frame
    by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        c = line.split(",")
        fr = int(c[0])
        x, y, bw, bh = float(c[2]), float(c[3]), float(c[4]), float(c[5])
        by_frame.setdefault(fr, []).append((x, y, bw, bh))

    img_out = ds_root / "images" / split_label
    lbl_out = ds_root / "labels" / split_label
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n = 0
    for fr in sorted(by_frame):
        if (fr - 1) % subsample != 0:  # keep frames 1, 1+subsample, ...
            continue
        src_img = img_dir / f"{fr:06d}{ext}"
        if not src_img.exists():
            continue
        stem = f"{seq_dir.name}__{fr:06d}"
        link = img_out / f"{stem}{ext}"
        if not link.exists():
            os.symlink(src_img.resolve(), link)
        lines = []
        for x, y, bw, bh in by_frame[fr]:
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    return n


def prepare_data(data_dir: Path, subsample: int) -> tuple[Path, dict]:
    """Download + lay out a YOLO detection dataset (basketball train/val). Returns (dataset_yaml, stats)."""
    cache = data_dir / "_hf_cache"
    ds_root = data_dir / "_yolo_ft"
    counts = {}
    for mot_split, label in (("train", "train"), ("val", "val")):
        seqs = _basketball_seqs(cache, mot_split)
        tar = download_split_tar(cache, mot_split)
        split_dir = data_dir / mot_split
        # extract only if not already present (val is already here from increment-01)
        missing = [s for s in seqs if not (split_dir / s / "gt" / "gt.txt").exists()]
        if missing:
            extract_sequences(tar, split_dir, set(missing))
        total = sum(_write_yolo_labels(split_dir / s, label, subsample, ds_root) for s in seqs)
        counts[label] = {"sequences": len(seqs), "images": total}

    dataset_yaml = ds_root / "sportsmot_basketball.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {"path": str(ds_root.resolve()), "train": "images/train", "val": "images/val",
             "names": {0: "athlete"}},
            sort_keys=False,
        )
    )
    return dataset_yaml, counts


def train(
    cfg_path: str, epochs: int, imgsz: int, batch: int, subsample: int, device: str,
    amp: bool = False, deterministic: bool = False, base_weights: str | None = None,
    out_subdir: str = "finetuned",
) -> dict:
    from ultralytics import YOLO

    cfg = load_config(cfg_path)
    data_dir = Path(cfg.eval.data_dir)
    dataset_yaml, counts = prepare_data(data_dir, subsample)
    print(f"YOLO dataset ready: {counts}  -> {dataset_yaml}")

    # base_weights lets us fine-tune a DIFFERENT backbone (e.g. yolov8n for the model-size Pareto axis) without
    # touching the committed yolov8m; out_subdir keeps its weights/ dir separate so best.pt is never clobbered.
    weights = base_weights or cfg.detect.weights or "yolov8m.pt"
    model = YOLO(weights)
    project = str((data_dir / "_ft_runs").resolve())
    name = f"basketball_ft_{Path(weights).stem}_e{epochs}_s{subsample}_i{imgsz}"
    # MPS notes (learned from the 1-epoch measurement): amp on MPS -> NaN losses; large batch/imgsz saturate
    # the 16GB unified memory and thrash. amp=False + batch 4 + imgsz 640 + cache off keep it stable.
    model.train(
        data=str(dataset_yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        seed=cfg.seed, project=project, name=name, exist_ok=True, patience=0, verbose=True,
        amp=amp, deterministic=deterministic, cache=False,
    )
    best = Path(model.trainer.best)  # robust to Ultralytics save_dir layout
    out_dir = Path("weights") / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "best.pt"
    if best.exists():
        shutil.copy(best, dst)
    meta = {
        "base_weights": weights, "epochs": epochs, "imgsz_train": imgsz, "batch": batch,
        "subsample": subsample, "device": device, "amp": amp, "deterministic": deterministic,
        "seed": cfg.seed, "counts": counts, "best_weights": str(dst), "run_dir": str(best.parent.parent),
    }
    (out_dir / "finetune_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Fine-tuned weights -> {dst}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the YOLO detector on SportsMOT basketball")
    ap.add_argument("--config", default="configs/v0.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)   # MPS-feasible on 16GB unified memory
    ap.add_argument("--batch", type=int, default=4)     # larger batches thrash + destabilize on MPS
    ap.add_argument("--subsample", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--amp", action="store_true", default=False)          # AMP on MPS -> NaN losses
    ap.add_argument("--deterministic", action="store_true", default=False)
    ap.add_argument("--base-weights", default=None, help="backbone to fine-tune (e.g. yolov8n.pt); "
                    "default = config weights or yolov8m.pt")
    ap.add_argument("--out-subdir", default="finetuned", help="weights/<subdir>/best.pt (keep separate to "
                    "avoid clobbering the committed yolov8m in weights/finetuned/)")
    args = ap.parse_args()
    train(args.config, args.epochs, args.imgsz, args.batch, args.subsample, args.device,
          amp=args.amp, deterministic=args.deterministic, base_weights=args.base_weights,
          out_subdir=args.out_subdir)


if __name__ == "__main__":
    main()
