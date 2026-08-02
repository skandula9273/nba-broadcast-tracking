"""Fetch/prepare SportsMOT and build the TrackEval GT layout. CLI entry for `make data`.

Honest by construction: this downloads the **official** `MCG-NJU/SportsMOT` distribution from the
HuggingFace Hub (the per-split tars + `splits_txt/`) — it does not scrape, relabel, or redistribute data.
It then extracts only the sequences we evaluate (sport ∩ MOT-split) and lays them out for TrackEval
following the authors' own `scripts/sportsmot_to_trackeval.py`:

    data/sportsmot/
      <mot_split>/<seq>/{img1/*.jpg, gt/gt.txt, seqinfo.ini}      # raw, for detection
      trackeval/
        gt/<BENCHMARK>-<eval_split>/<seq>/{gt/gt.txt, seqinfo.ini}
        gt/seqmaps/<BENCHMARK>-<eval_split>.txt                    # header 'name' + one seq per line
        trackers/<BENCHMARK>-<eval_split>/<tracker>/data/          # tracker outputs land here (Phase 3)
      manifest.json                                                # provenance: repo, seqs, counts

Only the ~6.6 GB val tar is pulled; test GT is withheld behind Codalab, so `val` is the split with public
ground truth. Selective extraction keeps only the basketball-val sequences (~15 of 45).
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config, load_config

HF_REPO = "MCG-NJU/SportsMOT"
SPORTS = ("basketball", "football", "volleyball")


def _load_split_names(names_dir: Path, name: str) -> set[str]:
    return set((names_dir / f"{name}.txt").read_text().split())


def download_splits_txt(dest: Path) -> Path:
    """Download splits_txt/*.txt (tiny). Returns the local splits_txt dir."""
    from huggingface_hub import hf_hub_download

    for name in (*SPORTS, "train", "val", "test"):
        hf_hub_download(
            HF_REPO, f"splits_txt/{name}.txt", repo_type="dataset", local_dir=str(dest)
        )
    return dest / "splits_txt"


def download_split_tar(dest: Path, mot_split: str) -> Path:
    """Download dataset/<mot_split>.tar (cached/resumed by huggingface_hub). Returns the tar path."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            HF_REPO, f"dataset/{mot_split}.tar", repo_type="dataset", local_dir=str(dest)
        )
    )


def extract_sequences(tar_path: Path, out_split_dir: Path, seqs: set[str]) -> list[str]:
    """Extract only members belonging to `seqs` into out_split_dir/<seq>/... (path-traversal safe)."""
    out_split_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            if Path(m.name).name.startswith("."):  # skip macOS AppleDouble/._* + hidden files
                continue
            parts = Path(m.name).parts
            # locate the sequence-name component; keep everything after it
            hit = next((i for i, p in enumerate(parts) if p in seqs), None)
            if hit is None:
                continue
            seq = parts[hit]
            rel = Path(*parts[hit + 1 :])
            dest = out_split_dir / seq / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            with src, open(dest, "wb") as f:
                shutil.copyfileobj(src, f)
            found.add(seq)
    return sorted(found)


def build_trackeval_gt(split_dir: Path, te_root: Path, gt_set: str, seqs: list[str]) -> Path:
    """Copy gt/ + seqinfo.ini into the TrackEval GT tree and write a header'd seqmap. Returns GT dir."""
    gt_dir = te_root / "gt" / gt_set
    seqmap_dir = te_root / "gt" / "seqmaps"
    gt_dir.mkdir(parents=True, exist_ok=True)
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        dst = gt_dir / seq
        (dst / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy(split_dir / seq / "gt" / "gt.txt", dst / "gt" / "gt.txt")
        shutil.copy(split_dir / seq / "seqinfo.ini", dst / "seqinfo.ini")
    # TrackEval MOTChallenge seqmaps skip row 0 as a header (verified in mot_challenge_2d_box.py).
    (seqmap_dir / f"{gt_set}.txt").write_text("name\n" + "\n".join(seqs) + "\n")
    (te_root / "trackers" / gt_set).mkdir(parents=True, exist_ok=True)
    return gt_dir


def prepare(cfg: Config) -> dict:
    data_dir = Path(cfg.eval.data_dir)
    cache = data_dir / "_hf_cache"
    sport, mot_split = cfg.eval.split, cfg.eval.mot_split
    gt_set = f"{cfg.eval.benchmark}-{cfg.eval.eval_split}"  # e.g. SportsMOT-basketball-val

    splits_dir = download_splits_txt(cache)
    if sport not in SPORTS:
        raise ValueError(f"eval.split must be one of {SPORTS}, got {sport!r}")
    seqs_target = sorted(_load_split_names(splits_dir, sport) & _load_split_names(splits_dir, mot_split))
    if not seqs_target:
        raise RuntimeError(f"no sequences for {sport} ∩ {mot_split}")

    tar_path = download_split_tar(cache, mot_split)
    split_dir = data_dir / mot_split
    extracted = extract_sequences(tar_path, split_dir, set(seqs_target))
    missing = sorted(set(seqs_target) - set(extracted))
    if missing:
        raise RuntimeError(f"{len(missing)} target sequences not found in tar: {missing[:3]}...")

    te_root = data_dir / "trackeval"
    build_trackeval_gt(split_dir, te_root, gt_set, extracted)

    manifest = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"hf_repo": HF_REPO, "repo_type": "dataset", "tar": f"dataset/{mot_split}.tar"},
        "sport": sport,
        "mot_split": mot_split,
        "gt_set": gt_set,
        "n_sequences": len(extracted),
        "sequences": extracted,
        "frames_per_seq": {
            # exclude hidden/AppleDouble (._*) files that pathlib.glob would otherwise count
            s: sum(1 for p in (split_dir / s / "img1").glob("*.jpg") if not p.name.startswith("."))
            for s in extracted
        },
        "note": "test GT is withheld (Codalab); val is the public-GT split. Detector is COCO-pretrained "
        "(person class) — NOT fine-tuned on SportsMOT athletes.",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare SportsMOT + TrackEval GT layout")
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = prepare(cfg)
    total = sum(m["frames_per_seq"].values())
    print(
        f"Prepared {m['n_sequences']} {m['sport']}-{m['mot_split']} sequences "
        f"({total} frames) -> {cfg.eval.data_dir}  [gt_set={m['gt_set']}]"
    )


if __name__ == "__main__":
    main()
