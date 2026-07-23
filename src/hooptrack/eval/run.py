"""Eval runner -> a timestamped JSON committed to eval_results/.

Honest by construction: it records the config + which metrics actually ran. It NEVER emits a fabricated
number — a metric that isn't wired/run yet is written as null with a `status`, so a scaffold run can't be
misread as a result (the SEC lesson: infra/absent results must not masquerade as numbers).

For increment-01 it runs **tracking HOTA/MOTA/IDF1** (via the TrackEval adapter) on the tracker outputs
that `make track` produced, and carries the honesty caveats alongside the number. Detection mAP, retrieval
recall@k, and the degradation study stay null until their stages land.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config, load_config
from .trackeval_adapter import run_hota

# SportsMOT has no distractor classes -> the authors' comparable setting (verified against TrackEval).
DO_PREPROC = False
CLASSES_TO_EVAL = ("pedestrian",)  # SportsMOT athletes are GT class 1 == MOTChallenge 'pedestrian'

CAVEATS = [
    "Detector is COCO-pretrained (person class) as an athlete proxy — NOT fine-tuned on SportsMOT; "
    "referees/bench/crowd it calls 'person' are false positives vs athlete-only GT and depress DetA.",
    "ByteTrack is motion-only (Kalman + IoU). SportsMOT shows motion-only underperforms on sports — "
    "this is the intended floor before the BoT-SORT/appearance ablation.",
    "HOTA is on the val split (test GT is withheld behind Codalab): comparable to published SportsMOT "
    "val baselines, not the test leaderboard.",
    "TrackEval DO_PREPROC=False (SportsMOT has no distractor classes); CLASSES_TO_EVAL=['pedestrian'].",
    "MPS inference is deterministic only same-machine / same-versions (seed fixed, versions pinned).",
]


def _paths(cfg: Config):
    gt_set = f"{cfg.eval.benchmark}-{cfg.eval.eval_split}"
    te = Path(cfg.eval.data_dir) / "trackeval"
    return {
        "gt_set": gt_set,
        "gt_folder": te / "gt",
        "trackers_folder": te / "trackers",
        "seqmap": te / "gt" / "seqmaps" / f"{gt_set}.txt",
        "tracker_data": te / "trackers" / gt_set / cfg.track.method / "data",
        "run_stats": te / "trackers" / gt_set / cfg.track.method / "run_stats.json",
    }


def build_report(cfg: Config) -> dict:
    p = _paths(cfg)
    manifest_path = Path(cfg.eval.data_dir) / "manifest.json"
    dataset = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    tracked = sorted(f.stem for f in p["tracker_data"].glob("*.txt")) if p["tracker_data"].exists() else []
    tracking: dict = {m: None for m in cfg.eval.metrics}
    provenance = None
    evaluated: list[str] = []
    if tracked:
        # Score exactly the sequences that were tracked (robust to smoke caps / partial runs). Order by
        # the GT seqmap for determinism. For the full uncapped run this is all basketball-val seqs.
        gt_order = [s for s in p["seqmap"].read_text().split() if s and s != "name"]
        evaluated = [s for s in gt_order if s in set(tracked)]
        eval_seqmap = p["tracker_data"].parent / "eval_seqmap.txt"
        eval_seqmap.write_text("name\n" + "\n".join(evaluated) + "\n")
        summary = run_hota(
            gt_folder=p["gt_folder"],
            trackers_folder=p["trackers_folder"],
            benchmark=cfg.eval.benchmark,
            split=cfg.eval.eval_split,
            tracker_name=cfg.track.method,
            seqmap_file=eval_seqmap,
            classes_to_eval=CLASSES_TO_EVAL,
            do_preproc=DO_PREPROC,
        )
        tracking = summary
        if p["run_stats"].exists():
            provenance = json.loads(p["run_stats"].read_text())
        status = (
            f"tracking measured — {cfg.track.method} on {p['gt_set']} "
            f"(HOTA={summary['HOTA']:.4f})"
        )
    else:
        status = f"scaffold — no tracker outputs at {p['tracker_data']}; run `make track` first"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "increment": "01-tracking-baseline",
        "status": status,
        "dataset": {
            "name": cfg.eval.dataset,
            "gt_set": p["gt_set"],
            "mot_split": cfg.eval.mot_split,
            "n_sequences_prepared": dataset.get("n_sequences"),
            "n_sequences_evaluated": len(evaluated),
            "sequences_evaluated": evaluated,
        },
        "config": cfg.model_dump(),
        "results": {
            "detection_mAP": None,  # own stage (needs GT box matching + a fine-tune step)
            "tracking": tracking,
            "retrieval": {f"recall@{k}": None for k in cfg.eval.retrieval_ks},  # V1
            "degradation_study": None,  # reconstructed-vs-GT gap (V1)
        },
        "trackeval": {
            "do_preproc": DO_PREPROC,
            "classes_to_eval": list(CLASSES_TO_EVAL),
            "metric": "HOTA (mean over alpha thresholds) + CLEAR + Identity",
        },
        "provenance": provenance,
        "caveats": CAVEATS,
        "notes": "Fill each result only when its stage is implemented and actually run. No fabricated numbers.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the eval harness -> committed JSON")
    ap.add_argument("--config", default="configs/v0.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    report = build_report(cfg)

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}  (status: {report['status']})")


if __name__ == "__main__":
    main()
