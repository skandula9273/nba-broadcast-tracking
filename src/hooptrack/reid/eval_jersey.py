"""Jersey-OCR coverage harness — reproducible, committed-JSON measurement of the OCR stage in isolation.

We run `JerseyOCR` over **GT-boxed athletes** (SportsMOT basketball-val GT tracks), so the number measures the
OCR stage's own capability *given correct tracking* — detector/tracker error is factored out. SportsMOT has no
jersey labels, so we cannot measure accuracy; we report **coverage** (fraction of tracks that get a confident
majority number) with **cross-frame vote consensus** as the precision proxy (a real number reads consistently
across a track's frames; noise doesn't). The precision thresholds (`min_conf`, `min_votes`) are held FIXED
across the ablation, so coverage gains come from better evidence (crops/contrast/scale), never a lowered bar.

Additive ablation — each config changes ONE lever from the previous:
  baseline -> +preprocess (CLAHE) -> +upscale -> +more/even crops -> +band. The winner is promoted to the
  `JerseyOCR` defaults (and hence the deployed re-ID overlay).

Run:  python -m hooptrack.reid.eval_jersey --limit-seqs 4          # ablation on a subset
      python -m hooptrack.reid.eval_jersey --configs band --seqs all   # winner on all seqs
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ..ingest.frames import load_mot_sequence
from ..pipeline import Track
from .jersey import JerseyOCR

GT_ROOT = Path("data/sportsmot/trackeval/gt/SportsMOT-basketball-val")
FRAMES_ROOT = Path("data/sportsmot/val")

# Additive ablation: each row changes exactly one lever from the row above. min_conf/min_votes fixed (0.4/2).
CONFIGS: dict[str, dict] = {
    "baseline":   dict(max_crops=15, band=(0.12, 0.55), upscale=3, preprocess=False, stride_sample=False),
    "preprocess": dict(max_crops=15, band=(0.12, 0.55), upscale=3, preprocess=True,  stride_sample=False),
    "upscale4":   dict(max_crops=15, band=(0.12, 0.55), upscale=4, preprocess=True,  stride_sample=False),
    "morecrops":  dict(max_crops=40, band=(0.12, 0.55), upscale=4, preprocess=True,  stride_sample=True),
    "band":       dict(max_crops=40, band=(0.10, 0.52), upscale=4, preprocess=True,  stride_sample=True),
    # --- attribution: the morecrops jump changed TWO knobs at once (count AND even-sampling). Isolate each,
    #     and test whether CLAHE preprocess (which HURT alone) helps or hurts inside the more-crops regime.
    "crops_head": dict(max_crops=40, band=(0.10, 0.52), upscale=4, preprocess=True,  stride_sample=False),
    "even_15":    dict(max_crops=15, band=(0.10, 0.52), upscale=4, preprocess=True,  stride_sample=True),
    "band_noprep": dict(max_crops=40, band=(0.10, 0.52), upscale=4, preprocess=False, stride_sample=True),
}


def load_gt_tracks(seq: str) -> list[Track]:
    """GT MOT rows -> Track objects. MOT: frame(1-based), id, x, y, w, h, conf, cls, vis (all athletes)."""
    rows = (GT_ROOT / seq / "gt" / "gt.txt").read_text().splitlines()
    tracks: list[Track] = []
    for line in rows:
        if not line.strip():
            continue
        f, tid, x, y, w, h = (float(v) for v in line.split(",")[:6])
        tracks.append(Track(track_id=int(tid), frame=int(f), cls="athlete",
                            xyxy=(x, y, x + w, y + h)))
    return tracks


def seq_names(spec: str, limit: int | None) -> list[str]:
    all_seqs = sorted(p.name for p in GT_ROOT.iterdir() if (p / "gt" / "gt.txt").is_file())
    seqs = all_seqs if spec == "all" else [s for s in spec.split(",") if s in all_seqs]
    return seqs[:limit] if limit else seqs


def aggregate(details: list[dict], hi_consensus: float = 0.6) -> dict:
    """Coverage + consensus over per-track `read_detail` dicts (pure; unit-tested). `track_coverage` = any
    confident majority number; `hi_consensus_coverage` additionally requires >= `hi_consensus` of the reads to
    agree (precision-controlled); `crop_read_rate` = per-crop hit rate."""
    n_tracks = len(details)
    n_covered = n_hi = tot_crops = tot_read = 0
    numbers: list[str] = []
    for d in details:
        tot_crops += d["n_crops"]
        tot_read += d["n_read_crops"]
        num = d["number"]
        if num is not None:
            n_covered += 1
            numbers.append(num)
            if d["reads"].count(num) / max(1, len(d["reads"])) >= hi_consensus:
                n_hi += 1
    return {
        "n_tracks": n_tracks,
        "track_coverage": round(n_covered / max(1, n_tracks), 3),
        "hi_consensus_coverage": round(n_hi / max(1, n_tracks), 3),
        "crop_read_rate": round(tot_read / max(1, tot_crops), 3),
        "numbers_read": dict(Counter(numbers).most_common()),
    }


def measure(config: dict, seqs: list[str]) -> dict:
    """Run one JerseyOCR config over all seqs' GT tracks; aggregate coverage + consensus."""
    ocr = JerseyOCR(min_conf=0.4, min_votes=2, **config)
    details: list[dict] = []
    for seq in seqs:
        frames = load_mot_sequence(FRAMES_ROOT / seq)
        details.extend(ocr.read_detail(load_gt_tracks(seq), frames).values())
    return aggregate(details)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="all", help="'all', or comma-separated sequence names")
    ap.add_argument("--limit-seqs", type=int, default=None, help="cap number of sequences (for a fast ablation)")
    ap.add_argument("--configs", default=None, help="comma-separated config names; default = full ablation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seqs = seq_names(args.seqs, args.limit_seqs)
    names = args.configs.split(",") if args.configs else list(CONFIGS)
    print(f"sequences ({len(seqs)}): {seqs}")

    results = {}
    for name in names:
        print(f"\n=== {name}: {CONFIGS[name]} ===")
        r = measure(CONFIGS[name], seqs)
        results[name] = {"config": {**CONFIGS[name], "min_conf": 0.4, "min_votes": 2}, **r}
        print(f"  coverage={r['track_coverage']}  hi_consensus={r['hi_consensus_coverage']}  "
              f"crop_read_rate={r['crop_read_rate']}  n_tracks={r['n_tracks']}")

    report = {
        "task": "jersey_ocr_coverage",
        "basis": "SportsMOT basketball-val GT-boxed athletes (OCR isolated from tracker error); "
                 "coverage not accuracy (no jersey labels); consensus = cross-frame precision proxy",
        "sequences": seqs,
        "utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "results": results,
    }
    out = Path(args.out) if args.out else Path("eval_results") / f"jersey_ocr_{report['utc']}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
