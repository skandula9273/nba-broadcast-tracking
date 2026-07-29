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
TRACKERS_ROOT = Path("data/sportsmot/trackeval/trackers/SportsMOT-basketball-val")

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


def _load_mot(path: Path) -> list[Track]:
    """MOT rows -> Track objects. MOT: frame(1-based), id, x, y, w, h, ... (all athletes)."""
    tracks: list[Track] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f, tid, x, y, w, h = (float(v) for v in line.split(",")[:6])
        tracks.append(Track(track_id=int(tid), frame=int(f), cls="athlete", xyxy=(x, y, x + w, y + h)))
    return tracks


def load_gt_tracks(seq: str) -> list[Track]:
    """GT boxes (10 full-length ids/seq) — the OCR ceiling given correct tracking."""
    return _load_mot(GT_ROOT / seq / "gt" / "gt.txt")


def load_tracker_tracks(seq: str, tracker: str) -> list[Track]:
    """Real tracker output (fragmented into many short ids, id-swaps) — the true operating point."""
    return _load_mot(TRACKERS_ROOT / tracker / "data" / f"{seq}.txt")


def seq_names(spec: str, limit: int | None) -> list[str]:
    all_seqs = sorted(p.name for p in GT_ROOT.iterdir() if (p / "gt" / "gt.txt").is_file())
    seqs = all_seqs if spec == "all" else [s for s in spec.split(",") if s in all_seqs]
    return seqs[:limit] if limit else seqs


def aggregate(details: list[dict], hi_consensus: float = 0.6, min_substantial: int = 10) -> dict:
    """Coverage + consensus over per-track `read_detail` dicts (pure; unit-tested). `track_coverage` = any
    confident majority number; `hi_consensus_coverage` additionally requires >= `hi_consensus` of the reads to
    agree (precision-controlled); `crop_read_rate` = per-crop hit rate. `coverage_substantial` = coverage among
    tracks long enough to have a chance (>= `min_substantial` crops) — separates fragmentation (many short
    tracker ids can't be read) from OCR capability, so the GT->tracker drop can be attributed."""
    n_tracks = len(details)
    n_covered = n_hi = tot_crops = tot_read = 0
    n_sub = n_sub_cov = 0
    numbers: list[str] = []
    for d in details:
        tot_crops += d["n_crops"]
        tot_read += d["n_read_crops"]
        num = d["number"]
        if d["n_crops"] >= min_substantial:
            n_sub += 1
            n_sub_cov += num is not None
        if num is not None:
            n_covered += 1
            numbers.append(num)
            if d["reads"].count(num) / max(1, len(d["reads"])) >= hi_consensus:
                n_hi += 1
    return {
        "n_tracks": n_tracks,
        "track_coverage": round(n_covered / max(1, n_tracks), 3),
        "hi_consensus_coverage": round(n_hi / max(1, n_tracks), 3),
        "coverage_substantial": round(n_sub_cov / max(1, n_sub), 3),
        "n_substantial": n_sub,
        "crop_read_rate": round(tot_read / max(1, tot_crops), 3),
        "numbers_read": dict(Counter(numbers).most_common()),
    }


def measure(config: dict, seqs: list[str], source: str = "gt", tracker: str = "bytetrack_ft") -> dict:
    """Run one JerseyOCR config over all seqs' tracks (GT or real tracker output); aggregate coverage."""
    ocr = JerseyOCR(min_conf=0.4, min_votes=2, **config)
    details: list[dict] = []
    for seq in seqs:
        frames = load_mot_sequence(FRAMES_ROOT / seq)
        tracks = load_gt_tracks(seq) if source == "gt" else load_tracker_tracks(seq, tracker)
        details.extend(ocr.read_detail(tracks, frames).values())
    return aggregate(details)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="all", help="'all', or comma-separated sequence names")
    ap.add_argument("--limit-seqs", type=int, default=None, help="cap number of sequences (for a fast ablation)")
    ap.add_argument("--configs", default=None, help="comma-separated config names; default = full ablation")
    ap.add_argument("--source", default="gt", choices=["gt", "tracker"], help="track source: GT boxes or tracker")
    ap.add_argument("--tracker", default="bytetrack_ft", help="tracker name under trackers/ (--source tracker)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seqs = seq_names(args.seqs, args.limit_seqs)
    if args.source == "tracker":                         # keep only seqs this tracker actually produced
        seqs = [s for s in seqs if (TRACKERS_ROOT / args.tracker / "data" / f"{s}.txt").is_file()]
    names = args.configs.split(",") if args.configs else list(CONFIGS)
    src_label = "GT boxes" if args.source == "gt" else f"tracker output ({args.tracker})"
    print(f"source: {src_label}  |  sequences ({len(seqs)}): {seqs}")

    results = {}
    for name in names:
        print(f"\n=== {name}: {CONFIGS[name]} ===")
        r = measure(CONFIGS[name], seqs, source=args.source, tracker=args.tracker)
        results[name] = {"config": {**CONFIGS[name], "min_conf": 0.4, "min_votes": 2}, **r}
        print(f"  coverage={r['track_coverage']}  substantial={r['coverage_substantial']} "
              f"(n={r['n_substantial']})  hi_consensus={r['hi_consensus_coverage']}  "
              f"crop_read_rate={r['crop_read_rate']}  n_tracks={r['n_tracks']}")

    basis = ("SportsMOT basketball-val GT-boxed athletes (OCR isolated from tracker error)" if args.source == "gt"
             else f"SportsMOT basketball-val REAL tracker output ({args.tracker}, HOTA 0.473) — the operating "
                  "point; fragmented/id-swapped tracks vs GT's 10 full-length ids/seq")
    report = {
        "task": "jersey_ocr_coverage",
        "source": args.source,
        "tracker": args.tracker if args.source == "tracker" else None,
        "basis": basis + "; coverage not accuracy (no jersey labels); consensus = cross-frame precision proxy",
        "sequences": seqs,
        "utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "results": results,
    }
    tag = "" if args.source == "gt" else f"{args.source}_"
    out = Path(args.out) if args.out else Path("eval_results") / f"jersey_ocr_{tag}{report['utc']}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
