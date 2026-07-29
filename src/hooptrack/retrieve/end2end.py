"""End-to-end wire: REAL pipeline output -> possession tensor -> FAISS retrieval, measured vs GT.

Closes the 'not wired' edge. inc-07 *simulated* reconstruction error on GT tracks with an order-of-magnitude
budget; this runs the ACTUAL tracker output (bytetrack_ft, HOTA 0.473 — the committed pipeline product) through
the real tensor adapter (`reconstruct.tracks_to_tensor`) and the real FAISS index, and measures reconstructed-
vs-GT retrieval on real broadcast frames. Per window: query = the tracker-reconstructed tensor, gallery = every
window's GT tensor, relevant = the same window's GT. Encoder = the hand-feature FLOOR — coordinate-space-
agnostic, so valid on the image-coordinate tensors where the SportVU-trained transformer is out-of-domain (no
broadcast homography). That domain gap is itself the finding: a broadcast-trained encoder is the missing piece.

Honest, MEASURED (not simulated) blockers surfaced by actually running it: no ball (entity 0 zero), no broadcast
homography (image coords, not court), tracker fragmentation (top-n most-present ids, canonical x-order). The
number is the real perception degradation the reconstruction imposes on retrieval.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..ingest.frames import read_seqinfo
from ..pipeline import Track
from .reconstruct import frame_windows, tracks_to_tensor
from .run import features
from .train import _faiss_rankings, _metrics_rk, _ver

GT_ROOT = Path("data/sportsmot/trackeval/gt/SportsMOT-basketball-val")
FRAMES_ROOT = Path("data/sportsmot/val")
TRACKERS_ROOT = Path("data/sportsmot/trackeval/trackers/SportsMOT-basketball-val")


def _load_mot(path: Path) -> list[Track]:
    tracks: list[Track] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f, tid, x, y, w, h = (float(v) for v in line.split(",")[:6])
        tracks.append(Track(track_id=int(tid), frame=int(f), cls="athlete", xyxy=(x, y, x + w, y + h)))
    return tracks


def build_tensors(seqs: list[str], tracker: str, T: int, window: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Window every seq; return (gt_tensors, recon_tensors) of shape (N, T, 11, 2), aligned by window."""
    gt_list, recon_list = [], []
    for seq in seqs:
        info = read_seqinfo(FRAMES_ROOT / seq)
        w, h = info["imWidth"] or 1280, info["imHeight"] or 720
        gt = _load_mot(GT_ROOT / seq / "gt" / "gt.txt")
        recon = _load_mot(TRACKERS_ROOT / tracker / "data" / f"{seq}.txt")
        for f0, f1 in frame_windows(gt, window):
            gt_list.append(tracks_to_tensor(gt, f0, f1, T, w, h))
            recon_list.append(tracks_to_tensor(recon, f0, f1, T, w, h))
    return np.stack(gt_list), np.stack(recon_list), len(seqs)


def run(args) -> dict:
    seqs = sorted(p.name for p in GT_ROOT.iterdir() if (p / "gt" / "gt.txt").is_file())
    seqs = [s for s in seqs if (TRACKERS_ROOT / args.tracker / "data" / f"{s}.txt").is_file()]
    gt, recon, n_seq = build_tensors(seqs, args.tracker, args.T, args.window)
    n = len(gt)

    genc = features(gt)                                          # gallery = GT windows (floor features)
    recon_vs_gt = _metrics_rk(_faiss_rankings(features(recon), genc))   # query = reconstructed
    gt_self = _metrics_rk(_faiss_rankings(genc, genc))          # sanity: GT retrieves itself (window distinctness)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "end2end-wire",
        "stage": "REAL pipeline output -> tensor -> FAISS retrieval (reconstructed vs GT)",
        "dataset": {"source": "SportsMOT basketball-val", "n_seqs": n_seq, "n_windows": n,
                    "window_frames": args.window, "T": args.T, "tracker": args.tracker,
                    "tracker_hota": 0.473, "chance_recall@1": round(1.0 / max(1, n), 4)},
        "method": {"query": "tracker-reconstructed window tensor", "gallery": "GT window tensors",
                   "relevant": "same window's GT", "retrieval": "FAISS IndexFlatIP (cosine)",
                   "encoder": "hand-feature floor (coordinate-agnostic; valid on image-coord tracks)"},
        "results": {"reconstructed_vs_gt": recon_vs_gt, "gt_self_retrieval_sanity": gt_self},
        "blockers_made_concrete": {
            "no_ball": "entity 0 (ball) zeroed — single-class athlete detector",
            "no_broadcast_homography": "image-coordinate foot-points, not court coords (keypoint net doesn't "
            "transfer to broadcast) — so the SportVU-trained transformer is out-of-domain; floor used instead",
            "fragmentation": "top-10 most-present tracker ids per window, canonical x-order; a fragmented/"
            "dropped player is real reconstruction error and shows as tensor mismatch",
            "windows_not_possessions": "SportsMOT is continuous tracking, not possession-segmented; fixed "
            f"{args.window}-frame non-overlapping windows are the retrieval items",
        },
        "provenance": {"versions": {p: _ver(p) for p in ("numpy", "faiss-cpu")}, "platform": platform.platform()},
        "notes": "First run of REAL pipeline output through the retrieval core + index (was fed only from "
        "SportVU GT). Upgrades inc-07's simulated error model to the actual tracker's errors on the perception "
        "side. Trained transformer needs a broadcast-domain retrain (documented blocker); floor is the valid "
        "coordinate-agnostic metric here.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end reconstructed-vs-GT retrieval on real tracker output")
    ap.add_argument("--tracker", default="bytetrack_ft")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / f"end2end_{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    d, r = report["dataset"], report["results"]
    print(f"wrote {out}")
    print(f"n_windows={d['n_windows']} (chance r@1={d['chance_recall@1']})")
    print(f"reconstructed-vs-GT: {r['reconstructed_vs_gt']}")
    print(f"GT self-retrieval sanity: {r['gt_self_retrieval_sanity']}")


if __name__ == "__main__":
    main()
