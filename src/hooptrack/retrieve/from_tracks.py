"""ONE demonstrated end-to-end path: broadcast frames -> detect -> track -> court coords -> possession tensor
-> embedding -> retrieval. Scoped to a SINGLE clip, with every crutch made explicit. (increment-10)

This converts a pipeline `TrackResult` into a `(T, 11, 2)` possession tensor the retrieval encoder consumes,
handling the REAL failure cases and being honest about each substitution:

  - **Court coords from a HAND-CLICKED homography** (fixture `tests/fixtures/court_correspondences_*.json`),
    not a homography front-end. Foot-points are projected through that H and normalized to the 94x50 court.
  - **Entity ordering IN PLACE OF re-ID.** Track IDs are NOT player identities. We take the `n_players`
    longest tracks in the window and order them by **(track length desc, then first-frame x asc)** — a
    deterministic but ARBITRARY slot assignment. This is *exactly* the player-slot permutation / order
    corruption the degradation study (inc-07/08/09) models — and that study showed the order-sensitive encoder
    is fragile to it. So a poor retrieval here is the study's prediction made empirical, not a bug.
  - **No ball** -> entity 0 (ball) is left zero (the detector is athlete-only).
  - **Gaps / != 10 tracks** -> per-track linear interpolation over the window; fewer than n_players -> zero-pad.
  - **One 48-frame window** of a moving-camera clip, retrieved against a DIFFERENT-season SportVU gallery.

`main()` runs the whole path on one clip and prints top-5. If the neighbours are garbage, that IS the result.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..pipeline import Track, TrackResult
from .possessions import COURT_L, COURT_W, N_ENTITIES


def load_homography(fixture_path: str | Path) -> tuple[np.ndarray, dict]:
    """Hand-clicked correspondences -> a 3x3 homography (image px -> court feet) + the fixture for provenance."""
    import cv2

    fx = json.loads(Path(fixture_path).read_text())
    src = np.array([c["px"] for c in fx["correspondences"]], dtype=np.float32)
    dst = np.array([c["court_ft"] for c in fx["correspondences"]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)                      # 4+ points; least-squares if >4
    return H, fx


def reprojection_error_ft(H: np.ndarray, fx: dict) -> float:
    """Mean reprojection error in FEET on the HELD-OUT check points (the honest accuracy estimate). Falls back
    to the fit correspondences only if no check points exist — but that number is 0 by construction (4 points
    determine an 8-DOF homography exactly), so it is a tautology, not accuracy."""
    pts = fx.get("check_points") or fx["correspondences"]
    errs = []
    for c in pts:
        u, v = c["px"]
        p = H @ np.array([u, v, 1.0])
        xy = p[:2] / p[2]
        errs.append(float(np.hypot(*(xy - np.array(c["court_ft"])))))
    return round(float(np.mean(errs)), 2)


def order_tracks(result: TrackResult, f0: int, f1: int, n_players: int) -> list[int]:
    """The re-ID SUBSTITUTE (documented): among tracks present in [f0, f1], take the `n_players` longest, then
    order them by (length desc, first-frame x asc). Deterministic but NOT player identity -> arbitrary slots."""
    by_id: dict[int, list[Track]] = defaultdict(list)
    for t in result.tracks:
        if f0 <= t.frame <= f1:
            by_id[t.track_id].append(t)
    stats = {}
    for tid, ts in by_id.items():
        ts.sort(key=lambda t: t.frame)
        first = min(ts, key=lambda t: t.frame)
        stats[tid] = (len(ts), (first.xyxy[0] + first.xyxy[2]) / 2.0)
    longest = sorted(stats, key=lambda t: -stats[t][0])[:n_players]     # take the n_players longest tracks
    return sorted(longest, key=lambda t: (-stats[t][0], stats[t][1]))   # then order: length desc, first-x asc


def _project_court(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    p = H @ np.array([u, v, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def track_result_to_tensor(result: TrackResult, H: np.ndarray, f0: int, f1: int, T: int = 48,
                           n_players: int = 10) -> np.ndarray:
    """`TrackResult` + hand-clicked H -> (T, 11, 2) court-normalized tensor for the window [f0, f1]."""
    by_id: dict[int, list[Track]] = defaultdict(list)
    for t in result.tracks:
        if f0 <= t.frame <= f1:
            by_id[t.track_id].append(t)
    order = order_tracks(result, f0, f1, n_players)
    times = np.linspace(f0, f1, T)

    out = np.zeros((T, N_ENTITIES, 2), float)                # slot 0 = ball (never filled); absent players = 0
    for slot, tid in enumerate(order, start=1):
        ts = sorted(by_id[tid], key=lambda t: t.frame)
        fs = np.array([t.frame for t in ts], float)
        court = np.array([_project_court(H, (t.xyxy[0] + t.xyxy[2]) / 2.0, t.xyxy[3]) for t in ts], float)
        xs = np.interp(times, fs, court[:, 0])               # linear interp over the track's frames (gaps bridged)
        ys = np.interp(times, fs, court[:, 1])
        out[:, slot, 0] = np.clip(xs / COURT_L, 0.0, 1.0)
        out[:, slot, 1] = np.clip(ys / COURT_W, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------------------------------------
FIXTURE = "tests/fixtures/court_correspondences_v_00HRwkvvjtQ_c007.json"
FRAMES_ROOT = Path("data/sportsmot/val")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="One-clip end-to-end: frames->detect->track->court->tensor->retrieval")
    ap.add_argument("--seq", default="v_00HRwkvvjtQ_c007")
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--checkpoint", required=True, help="a saved encoder (weights/retrieve/*.pt) — needs H6")
    ap.add_argument("--corpus", default="data/sportvu/corpus_g12_T48.npz")
    ap.add_argument("--config", default="configs/v0_finetuned.yaml")
    ap.add_argument("--f0", type=int, default=1)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    import platform
    from datetime import UTC, datetime

    from ..config import load_config
    from ..detect.detector import build_detector
    from ..ingest.frames import load_mot_sequence
    from ..pipeline import Pipeline
    from ..track.tracker import build_tracker
    from .checkpoint import load_checkpoint
    from .train import _faiss_rankings  # noqa: F401  (kept for parity; brute-force below is clearer for top-5)

    H, fx = load_homography(args.fixture)
    reproj = reprojection_error_ft(H, fx)

    # frames -> detect -> track (the REAL shared pipeline; homography/re-ID off — we apply H by hand below)
    cfg = load_config(args.config)
    seq = load_mot_sequence(FRAMES_ROOT / args.seq, max_frames=args.max_frames)
    pipe = Pipeline(cfg=cfg, detector=build_detector(cfg.detect), tracker=build_tracker(cfg.track))
    result = pipe.run(seq)

    f0, f1 = args.f0, args.f0 + args.window - 1
    tensor = track_result_to_tensor(result, H, f0, f1)
    order = order_tracks(result, f0, f1, 10)

    # embed with the saved checkpoint; retrieve against the SportVU gallery (top-5 by cosine)
    emb, ckpt = load_checkpoint(args.checkpoint, device="cpu")
    z = np.load(args.corpus, allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    q = emb.encode_batch(tensor[None])[0]
    gallery = emb.encode_batch(corpus)
    sims = gallery @ q
    top = np.argsort(-sims)[:5]
    top5 = [{"rank": i + 1, "sim": round(float(sims[j]), 4), "idx": int(j), **meta[j]} for i, j in enumerate(top)]

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "10-end-to-end-one-clip",
        "stage": "ONE clip: frames -> detect -> track -> court(H) -> (T,11,2) -> embed(checkpoint) -> retrieval",
        "clip": {"sequence": args.seq, "window_frames": [f0, f1], "n_tracks_in_window": len(order),
                 "n_frames_decoded": len(seq)},
        "crutches": {
            "homography": f"HAND-CLICKED fixture ({args.fixture}); HELD-OUT reprojection error {reproj} ft "
                          "(the 4 fit points reproject at 0 by construction — a tautology); single static H for "
                          "a moving camera; NCAA court vs NBA-trained encoder",
            "reid_substitute": "entity order = (track length desc, first-frame x asc); track IDs are NOT player "
                               "identities -> arbitrary slots == the order corruption the degradation study models",
            "ball": "entity 0 (ball) zeroed — athlete-only detector",
            "gallery": "SportVU 2015-16 (different season/data) — cross-corpus retrieval",
        },
        "homography_reprojection_error_ft": reproj,
        "encoder": {"checkpoint": args.checkpoint, "arch": ckpt.get("arch"),
                    "corpus_fingerprint": ckpt.get("corpus_fingerprint"), "git_sha": ckpt.get("git_sha")},
        "retrieval_top5": top5,
        "notes": "One clip, every crutch explicit. If the neighbours look unrelated, that is the EMPIRICAL "
        "version of the degradation study's prediction (approximate H + no re-ID + no ball corrupt the "
        "reconstruction), and is a more honest result than a clean number.",
        "provenance": {"platform": platform.platform()},
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / f"end2end_oneclip_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"clip={args.seq} window={f0}-{f1} tracks={len(order)}  H reproj={reproj} ft")
    print("top-5 SportVU neighbours (sim | game | eventId):")
    for r in top5:
        print(f"  {r['rank']}. {r['sim']:.4f}  {r['game']}  ev{r['eventId']}")


if __name__ == "__main__":
    main()
