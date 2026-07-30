"""hooptrack demo — the centerpiece where it works, and the honest limit where it doesn't. (`make demo`)

A scripted, reproducible walkthrough for a reviewer. It runs the REAL retrieval product on the SportVU
ground-truth corpus (NL query -> matching plays -> ranked by the trained encoder, with a CALIBRATED
confidence), then states plainly what does NOT work yet — the broadcast reconstruction path (inc-10: a hand-
clicked homography + no re-ID gives arbitrary neighbours). No new numbers: it composes committed pieces
(nl_query, the checkpoint encoder, the uncertainty calibration) into a five-minute story. Honest by design:
the demo shows the product on the data it works on and names the gap, rather than faking the broadcast demo
that inc-10 proved doesn't work.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .retrieve.checkpoint import load_checkpoint
from .retrieve.nl_query import parse_query, query_corpus

DEMO_QUERIES = ["transition plays", "half court sets on the left", "isolation on the right",
                "fast break with ball movement"]


def _confidence(qi: int, gemb: np.ndarray) -> tuple[float, list[int]]:
    """Top-5 neighbours of corpus item `qi` (excluding itself) + the calibrated confidence (top-1 cosine)."""
    sims = gemb @ gemb[qi]
    sims[qi] = -1e9
    top = np.argsort(-sims)[:5]
    return float(sims[top[0]]), [int(i) for i in top]


def run(args) -> None:
    thr = yaml.safe_load(Path(args.config).read_text())
    z = np.load(args.corpus, allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    emb, ckpt = load_checkpoint(args.checkpoint, device=args.device)
    gemb = emb.encode_batch(corpus)

    print("=" * 78)
    print("hooptrack demo — play retrieval on SportVU ground-truth tracks (the working product)")
    print(f"corpus: {len(corpus)} possessions | encoder: {ckpt.get('arch')} "
          f"(checkpoint git {str(ckpt.get('git_sha'))[:8]})")
    print("=" * 78)

    # 1) Natural-language play query over the validated semantic axes (transition/side/handler).
    print("\n[1] Natural-language play query  (text -> semantic constraints -> matching possessions)\n")
    for q in DEMO_QUERIES:
        r = query_corpus(corpus, thr, q, emb=gemb, k=3)
        print(f"  \"{q}\"")
        print(f"      constraints={parse_query(q)}  matches={r['n_matches']}")
        for i in r["results"]:
            print(f"      -> {meta[i].get('game')}  event {meta[i].get('eventId')}")

    # 2) Similar-play retrieval with a CALIBRATED confidence (the trained encoder + the uncertainty result).
    print("\n[2] Similar-play retrieval with confidence  (trained encoder; top-1 cosine is calibrated,")
    print("    corr 0.74 / ECE 0.06 -> selective prediction: answer the confident half -> ~1.0 accuracy)\n")
    rng = np.random.default_rng(args.seed)
    for qi in rng.choice(len(corpus), size=3, replace=False):
        conf, top = _confidence(int(qi), gemb)
        tag = "HIGH" if conf > 0.7 else "LOW (would abstain)"
        print(f"  query: {meta[qi].get('game')} event {meta[qi].get('eventId')}  confidence={conf:.3f} [{tag}]")
        for i in top[:3]:
            print(f"      -> {meta[i].get('game')}  event {meta[i].get('eventId')}")

    # 3) The honest limit — the broadcast reconstruction path.
    print("\n[3] What does NOT work yet — broadcast reconstruction (honest, measured)")
    print("    inc-10 ran ONE clip end-to-end (frames -> detect -> track -> court -> tensor -> retrieval).")
    print("    The neighbours were NOT meaningful: a hand-clicked homography (~30 ft off) + arbitrary player")
    print("    slots (no re-ID) + no ball = the degradation study's prediction made empirical. The retrieval")
    print("    core is real (above); making it work FROM broadcast needs a homography front-end + re-ID +")
    print("    ball tracking + a broadcast-domain encoder — each measured in its own increment, none faked.")
    print("\n" + "=" * 78)
    print("Run the pieces yourself: make retrieve-semantic-validate | retrieve-uncertainty | retrieve-oneclip")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="hooptrack demo walkthrough")
    ap.add_argument("--checkpoint", default=None, help="saved encoder; default = newest weights/retrieve/*.pt")
    ap.add_argument("--corpus", default="data/sportvu/corpus_g12_T48.npz")
    ap.add_argument("--config", default="configs/semantic_probe.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.checkpoint is None:
        ckpts = sorted(Path("weights/retrieve").glob("*.pt"))
        if not ckpts:
            raise SystemExit("no checkpoint in weights/retrieve/ — run `make retrieve-train` first.")
        args.checkpoint = str(ckpts[-1])
    run(args)


if __name__ == "__main__":
    main()
