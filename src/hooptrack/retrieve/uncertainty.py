"""End-to-end retrieval uncertainty — is the platform's confidence signal TRUSTWORTHY? (V2)

A retrieval system is only useful in production if it can say "I'm not sure" — so this measures whether a
cheap, model-free confidence signal (the **margin** between the top-1 and top-2 cosine similarities) is
CALIBRATED: does higher confidence actually mean higher correctness? If so, the pipeline can do **selective
prediction** — answer when confident, abstain when not — which is the concrete end-to-end-uncertainty upside.

Setup reuses the whole retrieval harness with a KNOWN correctness label: query = an augmented/degraded copy of a
val possession, gallery = the clean val corpus, correct@1 = top-1 is the query's own source. We sweep a range
of corruption (clean augmentations through id-swaps, from the degradation study) so confidence varies, then
measure calibration + the accuracy-vs-coverage tradeoff. Encoder = a saved checkpoint (ties to the same model
the retrieval/degradation artifacts describe).
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint
from .degrade import id_swap
from .possessions import augment
from .train import _ver, split_by_game


def _retrieve_with_confidence(qemb: np.ndarray, gemb: np.ndarray, relevant: np.ndarray) -> dict:
    """Per query: correct@1 + two confidence signals (top-1 sim, top1-top2 margin). `relevant[i]` = gallery
    index that is query i's own source."""
    sims = qemb @ gemb.T
    order = np.argsort(-sims, axis=1)
    top1 = order[:, 0]
    sim1 = sims[np.arange(len(sims)), order[:, 0]]
    sim2 = sims[np.arange(len(sims)), order[:, 1]]
    return {"correct": (top1 == relevant).astype(int), "sim1": sim1, "margin": sim1 - sim2}


def _calibration(correct: np.ndarray, conf: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability: bin by confidence, report accuracy per bin + expected calibration error (min-max scaled
    conf as the 'probability' proxy) + the confidence/correctness correlation."""
    c = (conf - conf.min()) / (conf.max() - conf.min() + 1e-9)          # scale to [0,1] as a prob proxy
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins) - 1, 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        acc, mean_conf, w = float(correct[m].mean()), float(c[m].mean()), m.mean()
        rows.append({"bin": b, "n": int(m.sum()), "mean_conf": round(mean_conf, 3), "accuracy": round(acc, 3)})
        ece += w * abs(acc - mean_conf)
    corr = float(np.corrcoef(conf, correct)[0, 1]) if conf.std() > 0 else 0.0
    return {"bins": rows, "ece": round(ece, 4), "confidence_correctness_corr": round(corr, 4)}


def _selective(correct: np.ndarray, conf: np.ndarray, coverages=(1.0, 0.75, 0.5, 0.25)) -> list[dict]:
    """Accuracy-vs-coverage: answer only the most-confident `coverage` fraction; report accuracy there. A
    calibrated signal makes accuracy RISE as coverage falls (abstaining on the uncertain ones)."""
    order = np.argsort(-conf)                                            # most confident first
    out = []
    for cov in coverages:
        k = max(1, int(cov * len(conf)))
        out.append({"coverage": cov, "n": k, "accuracy": round(float(correct[order[:k]].mean()), 3)})
    return out


def run(args) -> dict:
    emb, ckpt = load_checkpoint(args.checkpoint, device=args.device)
    z = np.load(args.corpus, allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    _, va, _, val_games = split_by_game(meta, args.val_stride, args.val_offset)
    val = corpus[va]
    gemb = emb.encode_batch(val)
    n = len(val)
    rng = np.random.default_rng(args.seed)

    # queries spanning a range of corruption so confidence varies: clean augmentations + id-swaps
    q_sources, relevant = [], []
    for kind in ["jitter", "crop", "mirror"]:
        q_sources += [augment(val[i], rng, kind) for i in range(n)]
        relevant += list(range(n))
    for nsw in (1, 2, 4):
        q_sources += [id_swap(val[i], rng, nsw) for i in range(n)]
        relevant += list(range(n))
    qemb = emb.encode_batch(np.stack(q_sources))
    relevant = np.array(relevant)

    res = _retrieve_with_confidence(qemb, gemb, relevant)
    overall_acc = round(float(res["correct"].mean()), 4)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "increment": "v2-retrieval-uncertainty",
        "stage": "is the retrieval confidence signal calibrated? (selective prediction — end-to-end uncertainty)",
        "encoder": {"checkpoint": args.checkpoint, "arch": ckpt.get("arch"),
                    "corpus_fingerprint": ckpt.get("corpus_fingerprint"), "git_sha": ckpt.get("git_sha")},
        "dataset": {"n_gallery": n, "n_queries": len(relevant), "val_games": val_games,
                    "queries": "jitter/crop/mirror augmentations + id_swap{1,2,4} — a corruption sweep"},
        "overall_accuracy@1": overall_acc,
        "confidence_signals": {
            "margin": {"calibration": _calibration(res["correct"], res["margin"]),
                       "selective_accuracy_vs_coverage": _selective(res["correct"], res["margin"])},
            "top1_similarity": {"calibration": _calibration(res["correct"], res["sim1"]),
                                "selective_accuracy_vs_coverage": _selective(res["correct"], res["sim1"])},
        },
        "notes": "Confidence = top1-top2 margin (and top-1 sim), model-free. A positive confidence/correctness "
        "correlation + accuracy RISING as coverage falls means the signal is trustworthy -> the pipeline can "
        "abstain when unsure. Correctness label = top-1 is the query's own source (augmentation-SSL relevant).",
        "provenance": {"seed": args.seed, "device": args.device,
                       "versions": {p: _ver(p) for p in ("torch", "numpy")}, "platform": platform.platform()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval confidence calibration / selective prediction")
    ap.add_argument("--checkpoint", required=True, help="saved encoder (weights/retrieve/*.pt)")
    ap.add_argument("--corpus", default="data/sportvu/corpus_g12_T48.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--val-offset", type=int, default=2)
    ap.add_argument("--out-dir", default="eval_results")
    args = ap.parse_args()

    report = run(args)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (Path(args.out_dir) / f"uncertainty_{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"wrote uncertainty_{stamp}.json | overall acc@1={report['overall_accuracy@1']} "
          f"n_queries={report['dataset']['n_queries']}")
    for sig, d in report["confidence_signals"].items():
        cal = d["calibration"]
        print(f"\n[{sig}] corr(conf,correct)={cal['confidence_correctness_corr']}  ECE={cal['ece']}")
        print("  selective acc @ coverage:", {s["coverage"]: s["accuracy"]
                                              for s in d["selective_accuracy_vs_coverage"]})


if __name__ == "__main__":
    main()
