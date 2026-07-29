"""Natural-language play query — a structured NL layer over the VALIDATED semantic dimensions.

This is honest scaffolding for the V2 'NL query' upside, and deliberately NOT a fake LLM. The critique was that
an NL query over an embedding with no semantics is a demo, not a feature — so this is grounded in the play-type
axes that `semantic_validate` showed are actually encodable (transition/half-court, initiation side, ball-
handler change). `parse_query` maps free text to `{scheme: bucket}` constraints by keyword; `query_corpus`
labels the corpus with the same derived-label functions the probe uses, filters to possessions matching ALL
constraints, and (optionally) ranks them by embedding similarity to the matched set's centroid.

Scope, stated: a keyword parser over three coarse derived axes, not a learned language model and not designed
set-plays. It demonstrates the text -> semantic-retrieval path over dimensions that are measured, not assumed.
A real product swaps the keyword map for a learned text encoder aligned to the (supervised) play embedding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .semantic_probe import SCHEMES, label_corpus

# keyword -> (scheme, bucket). First match per scheme wins; multiple schemes compose (AND).
_KEYWORDS: list[tuple[tuple[str, ...], str, str]] = [
    (("transition", "fast break", "fastbreak", "fast-break", "run"), "transition", "transition"),
    (("half court", "halfcourt", "half-court", "set offense", "set play", "settled"), "transition", "halfcourt"),
    (("left",), "initiation_side", "left"),
    (("right",), "initiation_side", "right"),
    (("middle", "center", "top of the key", "top"), "initiation_side", "middle"),
    (("handoff", "hand-off", "ball movement", "passing", "many handlers", "moves the ball"),
     "handler_change", "handler_high"),
    (("iso", "isolation", "one-on-one", "one handler", "hero ball"), "handler_change", "handler_low"),
]


def parse_query(text: str) -> dict[str, str]:
    """Free text -> {scheme: bucket} constraints (keyword match; one bucket per scheme, first hit wins)."""
    t = text.lower()
    out: dict[str, str] = {}
    for phrases, scheme, bucket in _KEYWORDS:
        if scheme in out:
            continue
        if any(p in t for p in phrases):
            out[scheme] = bucket
    return out


def query_corpus(corpus: np.ndarray, thr: dict, text: str, emb: np.ndarray | None = None, k: int = 10) -> dict:
    """Return up to k corpus indices whose derived play-type labels satisfy ALL parsed constraints. If `emb`
    (L2-normalized embeddings) is given, rank the matches by cosine similarity to the matched set's centroid
    (a 'most representative first' order); else return them in corpus order."""
    constraints = parse_query(text)
    if not constraints:
        return {"query": text, "constraints": {}, "n_matches": 0, "results": [],
                "note": "no known play-type keyword matched; try transition/halfcourt, left/right/middle, iso/handoff"}

    mask = np.ones(len(corpus), dtype=bool)
    for scheme, bucket in constraints.items():
        fn, _order = SCHEMES[scheme]
        mask &= label_corpus(corpus, fn, thr) == bucket
    idx = np.where(mask)[0]

    if emb is not None and len(idx) > 0:
        centroid = emb[idx].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        idx = idx[np.argsort(-(emb[idx] @ centroid))]           # most representative of the matched set first

    return {"query": text, "constraints": constraints, "n_matches": int(len(idx)),
            "results": [int(i) for i in idx[:k]],
            "note": "keyword parse over derived semantic axes (validated encodable in semantic_validate); "
            "not a learned LM"}


_DEMO_QUERIES = ["transition plays", "half court sets on the left",
                 "isolation on the right", "fast break with ball movement"]


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser(description="NL play query over the SportVU corpus (structured, no LLM)")
    ap.add_argument("--corpus", default="data/sportvu/corpus_g12_T48.npz")
    ap.add_argument("--config", default="configs/semantic_probe.yaml")
    ap.add_argument("--query", default=None, help="a single query; default runs a demo set")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    thr = yaml.safe_load(Path(args.config).read_text())
    z = np.load(Path(args.corpus), allow_pickle=True)
    corpus, meta = z["corpus"].astype(np.float32), list(z["meta"])
    queries = [args.query] if args.query else _DEMO_QUERIES
    for q in queries:
        r = query_corpus(corpus, thr, q, k=args.k)
        hits = [{"idx": i, **meta[i]} for i in r["results"]]
        print(f"\n'{q}'  -> constraints {r['constraints']}  matches={r['n_matches']}")
        print("  top:", json.dumps(hits, default=str))


if __name__ == "__main__":
    main()

