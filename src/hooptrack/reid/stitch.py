"""Fragment stitching — the lever to recover jersey coverage lost to tracker fragmentation.

The jersey-OCR operating point measured the disease (`reid.eval_jersey --source tracker`): the tracker splits
~10 real players into ~60 short track-ids per sequence, so most ids never accumulate `min_votes` and per-id
coverage collapses 0.73 -> 0.37 — while the per-crop read rate is unchanged (the OCR is fine; the *tracks* are
fragmented). The cheap fix that doesn't retrain the tracker: **stitch a player's fragments back together** so
their jersey votes pool.

Method — spatiotemporal gap-closing (classic tracklet linking): fragment B inherits fragment A's id when B
*starts* shortly after A *ends* (gap in [1, `max_gap`] frames) and B's first box is spatially near A's last box
(<= `max_dist_factor` x A's box diagonal). Greedy earliest-first, one successor per predecessor, union-find.

Why this is safe (won't merge teammates, unlike appearance re-ID which can't separate same-uniform players):
it only links fragments that are **temporally disjoint** (gap >= 1, so never co-present) and **spatially
adjacent at the seam** — two different players on court at the hand-off moment are elsewhere, so the nearest
endpoint is the same player continuing. It's a heuristic, not identity truth; measured, not assumed.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..pipeline import Track


def _center_size(xyxy) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]), float(np.hypot(x2 - x1, y2 - y1))


def stitch_fragments(tracks: list[Track], max_gap: int = 30, max_dist_factor: float = 2.0) -> list[Track]:
    """Rewrite `track_id` so fragments of the same player share an id (canonical = smallest original id in the
    merged group). `max_gap` in frames, `max_dist_factor` in units of the predecessor's box diagonal."""
    by_id: dict[int, list[Track]] = defaultdict(list)
    for t in tracks:
        by_id[t.track_id].append(t)

    frag: dict[int, dict] = {}
    for tid, ts in by_id.items():
        ts_sorted = sorted(ts, key=lambda t: t.frame)
        first_c, _ = _center_size(ts_sorted[0].xyxy)
        last_c, size = _center_size(ts_sorted[-1].xyxy)
        frag[tid] = {"start": ts_sorted[0].frame, "end": ts_sorted[-1].frame,
                     "first": first_c, "last": last_c, "size": size}

    parent = {tid: tid for tid in frag}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used_pred: set[int] = set()                             # a fragment can be extended by at most one successor
    for b in sorted(frag, key=lambda tid: frag[tid]["start"]):
        fb = frag[b]
        best, best_d = None, None
        for a in frag:
            if a == b or a in used_pred or find(a) == find(b):
                continue
            fa = frag[a]
            gap = fb["start"] - fa["end"]                   # B starts after A ends (temporally disjoint)
            if gap < 1 or gap > max_gap:
                continue
            d = float(np.linalg.norm(fa["last"] - fb["first"]))
            if d <= max_dist_factor * fa["size"] and (best_d is None or d < best_d):
                best, best_d = a, d
        if best is not None:
            parent[find(b)] = find(best)
            used_pred.add(best)

    groups: dict[int, list[int]] = defaultdict(list)
    for tid in frag:
        groups[find(tid)].append(tid)
    remap = {m: min(members) for members in groups.values() for m in members}

    return [Track(track_id=remap[t.track_id], frame=t.frame, cls=t.cls, xyxy=t.xyxy,
                  score=t.score, court_xy=t.court_xy, player_id=t.player_id) for t in tracks]
