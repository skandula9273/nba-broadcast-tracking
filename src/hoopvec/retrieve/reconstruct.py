"""Adapter: pipeline `Track` output -> a (T, 11, 2) possession tensor the retrieval core consumes.

This is the wire that closes the 'not wired' edge — the reconstruction pipeline (detect -> track ->
[homography -> re-ID]) emits `Track` objects; the retrieval encoder/index consume court-normalized
(T, 11, 2) tensors. `tracks_to_tensor` converts one window, honestly encoding what broadcast reconstruction
can NOT yet supply (the measured blockers, critique #2/#3):

  - NO BALL — the detector is single-class (athletes), so entity 0 (ball) is left zero.
  - NO BROADCAST HOMOGRAPHY — `court_xy` is null off broadcast (the keypoint net doesn't transfer), so
    positions fall back to the box foot-point in IMAGE pixels normalized by frame size: a broadcast-perspective
    plane, NOT the top-down court the SportVU-trained transformer expects. Pass `use_court=True` only when
    homography actually ran.
  - FRAGMENTATION / a variable id set — the tracker emits many (often fragmented) ids, not exactly 10; we keep
    the `n_players` most-present ids in the window and order them canonically by mean x (left-to-right), a
    court-position proxy for canonical player slots. Fragmentation that drops/duplicates a player is real
    reconstruction error and shows up as tensor mismatch — exactly what we want to measure.

The hand-feature FLOOR is coordinate-space-agnostic enough to give a valid reconstructed-vs-GT retrieval number
on these image-coordinate tensors; the trained transformer would need a broadcast-domain retrain (documented).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..pipeline import Track
from .possessions import COURT_L, COURT_W, N_ENTITIES


def _pos(t: Track, use_court: bool) -> tuple[float, float]:
    if use_court and t.court_xy is not None:
        return float(t.court_xy[0]), float(t.court_xy[1])
    x1, _, x2, y2 = t.xyxy
    return (x1 + x2) / 2.0, float(y2)                       # box foot-point (image px)


def _resample(ts: list[Track], times: np.ndarray, use_court: bool) -> np.ndarray:
    """One id's boxes -> (T, 2) sampled at `times` (linear interp over known frames, hold at the ends)."""
    fs = np.array([t.frame for t in ts], float)
    xy = np.array([_pos(t, use_court) for t in ts], float)
    return np.stack([np.interp(times, fs, xy[:, 0]), np.interp(times, fs, xy[:, 1])], axis=1)


def tracks_to_tensor(tracks: list[Track], f0: int, f1: int, T: int, width: int, height: int,
                     n_players: int = 10, use_court: bool = False) -> np.ndarray:
    """`Track` list -> (T, 11, 2) normalized tensor for frames [f0, f1]. Entity 0 (ball) is zero (not tracked);
    the `n_players` most-present ids fill slots 1.., ordered by mean x; absent slots stay zero."""
    by_id: dict[int, list[Track]] = defaultdict(list)
    for t in tracks:
        if f0 <= t.frame <= f1:
            by_id[t.track_id].append(t)

    ids = sorted(by_id, key=lambda i: -len(by_id[i]))[:n_players]     # most-present ids in the window
    times = np.linspace(f0, f1, T)
    trajs = {i: _resample(sorted(by_id[i], key=lambda t: t.frame), times, use_court) for i in ids}
    order = sorted(ids, key=lambda i: float(trajs[i][:, 0].mean()))   # canonical left-to-right slots

    out = np.zeros((T, N_ENTITIES, 2), float)                         # slot 0 ball + absent players = 0
    for slot, i in enumerate(order, start=1):
        out[:, slot, :] = trajs[i]
    sx, sy = (COURT_L, COURT_W) if use_court else (float(width), float(height))
    out[..., 0] = np.clip(out[..., 0] / sx, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] / sy, 0.0, 1.0)
    return out


def frame_windows(tracks: list[Track], window: int, stride: int | None = None) -> list[tuple[int, int]]:
    """Non-overlapping (default) [f0, f1] windows spanning the tracks' frame range."""
    if not tracks:
        return []
    stride = stride or window
    lo = min(t.frame for t in tracks)
    hi = max(t.frame for t in tracks)
    return [(f0, f0 + window - 1) for f0 in range(lo, hi - window + 2, stride)]
