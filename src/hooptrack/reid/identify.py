"""Player identity: appearance re-ID (OSNet) — fills `Track.player_id`. (V1)

What this does: extract an OSNet appearance embedding (boxmot's ReID, `osnet_x0_25_msmt17.pt`) for each
track's boxes, average per track_id, and consolidate track_ids into identity clusters by cosine similarity
(agglomerative / union-find above a threshold). Each track then gets a stable `player_id` = its cluster.

Honest limitation (reported, not hidden — this is the least-reliable stage): OSNet is trained on MSMT17
pedestrians, so it keys on overall appearance. Same-jersey teammates look near-identical, so the clusters are
**appearance / team-level**, not individual jersey identity. Measured on a real basketball sequence, the
per-track OSNet cosines smear **~0.32–0.87 with no clean gap** between same-player and different-player — so
no threshold cleanly separates individuals (0.50 → 1 cluster, 0.70 → 5, 0.80 → 15). Individual identity (jersey
number) needs a scene-text OCR (DBNet/PARSeq/easyocr) — a separate, model-dependent, broadcast-unreliable
piece, deferred. So `player_id` here is a weak appearance cluster, reported as such, not a verified identity.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..pipeline import Track


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def agglomerate(embeddings: np.ndarray, sim_threshold: float) -> np.ndarray:
    """Union-find clustering: merge any two rows with cosine >= `sim_threshold`. Deterministic; returns a
    0..K-1 cluster label per row (rows are assumed L2-normalized, so `E @ E.T` is cosine)."""
    n = len(embeddings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sim = embeddings @ embeddings.T
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= sim_threshold:
                parent[find(i)] = find(j)
    roots: dict[int, int] = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        labels[i] = roots.setdefault(find(i), len(roots))
    return labels


class ReIDIdentifier:
    """Appearance re-ID stage. Implements the pipeline's `ReID` protocol (`identify(tracks, frames)`)."""

    def __init__(self, weights: str = "osnet_x0_25_msmt17.pt", device: str = "cpu",
                 sim_threshold: float = 0.5, jersey_ocr: bool = False, jersey_min_votes: int = 2) -> None:
        self.weights = weights
        self.device = device
        self.sim_threshold = sim_threshold
        self.jersey_ocr = jersey_ocr
        self.jersey_min_votes = jersey_min_votes
        self._model = None

    def _reid(self):
        if self._model is None:
            from boxmot.reid.core.reid import ReID
            self._model = ReID(weights=self.weights, device=self.device)
        return self._model

    def track_embeddings(self, tracks: list[Track], frames) -> dict[int, np.ndarray]:
        """Mean L2-normalized OSNet embedding per track_id (extracted from each frame's boxes)."""
        import cv2

        frame_path = dict(iter(frames))                     # MotSequence -> {frame_idx: path}
        by_frame: dict[int, list[Track]] = defaultdict(list)
        for t in tracks:
            by_frame[t.frame].append(t)

        model = self._reid()
        acc: dict[int, list[np.ndarray]] = defaultdict(list)
        for fidx, ftracks in by_frame.items():
            path = frame_path.get(fidx)
            img = cv2.imread(str(path)) if path is not None else None
            if img is None:
                continue
            boxes = np.asarray([t.xyxy for t in ftracks], dtype=float)
            feats = np.asarray(model(img, boxes=boxes), dtype=np.float32)
            for t, f in zip(ftracks, feats):
                acc[t.track_id].append(f)
        return {tid: _l2(np.mean(fs, axis=0)) for tid, fs in acc.items() if fs}

    def identify(self, tracks: list[Track], frames) -> list[Track]:
        means = self.track_embeddings(tracks, frames)
        if not means:
            return tracks
        tids = sorted(means)
        labels = agglomerate(np.stack([means[t] for t in tids]), self.sim_threshold)
        tid_to_pid = {tid: f"p{labels[i]}" for i, tid in enumerate(tids)}   # appearance cluster
        if self.jersey_ocr:                                                 # overlay jersey numbers where read
            from .jersey import JerseyOCR
            nums = JerseyOCR(min_votes=self.jersey_min_votes, gpu=(self.device == "cuda:0")).read(tracks, frames)
            for tid, num in nums.items():
                tid_to_pid[tid] = f"#{num}"                                  # individual identity for ~40% of tracks
        for t in tracks:
            t.player_id = tid_to_pid.get(t.track_id)          # "#32" (jersey) | "p3" (appearance) | None
        return tracks


def build_reid(cfg):
    """Build the appearance re-ID stage from config, or None (when `reid.enabled` is false)."""
    if not cfg.reid.enabled:
        return None
    return ReIDIdentifier(weights=cfg.track.reid_weights, device=cfg.track.device,
                          sim_threshold=cfg.reid.sim_threshold, jersey_ocr=cfg.reid.jersey_ocr,
                          jersey_min_votes=cfg.reid.jersey_min_votes)
