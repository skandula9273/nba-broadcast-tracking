"""Jersey-number OCR — re-ID's reliable-but-sparse individual-identity signal.

Per track, OCR the torso crops (easyocr) and majority-vote the confident 1-2 digit reads. Broadcast reality,
MEASURED on SportsMOT: ~12% of single crops yield a number (small/blurry/turned-away), but aggregating over a
track's frames lifts it to **~40% of tracks** getting a plausible jersey number (e.g. 32, 30, 20). So jersey
identity is available for a *minority* of tracks; the rest fall back to appearance re-ID. Reported honestly.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

_DIGITS = re.compile(r"[0-9]{1,2}")


def majority_number(reads: list[str], min_votes: int) -> str | None:
    """Majority digit-string across a track's reads, if it has >= min_votes; else None (honest abstain)."""
    if not reads:
        return None
    num, votes = Counter(reads).most_common(1)[0]
    return num if votes >= min_votes else None


class JerseyOCR:
    """`read(tracks, frames) -> {track_id: number}` for tracks with a confident majority jersey read."""

    def __init__(self, min_conf: float = 0.4, min_votes: int = 2, max_crops: int = 15, gpu: bool = False) -> None:
        self.min_conf = min_conf
        self.min_votes = min_votes
        self.max_crops = max_crops
        self.gpu = gpu
        self._reader = None

    def _r(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
        return self._reader

    def read(self, tracks, frames) -> dict[int, str]:
        import cv2

        paths = dict(iter(frames))
        by_track: dict[int, list] = defaultdict(list)
        for t in tracks:
            by_track[t.track_id].append(t)

        reader = self._r()
        out: dict[int, str] = {}
        for tid, ts in by_track.items():
            reads: list[str] = []
            for t in ts[: self.max_crops]:
                p = paths.get(t.frame)
                img = cv2.imread(str(p)) if p is not None else None
                if img is None:
                    continue
                x1, y1, x2, y2 = (int(v) for v in t.xyxy)
                bh = y2 - y1
                crop = img[max(0, y1 + int(0.12 * bh)): y1 + int(0.55 * bh), x1:x2]   # torso / jersey region
                if crop.size == 0 or min(crop.shape[:2]) < 8:
                    continue
                crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3))       # upscale small crops
                for _, txt, c in reader.readtext(crop, allowlist="0123456789"):
                    if _DIGITS.fullmatch(txt) and c >= self.min_conf:
                        reads.append(txt)
            num = majority_number(reads, self.min_votes)
            if num is not None:
                out[tid] = num
        return out
