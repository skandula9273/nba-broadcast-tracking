"""Jersey-number OCR — re-ID's reliable-but-sparse individual-identity signal.

Per track, OCR the torso crops (easyocr) and majority-vote the confident 1-2 digit reads. Broadcast reality,
MEASURED on SportsMOT GT-boxed athletes (see `reid.eval_jersey`): a *minority* of tracks yield a confident
majority jersey number (small/blurry/turned-away crops dominate); the rest fall back to appearance re-ID.
Reported honestly — this is coverage (how many tracks get *a* number), not verified accuracy (SportsMOT has no
jersey labels), with cross-frame vote consensus as the precision proxy.

The crop/preprocess knobs (`band`, `upscale`, `preprocess`, `max_crops`) are levers, ablated one at a time in
`reid.eval_jersey`. Defaults below are the measured **winner**: coverage rose 0.175 -> ~0.70 (SportsMOT GT
subset) and the attribution is clean — the win is *evidence, not enhancement*. Two independent evidence levers
(more crops, and sampling them EVENLY across the possession so a camera-facing frame is caught) each roughly
double coverage and combine; CLAHE contrast-normalization consistently **hurt** (it manufactures false
digit-like reads), so `preprocess` is off. Thresholds (`min_conf`/`min_votes`) were held fixed throughout, so
the gain is real evidence, not a lowered bar.
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
    """`read(tracks, frames) -> {track_id: number}` for tracks with a confident majority jersey read.

    Levers (ablated in `reid.eval_jersey`): `band` = torso vertical fraction to crop (jersey region),
    `upscale` = crop magnification before OCR, `preprocess` = grayscale + CLAHE contrast (helps low-contrast
    digits), `max_crops` = frames sampled per track (more evidence -> more tracks clear `min_votes`)."""

    def __init__(self, min_conf: float = 0.4, min_votes: int = 2, max_crops: int = 40,
                 band: tuple[float, float] = (0.10, 0.52), upscale: int = 4, preprocess: bool = False,
                 stride_sample: bool = True, gpu: bool = False) -> None:
        self.min_conf = min_conf
        self.min_votes = min_votes
        self.max_crops = max_crops
        self.band = band
        self.upscale = upscale
        self.preprocess = preprocess
        self.stride_sample = stride_sample     # sample crops EVENLY across the track vs the first max_crops
        self.gpu = gpu
        self._reader = None

    def _r(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
        return self._reader

    def _prep(self, crop):
        """Optional contrast normalization + upscale — make small/low-contrast digits legible to easyocr."""
        import cv2

        if self.preprocess:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.resize(crop, (crop.shape[1] * self.upscale, crop.shape[0] * self.upscale),
                          interpolation=cv2.INTER_CUBIC)

    def _crop_reads(self, reader, img, box) -> list[str] | None:
        """Confident 1-2 digit reads from one track box's torso crop; None if the crop is unusable."""
        x1, y1, x2, y2 = (int(v) for v in box)
        bh = y2 - y1
        lo, hi = self.band
        crop = img[max(0, y1 + int(lo * bh)): y1 + int(hi * bh), max(0, x1):x2]      # torso / jersey region
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        crop = self._prep(crop)
        return [txt for _, txt, c in reader.readtext(crop, allowlist="0123456789")
                if _DIGITS.fullmatch(txt) and c >= self.min_conf]

    def read_detail(self, tracks, frames) -> dict[int, dict]:
        """Per track: `{reads, number, n_crops, n_read_crops}`. `number` is the confident majority (or None).
        `n_read_crops` / `n_crops` = per-crop read rate; votes vs total reads = cross-frame consensus."""
        import cv2

        paths = dict(iter(frames))
        by_track: dict[int, list] = defaultdict(list)
        for t in tracks:
            by_track[t.track_id].append(t)

        reader = self._r()
        out: dict[int, dict] = {}
        for tid, ts in by_track.items():
            ts = sorted(ts, key=lambda t: t.frame)
            if self.stride_sample and len(ts) > self.max_crops:      # spread crops across the whole track
                step = len(ts) / self.max_crops
                ts = [ts[int(i * step)] for i in range(self.max_crops)]
            else:
                ts = ts[: self.max_crops]
            reads: list[str] = []
            n_crops = n_read = 0
            for t in ts:
                p = paths.get(t.frame)
                img = cv2.imread(str(p)) if p is not None else None
                if img is None:
                    continue
                n_crops += 1
                r = self._crop_reads(reader, img, t.xyxy)
                if r:
                    n_read += 1
                    reads.extend(r)
            out[tid] = {"reads": reads, "number": majority_number(reads, self.min_votes),
                        "n_crops": n_crops, "n_read_crops": n_read}
        return out

    def read(self, tracks, frames) -> dict[int, str]:
        return {tid: d["number"] for tid, d in self.read_detail(tracks, frames).items() if d["number"]}
