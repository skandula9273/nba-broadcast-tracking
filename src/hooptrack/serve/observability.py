"""Serving observability — real per-request metrics, latency percentiles, and a drift signal. (V2)

Not a dashboard mock: an in-memory registry the `/track` handler feeds, exposed at `/metrics`. Records each
request's per-stage wall-clock (from `Pipeline` meta['timings']), throughput (frames, fps), and detector
signal (detections/frame). `drift()` flags when a request's detections-per-frame deviates from the running
baseline (a Welford mean/std) by more than `z_thresh` sigma — the cheap, honest proxy for input-distribution
shift (a broadcast that suddenly detects far fewer/more athletes than the norm is off-distribution).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return round(sorted_vals[i], 4)


@dataclass
class Metrics:
    """Rolling serving metrics. `window` caps the retained per-request records (memory-bounded)."""

    window: int = 500
    requests: int = 0
    frames_total: int = 0
    _records: deque = field(default_factory=lambda: deque(maxlen=500))
    # Welford running stats for detections-per-frame (the drift baseline)
    _dpf_n: int = 0
    _dpf_mean: float = 0.0
    _dpf_m2: float = 0.0

    def record(self, n_frames: int, timings: dict, n_tracks: int, n_dets: int) -> dict:
        """Log one /track request; return its drift verdict (computed vs the baseline BEFORE this update)."""
        self.requests += 1
        self.frames_total += n_frames
        total_s = sum(timings.values())
        dpf = n_dets / max(1, n_frames)
        verdict = self.drift(dpf)                            # judge against the prior baseline first
        self._records.append({"n_frames": n_frames, "total_s": round(total_s, 4),
                              "fps": round(n_frames / total_s, 2) if total_s else None,
                              "detect_s": timings.get("detect_s", 0.0), "track_s": timings.get("track_s", 0.0),
                              "n_tracks": n_tracks, "dets_per_frame": round(dpf, 2)})
        self._dpf_n += 1                                     # then fold it into the baseline (Welford)
        d = dpf - self._dpf_mean
        self._dpf_mean += d / self._dpf_n
        self._dpf_m2 += d * (dpf - self._dpf_mean)
        return verdict

    def drift(self, dets_per_frame: float, z_thresh: float = 3.0) -> dict:
        """z-score of this request's detections/frame vs the running baseline; flag beyond `z_thresh` sigma."""
        if self._dpf_n < 5:                                 # not enough history to judge
            return {"status": "baseline", "n_seen": self._dpf_n}
        std = math.sqrt(self._dpf_m2 / max(1, self._dpf_n - 1))
        dev = dets_per_frame - self._dpf_mean
        if std > 1e-9:
            z = dev / std
        else:                                               # zero-variance baseline: any real deviation is drift
            z = 0.0 if abs(dev) < 1e-6 else math.copysign(99.99, dev)
        return {"status": "drift" if abs(z) > z_thresh else "ok", "z": round(z, 2),
                "dets_per_frame": round(dets_per_frame, 2), "baseline_mean": round(self._dpf_mean, 2)}

    def summary(self) -> dict:
        recs = list(self._records)
        lat = sorted(r["total_s"] for r in recs)
        det = sorted(r["detect_s"] for r in recs)
        std = math.sqrt(self._dpf_m2 / max(1, self._dpf_n - 1)) if self._dpf_n > 1 else 0.0
        return {
            "requests": self.requests, "frames_total": self.frames_total, "window": self.window,
            "latency_s": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95), "max": round(max(lat), 4) if lat else 0.0},
            "detect_latency_s": {"p50": _pct(det, 0.5), "p95": _pct(det, 0.95)},
            "throughput_fps": {"mean": round(sum(r["fps"] or 0 for r in recs) / len(recs), 2) if recs else None},
            "detections_per_frame_baseline": {"mean": round(self._dpf_mean, 2), "std": round(std, 2),
                                              "n": self._dpf_n},
            "recent": recs[-5:],
        }
