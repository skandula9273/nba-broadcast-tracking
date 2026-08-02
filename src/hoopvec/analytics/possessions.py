"""Analytics on the reconstructed tracks — what's HONESTLY computable from player-only tracks.

The pipeline detects athletes (a single class), NOT the ball, so true possessions / shots (which need ball
control) are not computable here — stated plainly. What IS: player-configuration analytics — team **spacing**
(spread over time) and a ball-free **phase** segmentation (transition vs halfcourt by the players' collective
advance). In court units when homography ran (`court_xy`), else image px. Feeds the serve response.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..pipeline import TrackResult


def player_points(result: TrackResult) -> dict[int, list[tuple[float, float]]]:
    """Per-frame player positions: `court_xy` if homography ran, else the box foot point (image px)."""
    by_frame: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for t in result.tracks:
        if t.court_xy is not None:
            by_frame[t.frame].append((float(t.court_xy[0]), float(t.court_xy[1])))
        else:
            x1, _, x2, y2 = t.xyxy
            by_frame[t.frame].append(((x1 + x2) / 2.0, float(y2)))
    return dict(by_frame)


def _coords(result: TrackResult) -> str:
    return "court_cm" if any(t.court_xy is not None for t in result.tracks) else "image_px"


def spacing(result: TrackResult) -> dict:
    """Per-frame team spacing = mean pairwise distance between the tracked players (spread)."""
    series: dict[int, float] = {}
    for f, pts in player_points(result).items():
        a = np.asarray(pts, float)
        if len(a) < 2:
            continue
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
        series[f] = float(d[np.triu_indices(len(a), 1)].mean())
    vals = list(series.values())
    return {"coords": _coords(result), "n_frames": len(vals),
            "mean_spread": round(float(np.mean(vals)), 1) if vals else None,
            "per_frame": {int(f): round(v, 1) for f, v in sorted(series.items())}}


def segment_phases(result: TrackResult, min_run: int = 5) -> list[dict]:
    """Ball-free phase segmentation: label each frame transition/halfcourt by whether the player CENTROID is
    moving fast (above the median centroid speed), then merge consecutive frames into runs. A proxy for
    possession phases WITHOUT the ball."""
    pts = player_points(result)
    fs = sorted(f for f in pts if pts[f])
    cent = {f: np.asarray(pts[f], float).mean(axis=0) for f in fs}
    speeds = {fs[i]: float(np.linalg.norm(cent[fs[i]] - cent[fs[i - 1]])) for i in range(1, len(fs))}
    if not speeds:
        return []
    thr = float(np.median(list(speeds.values())))
    runs: list[dict] = []
    for f in fs:
        lab = "transition" if speeds.get(f, 0.0) > thr else "halfcourt"
        if runs and runs[-1]["phase"] == lab:
            runs[-1]["end"] = f
        else:
            runs.append({"phase": lab, "start": f, "end": f})
    merged = [r for r in runs if r["end"] - r["start"] + 1 >= min_run]
    return merged or runs


def ball_handler_timeline(result: TrackResult, ball_by_frame: dict) -> dict[int, int]:
    """Per frame with a ball: the track_id of the nearest player (image-coord box center vs ball center) = the
    ball-handler. `ball_by_frame` = {frame: (cx, cy, conf)} from `detect.ball.BallDetector`."""
    by_frame: dict[int, list] = defaultdict(list)
    for t in result.tracks:
        x1, y1, x2, y2 = t.xyxy
        by_frame[t.frame].append((t.track_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    timeline: dict[int, int] = {}
    for f, (bx, by, _conf) in ball_by_frame.items():
        players = by_frame.get(f)
        if players:
            timeline[f] = int(min(players, key=lambda p: (p[1] - bx) ** 2 + (p[2] - by) ** 2)[0])
    return timeline


def ball_possession(result: TrackResult, ball_by_frame: dict, min_run: int = 5, max_gap: int = 10) -> dict:
    """TRUE possession segmentation where the ball is visible: runs of frames with a stable ball-handler; a
    handler change is a pass/turnover. Coverage = fraction of frames with a ball (COCO sports-ball; no ball GT,
    so coverage not accuracy). Shots still need a hoop/trajectory model — stated, not faked."""
    timeline = ball_handler_timeline(result, ball_by_frame)
    fs = sorted(timeline)
    runs: list[dict] = []
    for f in fs:
        h = timeline[f]
        if runs and runs[-1]["handler"] == h and f - runs[-1]["end"] <= max_gap:
            runs[-1]["end"] = f
        else:
            runs.append({"handler": h, "start": f, "end": f})
    n_total = len({t.frame for t in result.tracks})
    changes = sum(1 for i in range(1, len(fs)) if timeline[fs[i]] != timeline[fs[i - 1]])
    return {"source": "ball (COCO sports-ball) -> nearest-player handler; coverage not accuracy (no ball GT); "
            "shots need a hoop model",
            "ball_frames": len(timeline), "frames_total": n_total,
            "ball_coverage": round(len(timeline) / max(1, n_total), 3),
            "n_handler_changes": changes, "possessions": [r for r in runs if r["end"] - r["start"] + 1 >= min_run] or runs}


def segment_possessions(result: TrackResult, ball_by_frame: dict | None = None) -> dict:
    """With a ball -> TRUE possessions (ball-handler runs). Without -> the ball-free PHASE segmentation, stated
    honestly as a proxy."""
    if ball_by_frame:
        return ball_possession(result, ball_by_frame)
    return {"note": "ball not tracked (player-only detector) -> phases, NOT true ball possessions",
            "coords": _coords(result), "phases": segment_phases(result)}


def analytics(result: TrackResult, ball_by_frame: dict | None = None) -> dict:
    """Player-config analytics (spacing + phases), upgraded to TRUE possessions when a ball track is supplied."""
    return {"n_tracks": len({t.track_id for t in result.tracks}), "spacing": spacing(result),
            "possessions": segment_possessions(result, ball_by_frame)}
