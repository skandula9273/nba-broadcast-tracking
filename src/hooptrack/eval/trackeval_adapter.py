"""Adapter to TrackEval for HOTA / MOTA / IDF1 on SportsMOT (V0).

TrackEval (github.com/JonathonLuiten/TrackEval) is the field-standard MOT metric toolkit. The flow:
  1. `write_mot` serialises tracks to MOT-challenge format (one txt per sequence);
  2. the GT tree + seqmap are laid out by `ingest/fetch.py` (the authors' sportsmot_to_trackeval layout);
  3. `run_hota` runs TrackEval's MotChallenge2DBox eval with HOTA + CLEAR + Identity metrics;
  4. `_summarize` parses the returned result dict into the numbers we commit to eval_results/.

Kept as an adapter so the rest of the harness never imports TrackEval directly. TrackEval is imported
**lazily inside `run_hota`**, so `write_mot`/`_summarize` (pure logic) import with no CV stack — that's
what keeps the unit test and CI green. APIs below are verified against the installed TrackEval
(MotChallenge2DBox config keys, the row-0 seqmap header, GT class column 7) — rule #1.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _restore_numpy_aliases() -> None:
    """TrackEval 1.0.dev1 predates NumPy's removal (1.24) of np.float/np.int/np.bool.

    NumPy 2.x still lets you *set* these attributes (only attribute *access* was removed), so we restore
    the exact builtin aliases TrackEval expects. Behavior-preserving — np.float was always just float —
    and it keeps the vendored TrackEval code running unmodified (rule #1: make the real API work).
    """
    for name, builtin in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, builtin)


_restore_numpy_aliases()


def write_mot(tracks: Iterable, out_path: str | Path) -> int:
    """Serialize tracks to MOT-challenge format for TrackEval. Returns the number of lines written.

    One line per track box: `frame,id,bb_left,bb_top,bb_w,bb_h,conf,-1,-1,-1` (frames are 1-indexed).
    `tracks` items must expose `.frame`, `.track_id`, `.xyxy = (x1,y1,x2,y2)`, and `.score`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in tracks:
        x1, y1, x2, y2 = t.xyxy
        rows.append(
            (int(t.frame), int(t.track_id), float(x1), float(y1), float(x2 - x1), float(y2 - y1),
             float(getattr(t, "score", 1.0)))
        )
    rows.sort(key=lambda r: (r[0], r[1]))  # by frame then id (MOTChallenge convention)
    with open(out_path, "w") as f:
        for fr, tid, x, y, w, h, c in rows:
            f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c:.4f},-1,-1,-1\n")
    return len(rows)


def _scalar(d: dict, key: str) -> float | None:
    """TrackEval reports HOTA/DetA/AssA as arrays over alpha thresholds; a scalar is their mean."""
    if key not in d:
        return None
    v = d[key]
    return float(np.mean(v)) if hasattr(v, "__len__") else float(v)


def _summarize(output_res: dict, tracker_name: str, cls: str = "pedestrian") -> dict:
    """Dig the combined + per-sequence metrics out of TrackEval's nested result dict."""
    dataset = next(v for v in output_res.values() if tracker_name in v)  # keyed by dataset class name
    per_tracker = dataset[tracker_name]
    # combined-over-sequences key is 'COMBINED_SEQ' in trackeval 1.0.dev1 (older forks: 'COMBINED_SEQS')
    combined_key = next((k for k in ("COMBINED_SEQ", "COMBINED_SEQS") if k in per_tracker), None)
    if combined_key is None:
        raise KeyError(f"no combined-sequences key in TrackEval result: {list(per_tracker)}")
    combined = per_tracker[combined_key][cls]
    hota, clear, ident = combined["HOTA"], combined["CLEAR"], combined["Identity"]
    summary = {
        "HOTA": _scalar(hota, "HOTA"),
        "DetA": _scalar(hota, "DetA"),
        "AssA": _scalar(hota, "AssA"),
        "LocA": _scalar(hota, "LocA"),
        "MOTA": _scalar(clear, "MOTA"),
        "MOTP": _scalar(clear, "MOTP"),
        "IDF1": _scalar(ident, "IDF1"),
        "IDSW": _scalar(clear, "IDSW"),
        "Frag": _scalar(clear, "Frag"),
    }
    per_seq = {
        seq: _scalar(res[cls]["HOTA"], "HOTA")
        for seq, res in per_tracker.items()
        if seq != combined_key
    }
    summary["per_seq_HOTA"] = dict(sorted(per_seq.items()))
    return summary


def run_hota(
    gt_folder: str | Path,
    trackers_folder: str | Path,
    benchmark: str,
    split: str,
    tracker_name: str,
    seqmap_file: str | Path | None = None,
    classes_to_eval: tuple[str, ...] = ("pedestrian",),
    do_preproc: bool = False,
) -> dict:
    """Run TrackEval MotChallenge2DBox (HOTA + CLEAR + Identity) and return the parsed summary.

    GT tree is `gt_folder/<benchmark>-<split>/<seq>/{gt/gt.txt,seqinfo.ini}`; tracker outputs are
    `trackers_folder/<benchmark>-<split>/<tracker_name>/data/<seq>.txt`. `do_preproc=False` for SportsMOT
    (no distractor classes — the authors' comparable setting).
    """
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
        }
    )
    ds_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_config.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_folder),
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": split,
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": list(classes_to_eval),
            "DO_PREPROC": do_preproc,
            "PRINT_CONFIG": False,
            "SEQMAP_FILE": str(seqmap_file) if seqmap_file else None,
        }
    )
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(ds_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
    return _summarize(output_res, tracker_name, classes_to_eval[0])
