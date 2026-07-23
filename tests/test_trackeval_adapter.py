"""Pure-logic tests for the TrackEval adapter — no CV stack, so they run in CI.

`run_hota` imports TrackEval lazily, so importing the module + exercising `write_mot`/`_summarize` needs
none of torch/ultralytics/boxmot/trackeval.
"""

import numpy as np
from pytest import approx

from hooptrack.eval.trackeval_adapter import _scalar, _summarize, write_mot
from hooptrack.pipeline import Track


def test_write_mot_format_and_sorting(tmp_path):
    # deliberately out of order -> must be sorted by (frame, id); xyxy -> left,top,w,h
    tracks = [
        Track(track_id=2, frame=2, cls="player", xyxy=(10, 20, 40, 80), score=0.9),
        Track(track_id=1, frame=1, cls="player", xyxy=(0, 0, 30, 60), score=0.5),
        Track(track_id=1, frame=2, cls="player", xyxy=(11, 21, 41, 81), score=0.7),
    ]
    out = tmp_path / "seq.txt"
    n = write_mot(tracks, out)
    assert n == 3
    lines = out.read_text().strip().splitlines()
    # sorted: (1,1), (2,1), (2,2)
    assert lines[0] == "1,1,0.00,0.00,30.00,60.00,0.5000,-1,-1,-1"
    assert lines[1] == "2,1,11.00,21.00,30.00,60.00,0.7000,-1,-1,-1"
    assert lines[2] == "2,2,10.00,20.00,30.00,60.00,0.9000,-1,-1,-1"


def test_scalar_means_arrays_but_passes_through_floats():
    assert _scalar({"HOTA": np.array([0.4, 0.5, 0.6])}, "HOTA") == 0.5
    assert _scalar({"MOTA": 0.55}, "MOTA") == 0.55
    assert _scalar({"MOTA": 0.55}, "missing") is None


def test_summarize_pulls_combined_and_per_seq():
    output_res = {
        "MotChallenge2DBox": {
            "bytetrack": {
                "COMBINED_SEQ": {  # trackeval 1.0.dev1 key (adapter also accepts legacy COMBINED_SEQS)
                    "pedestrian": {
                        "HOTA": {
                            "HOTA": np.array([0.4, 0.5, 0.6]),  # mean 0.5
                            "DetA": np.array([0.5, 0.5, 0.5]),
                            "AssA": np.array([0.3, 0.4, 0.5]),  # mean 0.4
                            "LocA": np.array([0.8, 0.8, 0.8]),
                        },
                        "CLEAR": {"MOTA": 0.55, "MOTP": 0.77, "IDSW": 12, "Frag": 5},
                        "Identity": {"IDF1": 0.6},
                    }
                },
                "seqA": {
                    "pedestrian": {
                        "HOTA": {"HOTA": np.array([0.4, 0.6])},  # mean 0.5
                        "CLEAR": {"MOTA": 0.5, "MOTP": 0.7, "IDSW": 3, "Frag": 1},
                        "Identity": {"IDF1": 0.55},
                    }
                },
            }
        }
    }
    s = _summarize(output_res, "bytetrack", "pedestrian")
    assert s["HOTA"] == approx(0.5)
    assert s["AssA"] == approx(0.4)
    assert s["MOTA"] == approx(0.55)
    assert s["IDF1"] == approx(0.6)
    assert s["IDSW"] == 12
    assert s["per_seq_HOTA"]["seqA"] == approx(0.5)
