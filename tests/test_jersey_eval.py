"""Jersey-OCR coverage aggregation — pure metric logic, no easyocr/data, CI-safe."""

from hooptrack.reid.eval_jersey import aggregate


def _d(reads, number, n_crops, n_read):
    return {"reads": reads, "number": number, "n_crops": n_crops, "n_read_crops": n_read}


def test_coverage_counts_only_tracks_with_a_number():
    details = [
        _d(["32", "32", "32"], "32", 5, 3),   # covered, unanimous
        _d(["7"], None, 4, 1),                 # a read but no majority -> abstain, not covered
        _d([], None, 6, 0),                    # nothing read
    ]
    r = aggregate(details)
    assert r["n_tracks"] == 3
    assert r["track_coverage"] == round(1 / 3, 3)
    assert r["crop_read_rate"] == round(4 / 15, 3)      # (3+1+0) reads / (5+4+6) crops
    assert r["numbers_read"] == {"32": 1}


def test_hi_consensus_gates_on_agreement():
    # both covered, but one has split reads (2 of 4 agree = 0.5 < 0.6) -> excluded from hi-consensus
    details = [
        _d(["10", "10", "10", "10"], "10", 4, 4),   # consensus 1.0
        _d(["23", "23", "8", "9"], "23", 4, 4),      # consensus 0.5
    ]
    r = aggregate(details)
    assert r["track_coverage"] == 1.0
    assert r["hi_consensus_coverage"] == 0.5           # only the unanimous one clears 0.6
