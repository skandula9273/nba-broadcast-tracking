"""Jersey-OCR coverage aggregation — pure metric logic, no easyocr/data, CI-safe."""

from hoopvec.reid.eval_jersey import aggregate


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


def test_coverage_substantial_ignores_short_fragments():
    # tracker output: many short fragments (few crops) can't be read; long tracks can. `coverage_substantial`
    # measures coverage among tracks with >= min_substantial (10) crops, factoring fragmentation out.
    details = [
        _d(["9", "9"], "9", 30, 2),        # substantial (30 crops), covered
        _d([], None, 25, 0),                # substantial, not covered
        _d(["4"], None, 3, 1),              # short fragment, uncovered -> excluded from substantial denom
        _d(["7"], None, 2, 1),              # short fragment
    ]
    r = aggregate(details, min_substantial=10)
    assert r["n_tracks"] == 4
    assert r["track_coverage"] == round(1 / 4, 3)          # raw: 1 of 4 ids
    assert r["n_substantial"] == 2                          # only the two >=10-crop tracks
    assert r["coverage_substantial"] == 0.5                 # 1 of those 2 got a number


def test_hi_consensus_gates_on_agreement():
    # both covered, but one has split reads (2 of 4 agree = 0.5 < 0.6) -> excluded from hi-consensus
    details = [
        _d(["10", "10", "10", "10"], "10", 4, 4),   # consensus 1.0
        _d(["23", "23", "8", "9"], "23", 4, 4),      # consensus 0.5
    ]
    r = aggregate(details)
    assert r["track_coverage"] == 1.0
    assert r["hi_consensus_coverage"] == 0.5           # only the unanimous one clears 0.6
