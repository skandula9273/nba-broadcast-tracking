"""Retrieval-uncertainty calibration + selective-prediction logic — pure numpy, but the module pulls torch
(checkpoint loader) at import, so it's skipped on the torch-less CI gate."""

import numpy as np
import pytest

pytest.importorskip("torch")  # uncertainty.py -> checkpoint.py imports torch; skip if absent (CI [dev] gate)

from hooptrack.retrieve.uncertainty import (  # noqa: E402
    _calibration,
    _retrieve_with_confidence,
    _selective,
)


def test_retrieve_with_confidence_correct_and_margin():
    # gallery = 3 orthonormal rows; query 0 aligned to row 0 (correct), query 1 aligned to row 2
    gemb = np.eye(3, dtype=float)
    qemb = np.array([[1.0, 0.1, 0.0], [0.0, 0.05, 1.0]])
    r = _retrieve_with_confidence(qemb, gemb, relevant=np.array([0, 2]))
    assert list(r["correct"]) == [1, 1]
    assert (r["margin"] > 0).all()                      # top-1 strictly beats top-2


def test_selective_accuracy_rises_when_confidence_is_calibrated():
    # correctness perfectly ordered by confidence -> restricting to the top-confidence half is 100% accurate
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    correct = np.array([1, 1, 1, 0, 0, 0])
    sel = {s["coverage"]: s["accuracy"] for s in _selective(correct, conf, coverages=(1.0, 0.5))}
    assert sel[0.5] == 1.0 and sel[1.0] == 0.5          # answer the confident half -> perfect; all -> 50%


def test_calibration_reports_positive_corr_for_aligned_signal():
    conf = np.linspace(0, 1, 100)
    correct = (conf > 0.5).astype(int)                  # confidence predicts correctness
    cal = _calibration(correct, conf, n_bins=5)
    assert cal["confidence_correctness_corr"] > 0.5
    assert 0.0 <= cal["ece"] <= 1.0
