"""Synthetic jersey render/degrade for the OCR accuracy study — opencv only, no easyocr, CI-safe."""

import numpy as np

from hoopvec.reid.eval_jersey_accuracy import degrade, render_number


def test_render_number_shape():
    img = render_number(23, np.random.default_rng(0), size=96)
    assert img.shape == (96, 96, 3) and img.max() == 255      # white digits present


def test_degrade_downscales_to_target_height():
    rng = np.random.default_rng(0)
    img = render_number(7, rng, size=96)
    d = degrade(img, target_h=16, blur=False, rng=rng)
    assert d.shape[0] == 16 and d.shape[1] >= 4                # height hit, aspect kept
