"""ONNX provider bench timing helper — pure, no onnxruntime/torch, CI-safe."""

import time

from hoopvec.detect.onnx_providers import _time


def test_time_returns_ms_per_call():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        time.sleep(0.001)                                   # 1 ms of work

    ms = _time(fn, iters=5, warmup=2)
    assert calls["n"] == 7                                  # warmup (2) + timed (5)
    assert ms >= 1.0                                        # ~1 ms/call floor (sleep), not the warmup
