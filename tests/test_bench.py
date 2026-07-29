"""Serving-latency arithmetic — ms/frame + fps, no CV stack, CI-safe."""

from hooptrack.serve.bench import _stage


def test_stage_ms_per_frame_and_fps():
    s = _stage(total_s=2.0, n=200)          # 200 frames in 2s -> 10 ms/frame, 100 fps
    assert s["total_s"] == 2.0
    assert s["ms_per_frame"] == 10.0
    assert s["fps"] == 100.0


def test_stage_zero_time_is_safe():
    s = _stage(total_s=0.0, n=10)           # guard: no division blow-up
    assert s["fps"] is None and s["ms_per_frame"] == 0.0
