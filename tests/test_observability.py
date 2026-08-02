"""Serving observability — metrics aggregation + drift signal, pure, CI-safe."""

from hoopvec.serve.observability import Metrics


def _timings(detect=0.1, track=0.01):
    return {"detect_s": detect, "track_s": track}


def test_summary_aggregates_latency_and_throughput():
    m = Metrics()
    for _ in range(4):
        m.record(n_frames=10, timings=_timings(0.2, 0.02), n_tracks=10, n_dets=100)
    s = m.summary()
    assert s["requests"] == 4 and s["frames_total"] == 40
    assert s["latency_s"]["p50"] > 0 and s["detect_latency_s"]["p50"] == 0.2
    assert s["detections_per_frame_baseline"]["mean"] == 10.0     # 100 dets / 10 frames


def test_drift_flags_off_distribution_detection_count():
    m = Metrics()
    for d in [98, 101, 99, 102, 100, 103, 97, 101, 99, 100]:    # realistic baseline ~10 dets/frame, small spread
        m.record(n_frames=10, timings=_timings(), n_tracks=10, n_dets=d)
    # a request with a wildly different detection count should flag drift (>3 sigma on the normal path)
    verdict = m.record(n_frames=10, timings=_timings(), n_tracks=1, n_dets=5)   # 0.5 dets/frame vs ~10
    assert verdict["status"] == "drift" and abs(verdict["z"]) > 3


def test_drift_is_baseline_until_enough_history():
    m = Metrics()
    v = m.record(n_frames=10, timings=_timings(), n_tracks=10, n_dets=100)
    assert v["status"] == "baseline"                            # <5 seen -> no judgement yet
