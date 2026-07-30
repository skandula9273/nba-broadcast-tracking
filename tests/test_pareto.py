"""Pareto-frontier logic for the detector accuracy-latency sweep — pure, CI-safe."""

from hooptrack.detect.pareto import _frontier


def test_frontier_marks_non_dominated_points():
    pts = [
        {"mAP50": 0.99, "fps": 10},   # highest accuracy, slowest -> on frontier
        {"mAP50": 0.90, "fps": 30},   # lower accuracy, fastest -> on frontier
        {"mAP50": 0.85, "fps": 20},   # worse on BOTH than pt1 -> dominated
    ]
    _frontier(pts)
    assert pts[0]["on_frontier"] is True
    assert pts[1]["on_frontier"] is True
    assert pts[2]["on_frontier"] is False


def test_equal_points_are_not_falsely_dominated():
    pts = [{"mAP50": 0.9, "fps": 20}, {"mAP50": 0.9, "fps": 20}]   # a tie dominates neither (no strict improvement)
    _frontier(pts)
    assert pts[0]["on_frontier"] and pts[1]["on_frontier"]
