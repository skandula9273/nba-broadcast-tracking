"""Tests for the numbers-consistency checker (src/hoopvec/eval/check_numbers.py).

Two layers: unit tests on the path/family primitives, and end-to-end drift tests that build a tiny
README + manifest + artifact in a tmp dir and assert each drift mode is caught. Keeps the checker itself
honest — a checker with a silent bug is worse than no checker.
"""
from __future__ import annotations

import json

import yaml

from hoopvec.eval.check_numbers import family_of, latest_in_family, read_value, resolve, run


def test_resolve_handles_keys_list_indices_and_dotted_keys():
    obj = {"results": {"sweeps": {"id_swap": [{"r": 1.0}, {"r": 0.4}]}, "coverage@0.25": 0.236}}
    assert resolve(obj, "results/sweeps/id_swap/1/r") == 0.4      # list index mid-path
    assert resolve(obj, "results/coverage@0.25") == 0.236         # key containing a dot (slash-separated)


def test_read_value_derived_divide():
    obj = {"points": [{"fps": 10.0}, {"fps": 26.0}]}
    claim = {"derived": {"op": "divide", "of": "points/1/fps", "by": "points/0/fps"}}
    assert read_value(obj, claim) == 2.6


def test_family_and_latest(tmp_path):
    # exact-prefix + timestamp only; must NOT collapse detection_ into detection_generalization_
    for name in ("detection_20260101T000000Z.json", "detection_20260202T000000Z.json",
                 "detection_generalization_20260303T000000Z.json"):
        (tmp_path / name).write_text("{}")
    assert family_of("detection_20260202T000000Z.json") == "detection"
    assert family_of("no_timestamp.json") is None
    assert latest_in_family(tmp_path, "detection_20260101T000000Z.json") == "detection_20260202T000000Z.json"
    # the generalization file is a different family and must not be picked as detection's latest
    assert latest_in_family(tmp_path, "detection_generalization_20260303T000000Z.json") \
        == "detection_generalization_20260303T000000Z.json"


def _scaffold(tmp_path, readme_body, claims, extra=None):
    (tmp_path / "art").mkdir(exist_ok=True)
    (tmp_path / "art" / "metric_20260101T000000Z.json").write_text(json.dumps({"results": {"acc": 0.873}}))
    readme = tmp_path / "README.md"
    readme.write_text(readme_body)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump({"claims": claims, **(extra or {})}))
    return readme, manifest, tmp_path / "art"


def _base_claim():
    return {"id": "acc", "anchors": ["accuracy **0.87**"], "artifact": "metric_20260101T000000Z.json",
            "path": "results/acc", "display": 0.87, "tol": 0.01}


def test_passes_when_consistent(tmp_path):
    readme, manifest, art = _scaffold(tmp_path, "model accuracy **0.87** on val.\n", [_base_claim()])
    rep = run(readme, manifest, art)
    assert rep.ok, [i.detail for i in rep.issues]


def test_catches_mismatch(tmp_path):
    claim = _base_claim()
    claim["display"] = 0.95   # README/manifest say 0.95 but the artifact says 0.873
    claim["anchors"] = ["accuracy **0.95**"]
    readme, manifest, art = _scaffold(tmp_path, "model accuracy **0.95** on val.\n", [claim])
    rep = run(readme, manifest, art)
    assert any(i.kind == "MISMATCH" for i in rep.issues)


def test_catches_missing_anchor(tmp_path):
    # README shows a different number than the manifest anchor -> the claim text is gone
    readme, manifest, art = _scaffold(tmp_path, "model accuracy **0.88** on val.\n", [_base_claim()])
    rep = run(readme, manifest, art)
    assert any(i.kind == "MISSING ANCHOR" for i in rep.issues)


def test_catches_stale_family_and_pin_reason_suppresses_it(tmp_path):
    readme, manifest, art = _scaffold(tmp_path, "model accuracy **0.87** on val.\n", [_base_claim()])
    (art / "metric_20270101T000000Z.json").write_text(json.dumps({"results": {"acc": 0.9}}))  # newer run
    assert any(i.kind == "STALE FAMILY" for i in run(readme, manifest, art).issues)
    # same, but the claim deliberately pins the older run -> no failure
    claim = _base_claim()
    claim["pin_reason"] = "cites the 2026 run deliberately"
    manifest.write_text(yaml.safe_dump({"claims": [claim]}))
    assert run(readme, manifest, art).ok


def test_flags_unsourced_number(tmp_path):
    body = "accuracy **0.87** on val, and an unmapped **0.42** somewhere.\n"
    rep = run(*_scaffold(tmp_path, body, [_base_claim()]))
    assert any(i.kind == "UNSOURCED" and "0.42" in i.detail for i in rep.issues)


def test_ignore_pattern_suppresses_unsourced(tmp_path):
    body = "accuracy **0.87** on val, built with Python 3.11.\n"
    rep = run(*_scaffold(tmp_path, body, [_base_claim()], extra={"ignore_patterns": [r"3\.11"]}))
    assert rep.ok, [i.detail for i in rep.issues]
