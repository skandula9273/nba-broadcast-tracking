"""Serve endpoint tests (fastapi-gated). These do NOT load the CV stack — the request/path is validated
before the pipeline is built, so 404/422 return without importing YOLO/boxmot."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from hooptrack.serve.app import app  # noqa: E402

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_track_missing_source_returns_404():
    r = client.post("/track", json={"source": "/tmp/hooptrack_does_not_exist"})
    assert r.status_code == 404          # path validated before the pipeline is built (no CV stack loaded)


def test_track_requires_source():
    r = client.post("/track", json={})
    assert r.status_code == 422          # pydantic: source is required
