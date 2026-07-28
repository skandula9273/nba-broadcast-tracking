"""FastAPI service — health-check stub. This module does NOT import or call the pipeline.

`/health` returns ok; `POST /track` raises 501. The intent is that once the perception stages are wired end
to end, /track will run `pipeline.Pipeline.run(frames)` (the same path the eval harness uses) — but that is
not wired today, so nothing here touches `pipeline.py`.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

app = FastAPI(title="hooptrack")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/track")
def track() -> dict:
    # V1: accept a clip -> run Pipeline.run(frames) -> return top-down tracks + analytics.
    raise HTTPException(status_code=501, detail="not implemented yet — wire the pipeline stages first")
