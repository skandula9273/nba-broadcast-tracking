"""FastAPI service. Calls the SAME pipeline the eval harness uses (one shared path).

V0: /health only (no external calls). /track is stubbed until the pipeline stages are wired.
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
