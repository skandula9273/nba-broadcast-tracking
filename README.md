# hooptrack

Reconstruct player-and-ball tracking ("moving dots") from ordinary **NBA broadcast video**, train a
**play-embedding model** on the reconstructed tracks for similarity retrieval, and wrap both in a **measured,
reproducible eval platform**. The point is the eval rigor and an honest research question, not a tracking demo:

> **How much do downstream basketball analytics (play retrieval, shot quality) degrade when computed on
> CV-reconstructed tracking versus ground-truth tracking — and which perception errors matter most?**

`hooptrack` is a placeholder name. See `docs/design-doc.md` for the full spec and engineering contract. Status: **pre-V0, scaffolding.**

## Pipeline

`broadcast clip → detect → track → homography → re-ID → top-down tracks → {analytics, play-embedding → retrieval}`,
wrapped by an eval harness (mAP, HOTA via TrackEval, recall@k), serving, and observability. The API and the eval
harness call one shared pipeline path.

## Quickstart (once implemented)

```
make install      # deps (+ TrackEval from git)
make test         # the real metric tests pass today
make eval         # eval harness -> eval_results/*.json
make serve        # FastAPI service
```

## Layout

Depth concentrates in `retrieve/` (the embedding core) and the degradation
study; the perception stages are competent SOTA integration, not the headline.

## Data & licenses

SportsMOT (HOTA GT), DeepSportradar (CC-BY-NC-ND), SoccerNet Game State Reconstruction (reference), Basketball-51/
NCAA. Broadcast clips are processed locally and never redistributed. If Ultralytics YOLO is used, this repo is
AGPL-3.0.
