# hooptrack

Reconstruct player-and-ball tracking ("moving dots") from ordinary **NBA broadcast video**, train a
**play-embedding model** on the reconstructed tracks for similarity retrieval, and wrap both in a **measured,
reproducible eval platform**. The point is the eval rigor and an honest research question, not a tracking demo:

> **How much do downstream basketball analytics (play retrieval, shot quality) degrade when computed on
> CV-reconstructed tracking versus ground-truth tracking — and which perception errors matter most?**

`hooptrack` is a placeholder name. See `docs/design-doc.md` for the full spec and engineering contract (with
dated amendments) and `docs/increment-0N-*.md` for per-increment writeups. Status: **V1 in progress** — V0
(detection + tracking + eval harness) done and floors locked; the embedding core, FAISS index, the degradation
study, an order-robustness study, and a permutation-invariant encoder are done; remaining V1 is re-ID, then a
demo + writeup.

## Results so far (measured, committed to `eval_results/`)

- **Detection** mAP@50 **0.987** (fine-tuned yolov8m) · **Tracking** HOTA **0.301 → 0.473** on SportsMOT
  basketball-val, via TrackEval — arc: ByteTrack → BoT-SORT → attribution → detector fine-tune.
- **Play retrieval** (the centerpiece): a trained trajectory transformer beats the hand-feature floor on a
  held-out split — overall recall@1 **0.41 → 0.98**, court-mirror **0.004 → 0.999**; a one-variable ablation
  attributes the win to the mirror augmentation, FAISS-indexed and verified to reproduce brute-force.
- **The degradation study** (the finding): reconstruction cost concentrates in **tracking association**
  (ID-switches) — at the measured error budget, retrieval recall@1 **1.0 → 0.68**, while homography/detection
  noise costs ~nothing; player-identity (re-ID) is the dominant unmeasured risk.

## Pipeline

`broadcast clip → detect → track → homography → re-ID → top-down tracks → {analytics, play-embedding → retrieval}`,
wrapped by an eval harness (mAP, HOTA via TrackEval, recall@k), serving, and observability. The API and the eval
harness call one shared pipeline path.

## Quickstart (once implemented)

```
make install         # deps (+ TrackEval from git)
make test            # the real metric tests pass today (31)
make eval            # tracking eval harness -> eval_results/*.json
make retrieve-corpus # build the SportVU possession corpus
make retrieve-floor  # recall@k hand-feature floor (inc-06a)
make retrieve-train  # trained trajectory transformer + FAISS index (inc-06b)
make retrieve-study  # reconstructed-vs-GT degradation study (inc-07)
make serve           # FastAPI service
```

## Layout

Depth concentrates in `retrieve/` (the embedding core) and the degradation
study; the perception stages are competent SOTA integration, not the headline.

## Data & licenses

SportsMOT (HOTA GT), DeepSportradar (CC-BY-NC-ND), SoccerNet Game State Reconstruction (reference), Basketball-51/
NCAA. Broadcast clips are processed locally and never redistributed. If Ultralytics YOLO is used, this repo is
AGPL-3.0.
