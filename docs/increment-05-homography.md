# Increment 05 — homography (court registration), the V1 top-down enabler

First V1 stage: map image pixels → real court coordinates so player foot points become the top-down
"moving dots" (`Track.court_xy`). Metric: **reprojection error (px)**. Method: **classical, no-training**
(simpler-first); homography is an **enabler**, so it's built solid but not gold-plated.

## What was built

- **`homography/court.py` — the machinery (validated + unit-tested):** GT homography from camera
  calibration (`H = K·[r₁ r₂ T]`), `solve_homography` (correspondences → H via OpenCV DLT + RANSAC),
  `image_to_court` (projects player feet → court coords, fills `Track.court_xy`), `reprojection_error`,
  and `CourtHomography` (the pipeline stage — leaves `court_xy=None` on registration failure rather than
  faking a position). Roundtrips validated to ~1e-5 px.
- **Data:** DeepSportradar basketball-instants camera-calibration set (Kaggle) — **728 instants / 15
  arenas**, each with GT `K/R/T`. (`deepsport-utilities` is broken on import in its current release, so I
  parse the JSONs directly.)

## The honest scope decision

The homography stage is two parts: a **solver** (correspondences → H) and a **registration front-end**
(detect court keypoints in an unlabelled image to *supply* those correspondences). The solver is done and
exact. A robust *classical* front-end (court line/circle detection + template matching across 15 arenas) is
a genuine, finicky CV project — and this is an enabler, so gold-plating it violates the scope discipline.

So this increment measures what can be measured honestly now, and defers the front-end (classical detector
or the learned KaliCalib the user chose not to train):
- **Stage accuracy vs keypoint quality:** simulate a keypoint detector by perturbing GT court keypoints
  with Gaussian pixel noise σ, solve H, measure reprojection error — quantifying exactly how good a
  front-end must be.
- **Trivial baseline:** one global-mean homography for every image (the no-registration floor).

## Results (DeepSportradar, 728 instants / 15 arenas)

| what | median reproj error (px) |
|---|---:|
| Trivial baseline (global-mean H) | **877** |
| Solver @ keypoints σ=1 px | 0.4 |
| Solver @ keypoints σ=3 px | 2.1 |
| Solver @ keypoints σ=5 px | 4.4 |
| Solver @ keypoints σ=10 px | 11.0 |

**The findings:** (1) registration is essential — the no-registration floor is **877 px**. (2) The solver
is **sub-pixel-to-few-px accurate** given keypoints detected to a few pixels. So the entire homography
accuracy budget lives in the **keypoint front-end**: hit ~3–5 px keypoint detection → ~2–4 px court
registration. That's the concrete target the deferred front-end must clear, now quantified rather than
guessed.

## Honest boundaries

- **This characterizes the solver, not an end-to-end automatic registration.** The keypoint-detection
  front-end (classical or KaliCalib) is the remaining component — its error would add to the σ-curve above.
- **DeepSportradar = fixed arena cameras**, not moving broadcast cameras (SportsMOT). So the *measured*
  number is on arena-camera calibration GT; applying homography to broadcast is qualitative (the design
  doc's stated "no aligned broadcast↔GT" boundary).
- Court frame = FIBA 2800×1500 cm (the DeepSportradar court coordinate system).

## Where this leaves V1

The top-down machinery exists and is measured: given court keypoints, we register to a few px and fill
`Track.court_xy`. The keypoint front-end is a clean, well-scoped follow-up (with a concrete accuracy
target). Next V1 stage per the pipeline: **re-ID / player identity**, then the embedding core + the
degradation study (the depth).
