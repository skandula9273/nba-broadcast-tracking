# Increment 07 — the reconstructed-vs-GT degradation study (the headline finding)

The question the whole platform exists to answer: **how much play-retrieval accuracy does broadcast
reconstruction cost vs ground-truth tracks — and which perception stage costs it?** No dataset has aligned
broadcast + SportVU truth, so this is a **controlled degradation**: take clean SportVU GT possessions, inject
the per-stage error budgets *this project already measured*, and measure the recall@k falloff. It reuses the
entire harness — the trained encoder, the FAISS index, recall@k — swapping the augmentation for a
physically-motivated reconstruction-error model.

## Method

Query = a **degraded** GT possession; gallery = the **clean** GT val corpus (858 held-out possessions,
4 games the encoder never trained on); relevant = the source possession; retrieval = the FAISS index.
Baseline (no degradation) = **recall@1 1.0** (the query is the gallery item), so every point below 1.0 is
recall the reconstruction costs. The **trained encoder** and the **hand-feature floor** are reported side by
side, one error source at a time (`degrade.py`, `study.py`).

## The error model, anchored to measured budgets (rule #5)

| knob | models | anchored to (committed) |
|---|---|---|
| `jitter_ft` | positional Gaussian noise | homography reg. **~2px @ 3px keypoints** (inc-05) + detection **LocA 0.841** |
| `dropout` + interp | occlusion / missed detection, gap bridged | detection **DetA 0.707**, **Frag 1847** (inc-04) |
| `id_swap` | two players' tracks swapped mid-possession | tracking **AssA 0.317**, **IDSW 955** / 12,557 frames (inc-04) |
| `permute_players` | k players' slots shuffled (arbitrary track order) | **re-ID — NOT built/measured**: sensitivity only |

The id-swap count is anchored order-of-magnitude (IDSW/frame ≈ 0.076 → ~2–4 swaps per 48-frame possession);
the **sweep** carries the finding, not any single derived number.

## Results — recall@1 (trained | floor), vs the 1.0 GT baseline

| severity → | jitter_ft (ft) | | dropout | | id_swap (#) | |
|---|---|---|---|---|---|---|
| | **trained** | floor | **trained** | floor | **trained** | floor |
| none | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| low | 1.000 (0.5) | 1.000 | 1.000 (0.1) | 1.000 | **0.810** (1) | 0.999 |
| mid | 1.000 (2.0) | 1.000 | 0.993 (0.3) | 1.000 | **0.681** (2) | 0.988 |
| high | 0.999 (4.0) | 1.000 | 0.918 (0.4) | 1.000 | **0.435** (4) | 0.959 |

**Combined realistic** (0.5 ft + 0.1 dropout + 2 swaps): **trained recall@1 0.677**, floor **0.992**.

**re-ID sensitivity** (pending measurement) — `permute_players` recall@1 (trained | floor):
`n=0` 1.00|1.00 · `n=2` 0.81|0.99 · `n=4` 0.41|0.95 · `n=6` 0.16|0.81 · `n=10` 0.02|0.20.

## The findings

1. **Reconstruction cost concentrates in ONE stage: tracking association.** Positional jitter (homography +
   detection localization) costs essentially nothing — even **4 ft** of noise leaves recall@1 at 0.999 — and
   fragmentation, bridged by interpolation, barely moves it until extreme. But **ID-switches degrade sharply**:
   each swap costs the trained encoder ~0.12–0.15 recall@1. So for play-retrieval off broadcast, **who-is-who
   over time is the bottleneck, not sub-pixel registration** — a concrete, actionable steer for where to spend.

2. **An honest reversal: the learned encoder is *more fragile* to reconstruction noise than the naive floor.**
   The encoder that crushed the floor on the augmentation-SSL eval (court-mirror 0.004 → 0.999, inc-06b) loses
   to it under id-swaps and permutation (combined realistic: 0.68 vs 0.99). Why: the floor mean-pools raw
   coordinates, so swapping 2 of 11 entities barely moves the cosine — it is **inherently order-robust** (and
   inherently mirror-blind). The transformer is **order-sensitive** by construction and was trained only with
   order-*preserving* augmentations (jitter/crop/mirror), so id-swaps are out-of-distribution. These are
   **complementary failure modes**, not "floor is better" — and the fix is the exact lever that worked for
   mirror: **add id-swap / permutation augmentation to training** to buy order-robustness while keeping
   mirror-invariance. The study turned that into a specific, testable next hypothesis.

3. **Player identity (re-ID) is the dominant *unmeasured* risk.** A real tracker emits arbitrary track order;
   the encoder needs canonical order. Even **2** mis-identified players drop trained recall@1 to 0.81; a full
   scramble → 0.02. This is shown as sensitivity only (no operating point — the re-ID stage isn't built), and
   it is precisely why **re-ID is the next stage to build and measure**.

## Honest boundaries

- **Controlled degradation, not real broadcast.** GT tracks perturbed by measured budgets — not tracks
  reconstructed end-to-end from a broadcast clip (no aligned broadcast+SportVU truth exists). The error model
  is a faithful, measured proxy, not the real pipeline output.
- **Mapping assumptions stated, not exact.** px→ft and IDSW/frame→swaps/possession are order-of-magnitude
  anchors; the sweeps (not a single point) carry the conclusions.
- **re-ID axis is unanchored** (that stage is unmeasured) — labelled sensitivity-only throughout.
- **Same proxy caveats as inc-06b** (augmentation-SSL invariance, not set-play semantics; val is selection +
  reporting; MPS not bit-exact — but the effect sizes dwarf the noise).

## Next

The study named two levers, in order: (1) **add id-swap/permutation augmentation** to the encoder and re-measure
this exact degradation curve (does order-robustness close the trained-vs-floor gap without losing mirror?);
(2) **build + measure the re-ID stage**, then replace the sensitivity axis with a real operating point. That
closes the loop from perception error budgets → the retrieval cost they impose.
