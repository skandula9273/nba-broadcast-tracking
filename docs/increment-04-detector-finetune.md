# Increment 04 — detector fine-tuning (the lever the data pointed at)

Increment-03 localized the tracking ceiling to the **detector**: every tracker variant's best DetA was
0.44 because COCO "person" was never told what an athlete is (it fires on referees, bench, crowd). This
increment fine-tunes yolov8m on SportsMOT athletes and measures the lift — one variable (the detector
weights), same ByteTrack, same inference imgsz 1280, against the 0.301 baseline.

## Setup

- **Data:** 15 basketball-train sequences → MOT boxes → YOLO detection labels (single class `athlete`),
  every 5th frame (25 fps frames are near-duplicates): **2,484 train / 2,516 val images**.
- **Training:** yolov8m from COCO weights, **20 epochs, imgsz 640, batch 4, AMP off**, seed 13, MPS.
  (The Standard budget's 960/8 was infeasible — see the MPS notes below.)
- **Inference:** imgsz **1280** (identical to the baseline), so the tracking A/B isolates the weights.

## Detection stage (the stage's own public number)

Fine-tuned yolov8m on basketball-val, **single class `athlete` — the ball is NOT detected**: **mAP50 = 0.987,
mAP50-95 = 0.795** (P 0.971 / R 0.958) — vs generic COCO "person," which capped DetA at 0.44. The detector now
reads the scene as *athletes*, not *people*. **Caveat, stated with the number (not three sections down):**
`best.pt` was model-**selected** on this same basketball-**val** split — the 15 sequences HOTA is scored on —
so the figure is **mildly optimistic** (the weights only ever saw basketball-train; a held-out SportsMOT *test*
number, GT withheld behind Codalab, is the rigorous version). **Archived** as a real artifact via
`make detect-eval` → `eval_results/detection_*.json` (previously a training-console figure only — every tracking
eval JSON has `detection_mAP: null`); re-running reproduces mAP50 0.9872 / mAP50-95 0.7951.

## Tracking A/B (fine-tuned + ByteTrack vs COCO + ByteTrack)

| variant | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | Frag | fps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO + ByteTrack (v0) | 0.301 | 0.325 | 0.279 | 0.837 | −0.395 | 0.281 | 901 | 2017 | 6.2 |
| **Fine-tuned + ByteTrack** | **0.473** | **0.707** | 0.317 | 0.841 | **+0.874** | 0.504 | 955 | 1847 | 7.6 |
| Δ | **+0.172** | **+0.382** | +0.038 | +0.004 | +1.268 | +0.223 | +54 | −170 |

**DetA more than doubled and MOTA flipped from −0.40 to +0.87** — the crowd/bench false-positive tracks are
gone. HOTA **+0.172** (+57% relative).

## The finding (the whole measured arc pays off)

**Fine-tuned + cheap motion-only ByteTrack (0.473) beats the expensive BoT-SORT bundle (0.375).** The
single detector variable delivered **2.3× the HOTA of the entire BoT-SORT tracker upgrade** (+0.172 vs
+0.074), at higher fps. This is the payoff of increment-02→03→04 done as measured single variables: the
tracker ablation + attribution said the detector was the binding constraint and appearance/CMC were dead
weight; fixing the detector — not gold-plating the tracker — was worth far more. We didn't guess the lever;
the committed numbers pointed at it.

## Honest caveats

- **Mildly optimistic number.** The *weights* only ever saw basketball-**train** (no frame leakage), but
  `best.pt` was model-**selected** by basketball-**val** mAP — the same 15 sequences HOTA is scored on. So
  the lift is real and large, but a held-out **test** number (SportsMOT test GT is withheld behind Codalab)
  would be the rigorous version. Reported as-is.
- **Association is now the relative bottleneck.** With far more athletes correctly detected (higher
  recall), motion-only ByteTrack has more real tracks to confuse — **IDSW actually rose (901 → 955)**. So
  the next association lever (which was dead weight on noisy detections) may now matter *on clean
  detections* — a hypothesis for a future ablation, not a claim.
- **MPS training reality (the saga).** The Standard budget (imgsz 960 / batch 8) **diverged to NaN losses
  and thrashed the 16 GB unified memory** — caught by a 1-epoch probe before the full run. Fixes: AMP off
  (the MPS NaN cause), batch 4 / imgsz 640 (fits memory). Even then, **per-epoch time grew over the run**
  (mosaic aug early + MPS memory pressure), so the actual wall-clock was **~7.3 h**, not the ~3 h the warm
  1-epoch probe implied — an honest ETA miss: a single-epoch measurement under-predicts MPS training.
- **Reproduce:** `python -m hoopvec.detect.finetune --epochs 20 --imgsz 640 --batch 4 --subsample 5
  --device mps`, then `make track/eval` with `configs/v0_finetuned.yaml`. (Fine-tuned weights are
  gitignored; the training set is re-derivable from `make data` + the finetune command.)

## Where this leaves V0

Both V0 perception numbers are now measured and committed: **detection mAP50 0.987** and **tracking HOTA
0.473** (fine-tuned + ByteTrack). The tracking ceiling has moved from the detector to association, on clean
detections. V0 is done — next is V1 (homography → re-ID → the embedding core → the degradation study).
