# Increment 02 — BoT-SORT vs ByteTrack (the first measured ablation)

**One variable changed:** the tracker. `configs/v0_botsort.yaml` is byte-for-byte identical to
`configs/v0.yaml` except `track.method: bytetrack → botsort` (+ `use_reid: true`). Same detector
(YOLOv8m, COCO person, conf 0.25), same 15 basketball-val sequences, same TrackEval settings. So every
delta below is attributable to the tracker: motion-only (Kalman + IoU) → motion + appearance (OSNet ReID)
+ camera-motion compensation (CMC), at boxmot's tuned defaults.

Sources: `eval_results/eval_20260722T210338Z.json` (ByteTrack) and `eval_results/eval_20260723T221710Z.json`
(BoT-SORT). Reported as-is.

## Result

| tracker | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | Frag | fps (M1/MPS) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack (v0) | 0.301 | 0.325 | 0.279 | 0.837 | −0.395 | 0.281 | 901 | 2017 | 6.2 |
| BoT-SORT | **0.375** | 0.439 | 0.320 | 0.890 | +0.113 | 0.351 | 728 | 2648 | 2.1 |
| Δ | **+0.073** | +0.114 | +0.041 | +0.052 | +0.507 | +0.070 | −173 | +631 | ~3× slower |

BoT-SORT beats the baseline: **HOTA 0.301 → 0.375 (+24% relative)**. Per-sequence HOTA improves on all 15.

## What the deltas mean — and where my prediction was wrong

Going in, I predicted DetA would be **capped**: both trackers consume the *same* YOLO detections, so I
reasoned the detection accuracy couldn't change and BoT-SORT could only help association (AssA). **That was
wrong, and the error is the interesting finding.**

- **DetA jumped +0.114 and MOTA flipped −0.395 → +0.113 (positive).** DetA/MOTA are computed on the
  tracker's **output tracks**, not on the raw detections — and the tracker decides which detections become
  confirmed output tracks. ByteTrack (motion-only, low thresholds) emits nearly every detection as a track,
  so all the referee/bench/crowd false positives land in the output and tank precision. BoT-SORT's default
  config is far stricter: it only *starts* tracks from high-confidence detections
  (`new_track_thresh ≈ 0.62`, `track_high_thresh ≈ 0.63` vs ByteTrack's `track_thresh = 0.45`), requires
  `min_hits` before confirming, and gates matches on appearance + CMC. That confirmation logic **acts as a
  detection filter**, suppressing the crowd-FP tracks that motion-only ByteTrack blindly output. The lesson:
  *a tracker is not a passive consumer of detections — its confirmation policy is itself a precision lever.*
- **AssA +0.041, with IDSW down (901 → 728) but Frag up (2017 → 2648).** BoT-SORT holds identity better
  through occlusion/motion (fewer ID switches — appearance + CMC working), but its stricter confirmation
  also **breaks more tracks**: when a real athlete briefly drops to low confidence, BoT-SORT stops covering
  them, fragmenting the track. Net association still improves, but it's a trade (fewer swaps, more breaks),
  not a pure win.
- **LocA +0.052** — CMC compensates for broadcast camera pans, so matched boxes align slightly better.
- **Cost: ~3× slower** (6.2 → 2.1 fps on M1/MPS) from per-box ReID embedding + per-frame optical-flow CMC.

## The honest caveat (and the next ablation it sets up)

This is a **method-vs-method** comparison — the "one variable" was the whole BoT-SORT bundle (higher
thresholds + `min_hits` confirmation + appearance + CMC). So it does **not** isolate *why* it won. The
biggest driver (DetA/MOTA) is most plausibly the **stricter confirmation thresholds**, not appearance per
se. To attribute the gain, the clean follow-ups — each one variable against this result — are:

1. **ByteTrack at BoT-SORT's thresholds** (raise `track_thresh`/add a `new_track` gate): how much of the
   DetA/MOTA jump is just "stop emitting low-confidence crowd boxes as tracks"?
2. **BoT-SORT with `with_reid: false`**: isolate the appearance contribution to AssA/IDSW.
3. **BoT-SORT with `use_cmc: false`**: isolate camera-motion compensation (the LocA/AssA part).

That decomposition is the natural increment-03. It matters because if a threshold change on the *cheap*
ByteTrack recovers most of the DetA gain, we'd keep the 3× speed and only pay for BoT-SORT's genuine
association benefit — a real accuracy/latency decision, measured rather than assumed.

## Where this leaves the V0 tracking story

The platform's core claim — *ablation-friendly, one variable at a time, committed deltas* — is now
demonstrated with real data: a config-only swap produced a measured, explained, committed HOTA improvement
(0.301 → 0.375), corrected a wrong prior, and named the next experiment. The dominant remaining lever is
still upstream (the COCO-vs-athlete detector: even BoT-SORT's DetA is only 0.44), so **detector
fine-tuning** remains the biggest single HOTA lever for a future increment — now against a stronger 0.375
floor.
