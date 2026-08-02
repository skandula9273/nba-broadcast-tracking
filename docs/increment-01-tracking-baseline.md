# Increment 01 — SportsMOT + ByteTrack HOTA baseline

**Status:** the first measured increment (pre-V0 → V0's headline public number).
**What it produces:** a real **HOTA** number for a *pretrained-detector + ByteTrack* baseline on
**SportsMOT (basketball, val split)**, computed with **TrackEval**, committed as timestamped JSON in
`eval_results/`, and reproducible with three `make` commands.

This document is the conceptual + technical explainer for that increment — written so you can read the
code, defend every choice, and talk/write about it. Numbers for *this* run live in
[Results](#9-results-this-run) and in the committed `eval_results/eval_*.json`.

---

## 1. What we built, and why this first

The project's engineering contract has one ordering rule above the rest: **the eval harness stands up
before any modeling**, and **every increment ends in a real public-benchmark number** — never a
fabricated or hand-tuned one. So the very first thing to build is not a model; it's the machinery that
can *grade* a model against a public benchmark, plus the simplest honest baseline to run through it.

That baseline is **tracking-by-detection**: an off-the-shelf, COCO-pretrained object detector finds
people in each frame, and **ByteTrack** links those detections into tracks across frames. We score the
tracks with **HOTA** — the field-standard multi-object-tracking metric — on **SportsMOT**, a public
basketball/volleyball/soccer tracking benchmark. Nothing here is trained by us yet. That is the point:
it establishes a **measured floor** that every later, heavier method must beat, one variable at a time.

Two properties make the number trustworthy:

- **One shared pipeline.** The eval harness and (later) the serving API call the *same* code path
  (`src/hoopvec/pipeline.py`). We measure the system we would deploy, not a lookalike.
- **The harness cannot fake a number.** `eval/run.py` writes a metric only if its stage actually ran;
  anything unrun is `null` with a `status` string. A scaffold run can never be misread as a result.

---

## 2. The mental model: multi-object tracking (MOT)

**Detection** answers "what objects are in *this frame*, and where?" → a set of boxes per frame, with no
memory of previous frames. **Tracking** adds identity over time: "box A in frame 12 and box B in frame 13
are *the same player*." The output is a set of **tracks**, each a consistent integer ID whose box moves
frame to frame.

**Tracking-by-detection** is the dominant paradigm and the one we use: run a detector independently per
frame, then a separate **association** step stitches detections into tracks. The two halves are decoupled,
which is exactly what we want for a measured platform — we can swap the detector or the tracker
*independently* and attribute any change to the one thing we changed ("one variable at a time" — a core
project rule). The alternative, joint detection-and-tracking models, couples the two and makes clean ablation
harder.

**Online vs offline.** An *online* tracker decides each frame using only past+present frames (needed for
live/streaming; ByteTrack is online). An *offline* tracker may look at the whole clip at once (can be more
accurate, not causal). We use online tracking — it matches a real broadcast-ingest product and is the
honest setting.

A **track**, concretely, is: `track_id` + a box `(x, y, w, h)` for each frame the object is visible.

---

## 3. ByteTrack — the baseline tracker, in depth

ByteTrack (Zhang et al., ECCV 2022) is the standard *simple, strong* tracking-by-detection baseline. Two
ideas:

1. **A Kalman-filter motion model.** Each track carries a constant-velocity Kalman filter over its box
   state. Each new frame, the filter **predicts** where the box should be; association then matches
   detections to those predictions. There is **no appearance model** — ByteTrack never looks at pixels,
   only at box geometry and motion. That's why it's called *motion-only*.

2. **Associate *every* detection box — the "BYTE" trick.** Detectors emit a confidence per box. Most
   trackers throw away low-confidence boxes and only match the high-confidence ones — but a briefly
   occluded or motion-blurred player often *drops to low confidence* rather than vanishing. ByteTrack
   keeps them and associates in **two stages**:
   - **Stage 1:** match **high**-score detections (≥ `track_thresh`) to existing tracks, using IoU
     between each detection and each track's Kalman-predicted box (Hungarian assignment, gated by
     `match_thresh`).
   - **Stage 2:** for tracks still unmatched, try to match them to the **low**-score detections
     (down to `min_conf`). This recovers occluded/blurred objects that would otherwise fragment into a
     new ID.
   Unmatched tracks are kept "lost" for `track_buffer` frames (so a track survives a short gap and keeps
   its ID) before deletion; unmatched high-score detections start new tracks.

**Why it's the right *floor*, and why sports strain it.** ByteTrack is cheap, reproducible, and needs no
training — a clean baseline. But motion-only association has a known weakness on sports, and SportsMOT was
built to expose it: players move **fast and non-linearly** (cuts, screens), so the constant-velocity
Kalman prediction is often wrong; they **occlude each other constantly**; and they wear **near-identical
uniforms**, so even if you *added* appearance features, teammates look alike. The result is **ID switches**
(two players' tracks swap) and **fragmentation** (one player becomes several IDs). Association accuracy —
not detection — is where sports tracking is hard, and it's precisely what our next ablation (BoT-SORT /
Deep-EIoU, which fuse appearance + better motion) will try to improve. ByteTrack first, measured; heavier
methods must then beat it, measured.

---

## 4. HOTA — the metric, in depth

A tracker makes two kinds of error: **detection** errors (miss a player, or hallucinate one) and
**association** errors (link the wrong boxes over time — ID switches). A good metric must weigh both.

- **MOTA** (the old default) = `1 − (FN + FP + IDSW) / GT`. It is dominated by detection errors (FN+FP are
  usually far larger than IDSW), can go **negative**, and barely reflects association quality. Good
  detection can mask terrible identity.
- **IDF1** swings the other way — it's an identity-matching F1 that emphasizes association, and can look
  good even with mediocre detection.
- **HOTA** (Luiten et al., IJCV 2021) was designed to **balance the two explicitly**, and it correlates
  best with human judgment of tracking quality. That's why SportsMOT and the modern MOT leaderboards
  report it, and why it's our headline.

How HOTA is computed (what TrackEval does under the hood):

- At a localization threshold **α** (how much box overlap counts as a match), compute
  - **DetA(α)** — *detection accuracy*: `TP / (TP + FN + FP)`, a Jaccard over boxes.
  - **AssA(α)** — *association accuracy*: for each matched pair, how consistently the two identities are
    linked across the whole track (averaged), penalizing ID switches and fragments.
- **HOTA(α) = √(DetA(α) × AssA(α))** — a geometric mean, so you can't buy a good score with only one half.
- The reported **HOTA** is HOTA(α) **averaged over α = 0.05 … 0.95** (19 thresholds). So a single HOTA
  number already integrates over "how strict is a match." Our adapter takes exactly this mean
  (`_scalar` in `eval/trackeval_adapter.py`).
- **LocA** (localization accuracy, the average IoU of matched pairs) is reported alongside — it tells you
  *how tight* the boxes are, separate from whether they were found.

We commit **HOTA, DetA, AssA, LocA, MOTA, MOTP, IDF1, IDSW, Frag** — HOTA is the headline; DetA vs AssA
tells the *story* (for a pretrained-detector + motion-only baseline on sports, expect DetA held back by
the class mismatch in §6, and AssA held back by motion-only association).

---

## 5. SportsMOT — the benchmark

**SportsMOT** (Cui et al., ICCV 2023) is a large multi-object-tracking dataset of **240 sequences** across
**basketball, volleyball, and soccer**, ~150k frames, all annotated in **MOT-Challenge format** with a
**single class: the on-court athletes**. It's the right benchmark here because it's public, HOTA-scored
via TrackEval, and specifically stresses the sports failure modes in §3.

- **Splits:** `train` / `val` / `test`. **Test ground truth is withheld** (you submit to a Codalab
  server for the leaderboard). For a *locally committed, reproducible* HOTA we therefore use **val**,
  whose GT is public. Our number is comparable to **published val baselines**, not the test leaderboard —
  stated plainly in the JSON `caveats`.
- **Sport split:** `splits_txt/basketball.txt` lists the basketball sequences. We evaluate
  **basketball ∩ val = 15 sequences** (see the run manifest). Keeping to basketball matches the project's
  domain.
- **MOT-Challenge layout** (per sequence): `img1/000001.jpg …` (frames), `gt/gt.txt` (annotations),
  `seqinfo.ini` (width/height/fps/length). A `gt.txt` row is
  `frame, id, bb_left, bb_top, bb_w, bb_h, conf, class, visibility`; **class column (index 7) is `1`** for
  athletes, which maps to TrackEval's `pedestrian` class.

We pull the official `MCG-NJU/SportsMOT` distribution from the HuggingFace Hub — just `dataset/val.tar`
(~6.6 GB) — and extract only the 15 basketball-val sequences (`ingest/fetch.py`). We do **not** relabel
or redistribute; `data/` is gitignored.

---

## 6. The honest boundary: pretrained COCO "person" ≠ SportsMOT "athlete"

The single most important caveat. Our detector is **COCO-pretrained YOLOv8m, not fine-tuned on
SportsMOT.** We keep only its `person` class (COCO index 0) as an athlete proxy. But:

- COCO "person" fires on **referees, bench players, coaches, and courtside crowd** — none of which are in
  SportsMOT's athlete-only GT. Every such box is a **false positive** → **DetA drops**.
- Fast motion blur and heavy occlusion cause **missed** athletes → **false negatives** → DetA drops too.

This is deliberate and reported as-is (the project's honesty rule). The gap between a generic person detector and
an athlete detector is *itself a finding*, and it sets up the next measured step: **fine-tune the detector
on SportsMOT and measure the DetA/HOTA lift** — one variable, against this committed floor. We do not
gold-plate the perception here; the project's depth is reserved for the embedding core and the degradation
study (the project's scope discipline).

Other boundaries, all carried in the JSON `caveats`:
- **val, not test** (§5).
- **Motion-only tracker** (§3) — the association floor.
- **`DO_PREPROC=False`** in TrackEval: SportsMOT has no distractor classes, so we evaluate all boxes as-is
  (the authors' comparable setting; verified against the installed TrackEval source).
- **MPS determinism:** inference is deterministic *same-machine / same-versions* only (seed fixed in
  `track/run.py`, versions pinned in `requirements.lock`). We don't claim bitwise reproducibility across
  different hardware.

---

## 7. The exact wiring (how the code fits together)

```
make data     ingest/fetch.py   HF val.tar -> data/sportsmot/val/<seq>/{img1,gt,seqinfo.ini}
                                 + TrackEval GT tree + seqmap + manifest.json
make track    track/run.py      per seq: Pipeline.run(seq) = detect -> track ; write MOT + run_stats.json
make eval     eval/run.py       run_hota over the MOT outputs -> eval_results/eval_<stamp>.json
make test     tests/            pure-logic metric + adapter tests (no CV stack; CI-safe)
```

**Data → frames.** `ingest/frames.py::MotSequence` holds a sequence as ordered *frame paths* + `(H, W,
fps)` from `seqinfo.ini` — **not** decoded pixels — so a 1500-frame 720p clip costs kilobytes and never
blows up memory. The detector reads images lazily in mini-batches.

**The one shared pipeline** (`pipeline.py`). `Pipeline.run(frames)` = `detector.detect(frames)` →
`tracker.track(dets, frames)`. Homography/re-ID are `enabled: false` for V0 and skipped. Every stage is a
`Protocol`, so each is swappable via config and independently testable. (The tracker's `track(dets,
frames)` signature was aligned with the Homography/ReID stages so a later appearance tracker can reach
pixels.)

**Detection** (`detect/detector.py::YOLODetector`). Lazy `from ultralytics import YOLO`; default weights
`yolov8m.pt` (COCO); `model.predict(paths, conf, iou, imgsz, classes=[person_class], device="mps")`;
extract `boxes.xyxy/.conf` → `Detection` objects (image coords). Batched (`detect.batch`) for MPS.

**Tracking** (`track/tracker.py::ByteTrackTracker`). Lazy
`from boxmot.trackers.bbox.bytetrack import ByteTrack`. A **fresh ByteTrack per sequence** (IDs never leak
across sequences), constructed with `min_conf/track_thresh/match_thresh/track_buffer` from config and
`frame_rate` from the sequence. Detections are grouped by frame and fed **in order**, one `update` per
frame (empty frames included, so the motion model keeps stepping). boxmot requires an image array for
shape validation even though ByteTrack ignores pixels, so we pass a **blank canvas** sized to the
sequence — honest, since no pixel content is used. `TrackResults.{id,xyxy,conf}` → `Track` objects.

**MOT serialization** (`eval/trackeval_adapter.py::write_mot`). Each track box becomes one line
`frame,id,bb_left,bb_top,bb_w,bb_h,conf,-1,-1,-1`, sorted by `(frame, id)` — the MOT-Challenge tracker
format TrackEval reads. Files land in
`data/sportsmot/trackeval/trackers/<benchmark>-<split>/<method>/data/<seq>.txt`.

**HOTA** (`eval/trackeval_adapter.py::run_hota`). The *only* place TrackEval is imported (lazily).
Configures `MotChallenge2DBox` (`GT_FOLDER`, `TRACKERS_FOLDER`, `BENCHMARK`, `SPLIT_TO_EVAL`,
`SEQMAP_FILE`, `CLASSES_TO_EVAL=['pedestrian']`, `DO_PREPROC=False`), runs `HOTA + CLEAR + Identity`, and
`_summarize` digs the combined + per-sequence numbers out of TrackEval's nested result dict.

**The committed JSON** (`eval/run.py`). Fields:
- `results.tracking` — the real HOTA/DetA/AssA/LocA/MOTA/MOTP/IDF1/IDSW/Frag + `per_seq_HOTA`.
- `results.{detection_mAP, retrieval, degradation_study}` — `null` (their stages haven't run).
- `provenance` — merged from `run_stats.json`: pinned **versions** (torch/ultralytics/boxmot/trackeval),
  seed, device, detector weights+conf+iou, tracker params, per-sequence frames/fps, totals.
- `dataset` — gt_set, split, the exact sequence list (from `manifest.json`).
- `trackeval` — the eval settings used.
- `caveats` — the §6 boundaries, so the number always travels with its limits.

**Config levers** (`configs/v0.yaml`) — every knob is a config lever (one variable at a time):
`detect.{weights,conf,iou,imgsz,device,person_class}`, `track.{min_conf,track_thresh,match_thresh,
track_buffer}`, `eval.{split,mot_split,benchmark,max_sequences,max_frames}`. `max_sequences/max_frames`
are the **smoke levers** (cap to 1–2 seqs / a few frames to validate the whole path fast) and are recorded
in the JSON so a capped run can't be mistaken for the full one.

---

## 8. Reproduce

```bash
make install                 # torch/ultralytics/boxmot + TrackEval (from git); then `make lock`
make data                    # ~6.6 GB val.tar from HF -> 15 basketball-val seqs + TrackEval GT tree
make track                   # detect + ByteTrack -> MOT outputs + run_stats.json   (MPS)
make eval                    # HOTA via TrackEval -> eval_results/eval_<stamp>.json
make test                    # pure-logic tests (no CV stack)
```

Smoke first (fast end-to-end check): set `eval.max_sequences: 1` and `eval.max_frames: 60` in the config,
run `make track && make eval`, confirm a real (small) HOTA, then remove the caps for the committed run.

**Adapter sanity check.** Before trusting the number, we feed the **GT as if it were the tracker
prediction** and confirm HOTA ≈ 1.0 — proving the layout + adapter are wired correctly (a real check, not
a fabricated metric).

---

## 9. Results (this run)

Source: `eval_results/eval_20260722T210338Z.json` — COCO-pretrained **YOLOv8m** + **ByteTrack**, all
**15 basketball-val sequences** (12,557 frames), on Apple **M1 Pro (MPS)** at **6.17 fps**. This is a
**baseline to beat**, not a target hit — reported as-is (the project's honesty rule).

| Metric | Value | What it says |
|---|---:|---|
| **HOTA** | **0.301** | the headline floor (geometric mean of DetA·AssA, averaged over α) |
| DetA | 0.325 | detection accuracy — held down by false positives (see below) |
| AssA | 0.279 | association accuracy — motion-only ByteTrack fragments identities on sports |
| LocA | 0.837 | **localization is good**: when a box matches GT, it fits well |
| MOTA | **−0.395** | **negative** — FP + FN + IDSW exceed the number of GT boxes |
| IDF1 | 0.281 | identity F1 |
| IDSW | 901 | id switches |
| Frag | 2017 | track fragmentations |

Per-sequence HOTA is tight across all 15 (0.250 – 0.369), so this is a stable regime, not an outlier
average.

**What the split of numbers means — the whole point of measuring, not just scoring.**
- **LocA 0.84 vs DetA 0.33**: the geometry is fine (matched boxes fit), but *detection accuracy* is low.
  During tracking the detector emitted **~24 person boxes per frame** against only ~10 on-court athletes —
  the extra ~14 are **referees, bench, and crowd** that COCO calls "person" but SportsMOT GT excludes
  (§6). Those are false positives, and they dominate DetA.
- **MOTA is negative** for the same reason: `MOTA = 1 − (FN + FP + IDSW)/GT`, and the crowd/bench FPs alone
  push `FP > GT`. This is a vivid demonstration of *why we don't headline MOTA* — a detector that finds
  every athlete but also the crowd scores below zero, while HOTA (0.30) and LocA (0.84) still report the
  real, partial signal honestly.
- **AssA 0.28 with 901 IDSW / 2017 Frag**: motion-only association can't hold identities through the fast,
  non-linear motion and mutual occlusion of basketball (§3) — hundreds of ID switches per split.

**Context.** The public SportsMOT leaderboard sits far higher, but those numbers come from detectors
**trained on SportsMOT** and are scored on the withheld **test** set. This is a deliberately un-fine-tuned
**COCO-pretrained** detector on **val-basketball** — a *floor*. The distance from the leaderboard is not a
failure; it is exactly the headroom the next two measured steps (§10) exist to close: fine-tune the
detector (attack DetA/the FP problem) and swap in an appearance tracker (attack AssA).

---

## 10. What gets measured next (one variable at a time)

1. **Fine-tune the detector on SportsMOT** (kills the COCO-person-vs-athlete FP problem) → measure the
   DetA/HOTA lift vs this floor.
2. **Swap ByteTrack → BoT-SORT / Deep-EIoU** (appearance + camera-motion compensation) → measure the AssA
   lift on sports.
3. Only then move down the pipeline (homography → re-ID → the embedding core), each measured before the
   next begins (the project's drift rules).

Each is a single config change against this committed baseline, re-run with `make track && make eval`, and
its own timestamped JSON — so every claim of improvement is a measured delta, not an assertion.
