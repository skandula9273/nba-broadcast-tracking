# Depth round — hooptrack (NBA broadcast tracking)

A running notes file for the depth-round interview (an interviewer picks this project and drills it for
~45 min: *did you actually do the work, and do you understand the choices you made?*). Also the draft of
the eventual writeup. Update it the moment a decision is made, not after.

**Format for each decision:** the choice → what it produced (outcome) → alternatives considered → the
tradeoff → what I'd do differently.

**Framing note (say this if they treat it like a finished product):** this is a *measured eval platform*
first, a tracker second. The centerpiece — the play-embedding model and the reconstructed-vs-ground-truth
degradation study — is still ahead (V1). What exists today is the harness, the one shared pipeline, and the
first honest public-benchmark number that every later method must beat. The value on display right now is
**measurement discipline**, not a state-of-the-art tracker. Don't oversell the 0.30.

---

## Part 1 — What was actually built (the work, in order)

The order is itself a decision: **the eval harness stands up before any modeling** (the project's top
ordering rule). Nothing here is trained; it's the machinery that can *grade* a model against a public
benchmark, plus the simplest honest baseline run through it.

1. **Read the contract, grounded the environment.** Confirmed pre-V0 (every stage a `NotImplementedError`
   skeleton). Probed the machine *before* planning: Apple **M1 Pro / MPS, no CUDA**; PyPI/HF/GitHub
   reachable; 94 GB free. That decided "run a subset on-device," not "assume a GPU host."
2. **Resolved the owner-decisions up front.** Detector = **YOLO (AGPL)** (a repo-license call); scope =
   **full basketball-val**. Locked before writing code so the first `pip install` pinned the right stack.
3. **Installed + pinned the CV stack, then verified every third-party API against the *installed* version**
   (rule #1 — no fake APIs). This caught real surprises before I built on them: boxmot 22 moved ByteTrack
   to `boxmot.trackers.bbox.bytetrack`; `TrackResults` is an ndarray subclass with `.id/.xyxy/.conf`;
   TrackEval's `MotChallenge2DBox` needs a header'd seqmap and reads the GT class from column 7.
4. **Built the data path** (`ingest/fetch.py`): pulled the **official** `MCG-NJU/SportsMOT` `val.tar`
   (6.6 GB) from HuggingFace — one archive, not the 40 GB monolith — selectively extracted the **15
   basketball-val sequences (12,557 frames)**, and built the TrackEval GT tree + seqmap + a provenance
   manifest, following the authors' own `sportsmot_to_trackeval.py` layout. No scraping/relabeling; `data/`
   is gitignored.
5. **Wired the TrackEval adapter** (`eval/trackeval_adapter.py`) as the *only* module that imports
   TrackEval (lazily), so the pure-logic parts unit-test with no CV stack.
6. **Proved the adapter before trusting it.** Fed **GT as if it were the tracker prediction** → **HOTA =
   1.000**. This is the honest wiring check: if perfect input doesn't score perfectly, the harness is
   broken, and no real number can be believed.
7. **Built detect + track through the ONE shared pipeline** (`pipeline.py`): YOLOv8m on MPS (person class)
   → boxmot ByteTrack (fresh instance per sequence). The eval scores exactly the path a deployed API would
   run.
8. **Smoke-tested end-to-end on 1 sequence**, then ran the **full 15-sequence basketball-val** → the first
   committed HOTA JSON (`eval_results/eval_*.json`).
9. **Wrote it down and locked it:** a conceptual+technical explainer (`docs/increment-01-...md`), unit tests
   (7 passing, no CV stack), a pinned `requirements.lock` (TrackEval by git SHA), and the first sole-author
   commit.

**The three failure modes I hit and debugged** (the "tell me about a bug" answers — all found by *running*
the real thing, not assuming):
- **TrackEval uses `np.float`/`np.int`/`np.bool`**, removed in NumPy 2.x (we're on 2.4.6). Fixed with a
  documented compat shim that restores the exact builtin aliases; the vendored TrackEval runs unmodified.
  *Lesson:* when a pinned dependency predates a breaking upstream change, make the real API work — don't
  fork it or fake around it.
- **The combined-results key is `COMBINED_SEQ`** (singular) in trackeval 1.0.dev1, not the `COMBINED_SEQS`
  I assumed. Found by introspecting the actual return dict; the adapter now accepts both. *Lesson:* verify
  the shape of what a library returns; don't code to the shape you remember.
- **macOS `._` AppleDouble files inside the tar** inflated the manifest to 25,114 frames and broke image
  decode. The real count is **12,557**. Caught because the run's own totals didn't match the manifest —
  an **internal-consistency check surfaced it**, not a crash. Filtered them in the reader + extractor,
  cleaned 12,617 files, corrected the manifest. *Lesson:* cross-check independently-derived counts; a
  metadata bug that doesn't crash is the dangerous kind.

---

## Part 2 — Decision log (choice → outcome → alternatives → tradeoff)

### Eval harness before any modeling — smallest slice to a verifiable public number
- **Outcome:** a working detect→track→HOTA→committed-JSON loop with a real number (0.301) before a single
  weight was trained. The platform can now grade every future change.
- **Alternatives:** build a model first and bolt on evaluation later (the common order).
- **Tradeoff:** slower to something that *looks* impressive; but it means every later number is trustworthy
  by construction and attributable to one change. This is the SEC project's lesson applied.

### Detector — YOLOv8m (Ultralytics), COCO-pretrained
- **Outcome:** clean integration; the `person` class as an athlete proxy. **DetA 0.325** — held down by
  false positives (below).
- **Alternatives:** RF-DETR (Apache-2.0, better generalization), YOLOX (MIT). Both avoid AGPL.
- **Tradeoff:** YOLO is the documented SportsMOT/SoccerNet baseline recipe and CoreML-exportable (the Apple
  angle), but **AGPL-3.0 makes the whole repo AGPL** (blocks commercialization; fine for an open
  portfolio). Kept a config lever (`detect.model`) so the license/generalization ablation stays open.

### Pretrained, NOT fine-tuned — "person" ≠ "athlete" (the deliberate floor)
- **Outcome:** measured **~24 person boxes/frame against ~10 on-court athletes** — the extra ~14 are
  referees, bench, and crowd that COCO calls "person" but SportsMOT GT excludes. This is *why* DetA is low
  and **MOTA is negative (−0.395)**.
- **Alternatives:** fine-tune the detector on SportsMOT first (bigger DetA, but a training job — and it
  gold-plates the enabler, which the design doc explicitly warns against).
- **Tradeoff:** an honest floor + a named lever beats a tuned number with no baseline. Fine-tuning is a
  *later measured variable*, not a prerequisite. Reported as-is (rule #2).

### Tracker — ByteTrack (motion-only), the honest floor
- **Outcome:** **AssA 0.279, 901 ID-switches, 2017 fragmentations** — motion-only association can't hold
  identity through fast, non-linear basketball motion and mutual occlusion.
- **Alternatives:** BoT-SORT / Deep-EIoU (motion + appearance + camera-motion compensation).
- **Tradeoff:** ByteTrack needs no training and is the simplest thing that works (rule #7 — simpler first).
  SportsMOT's own paper shows motion-only underperforms on sports, so this is deliberately the floor. The
  appearance tracker is the *next* one-variable ablation.

### Increment-02 — ByteTrack → BoT-SORT (the first measured ablation; my prior was wrong)
- **Outcome:** HOTA **0.301 → 0.375** (+0.073), DetA +0.11, AssA +0.04, MOTA **−0.395 → +0.113** (flipped
  positive), IDSW 901 → 728, Frag 2017 → 2648, ~3× slower (6.2 → 2.1 fps). One variable: `track.method`.
- **The wrong prediction (the depth-round answer to "what surprised you"):** I predicted DetA was *capped*
  — same detections in, so only association could improve. Wrong. DetA/MOTA are scored on the tracker's
  **output tracks**, and BoT-SORT's stricter confirmation (higher start thresholds + `min_hits` + appearance/
  CMC gating) suppresses the crowd-FP tracks that motion-only ByteTrack blindly emits. **A tracker's
  confirmation policy is itself a precision lever** — not a passive detection consumer.
- **The trade, not a pure win:** fewer ID-switches (appearance holding identity) but *more* fragmentation
  (stricter confirmation drops briefly-low-confidence athletes). Net AssA up.
- **Honest caveat:** this is method-vs-method — it doesn't isolate *why* (thresholds vs appearance vs CMC).
  Next: ByteTrack at BoT-SORT's thresholds, and BoT-SORT with reid/cmc off, each one variable, to attribute
  the gain (and possibly keep ByteTrack's 3× speed if thresholds explain most of it).
- **Would-do-differently:** predict less, measure first — and design the ablation to isolate the driver up
  front, not method-bundle vs method-bundle.

### Increment-03 — attribution study: what drove the BoT-SORT win?
**What was done (the work, in order):**
1. Followed up increment-02's honest caveat ("method-bundle vs method-bundle doesn't say *why*") with a
   proper decomposition — the SEC-style "measure the tool, don't assume the opportunity is yours."
2. **Built the detection cache I had explicitly deferred in increment-02** (YAGNI then; the four
   tracker-only runs here crossed the threshold). `CachingDetector` is config-keyed and transparent
   (implements the `Detector` protocol, so the shared pipeline is untouched).
3. **Reproduce-check before trusting it:** cached ByteTrack reproduced the committed baseline tracker
   output **byte-for-byte** — the cache is lossless, so cached-detector ablations are trustworthy.
4. Ran four one-variable configs on **identical detections**: `bytetrack_hi` (threshold 0.45→0.60),
   `botsort_noreid` (appearance off), `botsort_nocmc` (CMC off), against the committed baseline + full.
5. Built the decomposition table + attributed the +0.074 HOTA.

**Outcome (the finding):** the +0.074 splits ~47% **detection threshold** (reachable on cheap motion-only
ByteTrack: DetA 0.325→0.395, MOTA flips positive), ~46% BoT-SORT's **confirmation cascade**, and only
**+0.005 appearance / ~0.000 CMC**. BoT-SORT's two marquee features buy nothing on basketball broadcast —
*removing* them even improved MOTA (−CMC 0.156, −ReID 0.127 vs full 0.113). And ReID cost ~4× the
association time (cached: ~95 vs ~22 fps) for that +0.005. The win is a *precision* effect (stricter
confirmation drops crowd-FP tracks), not identity modelling.
- **Alternatives:** accept the increment-02 bundle result and move on; or fine-tune the detector next
  without understanding the tracker win.
- **Tradeoff:** three extra runs (~2 h, cut by the cache) to *understand* the win vs. just banking it. Worth
  it — it retired appearance/CMC as dead weight on this domain and pointed the next lever upstream
  (detector, DetA ceiling 0.44), all measured.
- **The depth-round lesson:** a method that "wins" isn't understood until you isolate *why*; here the two
  features everyone reaches for (ReID, CMC) were inert, and a cheap threshold recovered half the gain.

### Increment-04 — detector fine-tuning: the lever the data pointed at (and the MPS saga)
**What was done (the work, in order):**
1. Acted on increment-03's conclusion (DetA ceiling 0.44 = the detector). Built `detect/finetune.py`:
   basketball-train MOT boxes → YOLO athlete labels (every 5th frame, 2,484 imgs), fine-tune yolov8m from
   COCO, then a one-variable A/B config (`v0_finetuned.yaml`: swap weights, everything else identical).
2. **Measured 1 epoch before committing hours** (as promised). It *failed loudly*: the Standard budget
   (imgsz 960 / batch 8) diverged to **NaN losses** and thrashed the 16 GB unified memory (0.5s→26s/it).
   The probe did its job.
3. Diagnosed + fixed: **AMP off** (the MPS NaN cause), **batch 4 / imgsz 640** (fits memory), and a real
   `save_dir` path bug (read `model.trainer.best` instead of a constructed path). Re-measured: clean, no
   NaN, mAP50 0.34 after one epoch.
4. Ran 20 epochs → detection **mAP50 0.987 / mAP50-95 0.795**; re-ran ByteTrack tracking with the fine-tuned
   weights.

**Outcome:** HOTA **0.301 → 0.473 (+0.172)**; **DetA 0.325 → 0.707** (doubled); **MOTA −0.40 → +0.87**.
Fine-tuned + *cheap* ByteTrack (0.473) **beats the expensive BoT-SORT bundle (0.375)** — the detector lift
is **2.3× the tracker lift**, exactly where increment-03 pointed.
- **Alternatives:** keep gold-plating the tracker (Deep-EIoU); or accept the COCO detector. The data said
  neither — the detector was the ceiling.
- **Tradeoff / honesty:** the number is *mildly optimistic* (`best.pt` model-selected on the val set that
  HOTA is scored on; weights never saw val frames) — a held-out test would be rigorous, but SportsMOT test
  GT is withheld. Stated plainly. And IDSW *rose* (901→955): clean detections mean more real tracks to
  confuse, so association quietly becomes the next relative lever.
- **The depth-round lessons:** (1) the measured arc (baseline → tracker ablation → attribution → detector)
  *earned* the conclusion — I didn't guess the lever, the committed numbers pointed at it. (2) **A warm
  1-epoch probe under-predicts:** actual training was ~7.3 h vs the ~3 h the probe implied, because MPS
  per-epoch time grows over a run — I own the ETA miss. (3) Measure before you spend: the probe caught a
  NaN/OOM failure that would have wasted hours.

### V0 complete — success floors locked (2026-07-25)
**What was done:** after the measured V0 (increments 01–04), locked the design-doc Success-criteria floors
to the committed numbers via a dated amendment: **detection mAP50 0.987** (fine-tuned yolov8m, **single class
`athlete` — ball not detected**; archived 2026-07-28 as `eval_results/detection_*.json` via `make detect-eval`,
having been a training-console figure only) and **tracking HOTA 0.301 baseline → 0.473 best**, all on
SportsMOT basketball-val (the fine-tuned mAP is model-selected on that val split → mildly optimistic, below).
- **Decision / outcome:** lock the *committed* numbers, not aspirational SOTA (the SEC honesty rule). The
  baseline (0.301) is the floor every future change must beat; the best-so-far (0.473) is the current bar;
  regressions become visible against a fixed line.
- **Alternatives:** leave floors "provisional" (no accountability), or lock an aspirational target (invites
  fudging). Rejected both.
- **Tradeoff / honesty:** locking creates accountability — a future change that regresses is now obvious in
  the JSON diff. Caveats carried into the amendment: all numbers are basketball-val (test GT withheld), and
  the fine-tuned mAP is mildly optimistic (val-set model selection).

### Increment-05 — homography: measure what's real, defer the finicky front-end
**What was done (the work, in order):**
1. Started V1: image→court homography for the top-down "moving dots". Chose **classical, no-training**
   (simpler-first) over training KaliCalib.
2. Got the calibration data (DeepSportradar, Kaggle-gated — the user provided credentials): 728 instants /
   15 arenas with GT K/R/T. `deepsport-utilities` was broken on import, so I parsed the JSONs directly.
3. Built + unit-tested the homography *machinery*: GT `H = K[r1 r2 T]`, DLT+RANSAC solver, image→court
   projection, reprojection-error metric (roundtrips to ~1e-5 px).
4. Split the stage honestly into **solver** (done) vs **registration front-end** (detect court keypoints —
   the finicky part). Measured: trivial baseline **877 px**, solver vs keypoint noise (σ=3px → **2.1 px**).

**Outcome / decision:** the solver is exact and crushes the 877 px no-registration floor; the whole
accuracy budget lives in keypoint detection (~3–5 px keypoints → ~2–4 px registration). I **deliberately
did not build a bespoke classical court-line detector** — it's a finicky multi-arena CV project, and
homography is an *enabler* the design doc says not to gold-plate.
- **Alternatives:** sink hours into classical line/template registration (gold-plating the enabler), or
  train KaliCalib (the heavy MPS path the data said isn't the depth).
- **Tradeoff / honesty:** I ship a *characterization + a concrete target* for the front-end instead of a
  weaker end-to-end number I'd have over-invested in. Stated plainly: this measures the solver, not
  auto-registration; and DeepSportradar is fixed arena cameras, not moving broadcast (the design doc's
  no-aligned-broadcast-GT boundary).
- **The depth-round lesson:** know where the depth is. Perception stages are enablers — the honest move is
  a solid, measured stage + a quantified target for the deferred piece, then spend the real effort on the
  centerpiece (the embedding core + the degradation study), not on gold-plating court lines.

### Increment-06 — the embedding core: the recall@k floor (simpler-first)
**What was done (the work, in order):**
1. Started the centerpiece (play/possession embedding for similar-play retrieval). Locked two design-doc
   open decisions: **augmentation-SSL** recall@k eval, and a **compact trajectory transformer** encoder.
2. Settled + validated the data end-to-end: `linouk23` SportVU 2015-16 (last public NBA tracking season) →
   parsed a real game (452 events, ball+10 players @25Hz) → possession tensors (T=48 × 11 × 2), court-norm.
   Corpus: 1,336 possessions / 6 games.
3. Built the **hand-feature floor first** (rule #7): flattened normalized trajectory + cosine NN, evaluated
   with the augmentation-SSL scheme (query = augmented possession, relevant = original), reusing the tested
   recall@k / MRR.

**Outcome (the floor + the target):** overall **recall@1 0.41** (~550× random). The split is the story:
raw features **solve jitter and crop** (recall@5 1.0 / 0.91) but are **at chance on court-mirror (0.001)** —
mirroring flips coordinates, so a raw trajectory can't see that a mirrored play is the *same* play. So the
trained encoder has a precise, falsifiable job: **add mirror-invariance and beat 0.41**, especially drag
mirror off the floor.
- **Alternatives:** jump straight to the transformer (no floor to beat = no honest claim); or a semantic
  PBP/hand-labeled eval (more meaningful but a manual effort — deferred as a refinement).
- **Tradeoff / honesty:** the augmentation-SSL eval measures invariance, not full "same set-play" semantics
  (design-doc's stated limitation). And the floor is honestly *easy where it should be* (jitter/crop) and
  *hard where the value is* (mirror) — so "beat the floor" isn't rigged.
- **The depth-round lesson:** build the floor before the model. The floor didn't just give a number — it
  **localized exactly what the learned model must add** (mirror-invariance), turning "train a transformer"
  into a specific, measurable hypothesis.

### Increment-06b — the trained trajectory transformer: beat the floor, then attribute it
**What was done (the work, in order):**
1. Scaled the corpus **beyond 6 games** (12 games / 2,612 possessions) and split **by game** (train 8 / val
   4) so the encoder is scored on held-out plays it never saw — a random split would leak overlapping SportVU
   events across train/val.
2. **Found + fixed a data-contamination bug while scaling** (the honesty catch of this increment):
   `_game_json` picked the extracted JSON by newest mtime, but py7zr preserves each file's stored 2016 mtime,
   so with many cached archives every call returned the *same* game. First scaled build = 2,994 "possessions"
   but only **499 unique** (two games duplicated 6× each). Caught by a unique-possession ratio check, not a
   crash. Fixed at the root (read the member `getnames()` says this archive contains); added the ratio as a
   guard in the training run. The committed 06a floor was on that latently-duplicated corpus — noted plainly.
3. Built a **compact trajectory transformer** (timestep-as-token, pre-LN, 301k params) + **InfoNCE/NT-Xent**,
   with mirror applied at `p=0.5` inside each contrastive view — the deliberate signal aimed at the floor's
   one weakness. Applied the increment-04 MPS caution: tiny model, **AMP off**, **1-epoch probe first** (clean,
   no NaN), non-finite-loss guard, then the full 300-epoch run (~3 min on MPS).
4. Recomputed the hand-feature floor on the **identical** val gallery + queries → a one-variable comparison.
5. **Attributed the win with a one-variable ablation** (`p_mirror=0`), the increment-03 move.

**Outcome (the finding):** overall recall@1 **0.618 → 0.981**, and the win is almost entirely the mirror axis
— **0.004 → 0.999** — while jitter stays saturated (1.0) and crop improves (0.851 → 0.944). The learning
curve shows mirror snapping on around epoch 50 as InfoNCE loss drops 6.69 → 0.63. **The ablation is the
depth:** with `p_mirror=0`, mirror stays at the floor (0.019) — the transformer's *capacity* buys no
mirror-invariance; the entire mirror gain is the augmentation. Two honest nuances: no-mirror gets *better*
crop (0.981 vs 0.944) — mirror-invariance costs a little temporal discrimination, a real tradeoff — and
no-mirror reaches a **lower** training loss (0.168 vs 0.632) yet is far worse on the retrieval metric (loss
is a proxy, not the objective).
- **Alternatives:** report the beat-the-floor number and move on (no attribution); tune hyperparameters on a
  separate split (more rigorous, but 12 games is a bounded corpus — deferred with the caveat stated).
- **Tradeoff / honesty:** the augmentation-SSL eval measures *invariance*, not play *semantics* — mirror
  0.999 is a near-solved invariance, not proof the embedding understands set-plays (the next, harder bar).
  MPS isn't bit-exact, but 0.004 → 0.999 dwarfs that noise. Val is both the selection and reporting set.
- **The depth-round lesson:** the floor turned "train a transformer" into a falsifiable hypothesis, and the
  ablation proved the *mechanism* (mirror-augmented positives), not just the outcome. Same pattern as the
  detector arc: don't guess the lever or bank a win — isolate the one variable that caused it. Also: a
  metadata bug that produces a plausible number (the duplicated corpus) is more dangerous than a crash;
  an internal-consistency check is what catches it.

**FAISS index (same increment):** indexed the embeddings with an **exact flat inner-product** index
(`IndexIDMap(IndexFlatIP)`, simpler-first) — on L2-normalized vectors it reproduces the brute-force cosine
retrieval **bit-for-bit**, and the training run *asserts* index-recall == brute-force-recall (the
GT-as-tracker → 1.000 discipline, applied to the index). **Second "tell me about a bug":** torch + faiss-cpu
each link their own `libomp` on macOS; `KMP_DUPLICATE_LIB_OK=TRUE` lets them import together but the process
**segfaults (139)** under real OpenMP compute (torch MPS + faiss). Reproduced it, isolated the fix
(`faiss.omp_set_num_threads(1)` — robust; import-order / `OMP_NUM_THREADS=1` are fragile), and noted the
brute-force assertion doubles as the guard against silent OMP corruption. Matters for serving too (encode +
search in one process). See `docs/increment-06b-embedding-core-trained.md`.

### Increment-07 — the reconstructed-vs-GT degradation study (the headline finding)
**What was done (the work, in order):**
1. Framed it honestly: no dataset has aligned broadcast+SportVU truth, so a **controlled degradation** —
   perturb clean GT possessions with the per-stage error budgets *already measured* (homography ~2px inc-05;
   DetA 0.707 / Frag 1847; AssA 0.317 / IDSW 955 inc-04), one variable at a time. Query = degraded GT,
   gallery = clean GT val, baseline = 1.0, so every drop is the reconstruction's cost. Reused the whole
   harness (trained encoder + FAISS + recall@k); the "augmentation" is now a measured error model.
2. Built `degrade.py` (jitter_ft, dropout+interp, id_swap, permute_players — each mapped to a measured stage)
   + `study.py` (per-source sweeps + combined realistic point + a re-ID *sensitivity*, labelled unmeasured).

**Outcome (three findings):**
- **Cost concentrates in ONE stage — tracking association.** Jitter (homography+detection localization) costs
  ~nothing (4 ft → r@1 0.999); fragmentation bridged by interp barely moves it; but each **ID-switch** costs
  ~0.12–0.15 r@1. Combined realistic (0.5ft + 0.1 drop + 2 swaps): trained **r@1 0.68** vs 1.0 on GT. So
  who-is-who over time is the bottleneck, not sub-pixel registration — a concrete steer.
- **Honest reversal — the learned encoder is *more fragile* than the naive floor under reconstruction noise.**
  The encoder that beat the floor on mirror (0.004→0.999) loses to it on id-swaps/permutation (combined 0.68
  vs floor 0.99): the floor mean-pools raw coords → order-robust (and mirror-blind); the transformer is
  order-sensitive and was trained only with order-*preserving* augmentations. Complementary failure modes,
  not "floor wins" — and the fix is the same lever that bought mirror-invariance: add id-swap/permutation
  augmentation.
- **Player-identity (re-ID) is the dominant risk — now MEASURED, not just flagged (2026-07-29).** The axis was
  a sensitivity sweep (2 wrong → r@1 0.81, full scramble → 0.02). Then jersey-OCR on real tracker output gave a
  real anchor: it resolves an identity for ~0.56 of substantial tracks, so ~round((1-0.56)·10)=**4 of 10
  players per possession** land in arbitrary slots ≈ `permute_players(4)`. Folding that measured point into the
  realistic budget (`--re-permute 4`, new `combined_realistic_with_reid`): trained r@1 **0.669 → 0.266**, floor
  0.992 → 0.890. So re-ID is now the **quantified** dominant realistic cost — and coverage ≥ accuracy (no
  jersey labels), so n_wrong=4 is a **lower bound** on the damage (raw per-id coverage 0.365 → a pessimistic 6,
  where trained falls to 0.176). The trained encoder's order-fragility (finding 2) is what makes re-ID error so
  expensive for it specifically.
- **Alternatives:** claim reconstruction "barely hurts" from the jitter/dropout results alone (misleading —
  omits the order-corruption stages); or wait for aligned broadcast GT that doesn't exist. Rejected both.
- **Tradeoff / honesty:** a controlled proxy with stated px→ft / IDSW→swaps mapping assumptions, not the real
  end-to-end pipeline; the sweeps (not one point) carry the finding; the re-ID axis is labelled unmeasured.
- **The depth-round lesson:** the study connected the perception error budgets to the retrieval cost they
  impose, localized it to one stage, and *surprised* me (the fancy encoder is less robust than the baseline
  where it wasn't trained to be) — then named the two next levers. Measuring beats assuming, again.

### Increment-08 — order-robustness augmentation: the lever works, and exposes a tradeoff
**What was done:** acted on inc-07's named lever — add order-perturbation augmentation (reuse the study's
*exact* `permute_players`/`id_swap` as train-time augs; "train on what you measure") behind `p_permute`/`p_swap`
levers in the shared training loop, default-off (preserves the inc-06b/07 baselines). Ran two strengths (mild
p=0.25, aggressive p=0.5) end-to-end on *both* axes: clean recall@k and the degradation study, one variable.

**Outcome (three findings):**
- **The lever works — hypothesis confirmed exactly.** id-swap and full-permute robustness → **1.000 at every
  severity** (baseline 0.44 / 0.02); combined realistic **0.68 → 1.0**, now *beating* the floor. The inc-06b
  trick (train on a transform → gain invariance) generalizes from court-mirror to player-order.
- **It's a regime switch, not a dial.** Even *mild* order-aug flips the encoder fully into
  permutation-invariance — mild ≈ aggressive on both axes; no cheap partial order-robustness.
- **The cost is TEMPORAL robustness, and it's steep.** Crop **0.944 → ~0.45**, high-dropout 0.92 → ~0.39, while
  spatial invariances survive (jitter 1.0, mirror 0.999 → ~0.95). The order-blind encoder used player-slot
  identity as an *anchor* to re-match a temporally-degraded play; permutation-invariance removes that anchor,
  so crop/dropout now bite. Order-invariance and temporal-crop robustness are **entangled at fixed capacity**.
- **Alternatives / next:** the augmentation fix *spends capacity* (why crop pays). Better routes it points at:
  a **permutation-invariant architecture** (per-player tokens + symmetric pool — DeepSets/set-transformer) to
  bake in the invariance without the crop cost; or **team-structured** permutation (within-team only, via cheap
  jersey colour). Regime choice is deployment-driven: off broadcast association is the weak stage (AssA 0.317),
  so the invariant regime (0.68 → 1.0) is right; with stable IDs, keep the order-sensitive regime's crop recall.
- **Tradeoff / honesty:** two points + a sharp threshold, not a fully-resolved frontier; train-on-what-you-
  measure (legit, same as mirror) measures the *modelled* corruption; same proxy caveats as inc-06b/07.
- **The depth-round lesson:** a lever that "works" isn't the end of the story — measuring *both* axes turned a
  clean win into an honest tradeoff (order-robustness costs temporal robustness), and the *mechanism*
  (slot-identity as a temporal anchor) is the real insight, not the number.

### Increment-09 — permutation-invariant architecture: a partially-negative result, measured
**What was done:** built `SetTrajectoryTransformer` (per-entity tokens, factorized time/entity attention with
no player position, symmetric player mean-pool, ball distinguished by a type-embedding) — **exactly**
permutation-invariant by construction (unit test: `encode(P)==encode(permute(P))` to 1e-5; baseline confirmed
NOT invariant). Trained with order-augmentation OFF (`p_permute=p_swap=0`) — the invariance is purely
architectural — and measured both axes vs the inc-06b/08 baselines. Hypothesis (from inc-08): the architecture
gets order-robustness *without* the crop cost.

**Outcome (split verdict — the honest kind):**
- **Order-robustness, exact and free ✓.** permute → **1.0 at every severity by construction** (no aug), id_swap
  → 0.99 (the 0.01 gap = the mid-possession discontinuity the per-entity temporal attention sees), combined
  realistic 0.68 → **0.998** (beats floor). A real edge over augmentation: a *guarantee* over all 10!
  permutations, not a learned approximation, and zero augmentation.
- **But the crop cost is NOT escaped ✗ — hypothesis refuted.** crop **0.487** ≈ augmentation's 0.446 (baseline
  0.944); high-dropout collapses identically. Building the invariance in didn't buy back temporal robustness.
- **The insight: the crop cost is *intrinsic to order-invariance*, not the method.** Convergent evidence — the
  augmentation route (full d128) and the architecture route (d64) both collapse crop to ~0.45–0.49 and both
  collapse high-dropout, while only the order-sensitive baseline keeps 0.94. Player-slot identity was a
  temporal anchor for re-matching a cropped play; removing it (however you do it) costs crop. Order-invariance
  and temporal-crop robustness are fundamentally entangled in this representation.
- **Tradeoff / honesty:** the set-arch runs at d64/150ep (per-entity tokens OOM at d128/batch512 on MPS — the
  inc-04 tradeoff again) vs baseline d128/300ep — a capacity confound, named. But the augmentation route at
  full d128 collapsed crop just as hard, so the finding tracks the invariance, not the capacity. Also hit and
  fixed a too-strict FAISS check (near-tied neighbours reorder under float non-associativity on poorly-separated
  embeddings; now tolerance-based, records the diff — exact 0.0 on the well-trained models).
- **The depth-round lesson:** a "next lever" can be a *partial* win — measuring both axes turned "architecture
  fixes it" into "architecture gives the invariance for free but confirms the tradeoff is intrinsic." A clean
  negative result (with the mechanism) is worth more than a hoped-for positive one, and I reported it as-is.

### Provenance fix — a hand-maintained detector field went stale; fixed the mechanism, re-ran (2026-07-28)
**The bug:** the committed fine-tuned eval JSON (`eval_20260725T220105Z.json`, HOTA 0.473 / DetA 0.707) carried
`provenance.detector.fine_tuned: false` and a "COCO-pretrained person-class proxy" note — while its own
`config.detect.weights` pointed at `weights/finetuned/best.pt`. Config and provenance disagreed. Ground truth =
**fine-tuned**: the detector resolves `cfg.weights or DEFAULT_WEIGHTS`, so a non-null weights path is the
fine-tuned checkpoint, and DetA 0.707 is the increment-04 fine-tuned number (the COCO baseline was ~0.325).
So the provenance was the wrong one.
- **Root cause (the real bug):** `fine_tuned` and its note were **hardcoded literals** in `track/run.py`'s
  `run_stats` dict — every field around them derived from `cfg.detect.*`, but these two didn't, so on the
  fine-tuned run they stayed at their COCO defaults. The identical claim was independently hand-maintained in
  `eval/run.py`'s `CAVEATS[0]` ("NOT fine-tuned").
- **The fix (mechanism, not symptom):** `_detector_provenance(cfg.detect)` now derives the whole detector block
  from the resolved weights — `fine_tuned = (weights != DEFAULT_WEIGHTS)`, note generated from that fact. The
  eval caveats read the regime back from that single source (`_caveats(cfg, provenance)`); the tracker caveat
  now derives from `cfg.track.method` (it had the same latent bug — it said "ByteTrack motion-only" regardless
  of tracker). Config and provenance can no longer disagree.
- **The artifact — re-ran, did NOT hand-patch.** Re-running was feasible (fine-tuned weights + prepared data +
  detection cache all present) and **deterministic** (cached detections + seeded tracker → tracker files
  byte-for-byte identical), so the regenerated `eval_20260728T201221Z.json` has HOTA/DetA/AssA/MOTA/IDF1/IDSW
  **identical** to the old artifact — only the provenance metadata is corrected. Retired the old wrong JSON (no
  doc references it by filename). Chose re-run over a correction note *because* it was cheap and lossless and it
  proves the mechanism fix yields a correct artifact end to end.
- **The lesson:** provenance that's hand-maintained goes stale the moment the run it describes changes — a
  metadata field that can disagree with the config beside it is a latent lie. Generate it from the run's own
  config. Same family as the inc-06b `_game_json` mtime bug and the inc-01 AppleDouble count: an
  internal-consistency mismatch (config vs provenance), which is exactly how it was caught.

### Repo honesty audit — withdrew an inflated headline, scoped the eval, documented what actually runs (2026-07-27/28)
A documentation-accuracy sweep from re-reading the committed claims against the artifacts and the code.
- **Withdrew the recall headline "0.41 → 0.98".** The 0.41 was the inc-06a floor computed on a corpus later
  found corrupted by the `_game_json` mtime-duplication bug; the honest same-split floor is **0.62**, so the
  headline is now **0.62 → 0.98** — and I stated the withdrawal, not just the new number.
- **Scoped the retrieval eval honestly in the README:** its "relevant" item for query *i* is corpus item *i*
  itself under a known augmentation family, so it measures **instance-level invariance, not semantic play
  similarity** (there are no play-type labels in the repo). And surfaced the inc-07 reversal there — the
  zero-parameter floor beats the trained encoder under every association-error mode — and **demoted the 0.98**.
- **"What runs today" audit,** each claim verified in the code: `retrieve/` imports nothing from
  `detect/track/homography/pipeline` (it is fed from SportVU GT); `Pipeline` is built with `homography=None,
  reid=None` (both `enabled:false` in every config); homography is **solver-only** (no keypoint front-end → no
  court coords from an unlabelled frame); `reid`/`analytics` raise `NotImplementedError`; `serve /track` is a
  501 stub that imports no pipeline. Added a "What runs today" box + a marked mermaid, removed a dead
  `make demo` + the empty `demo/`, fixed serve's docstring (it claimed to "call the SAME pipeline"). Published
  the repo public.
- **Lesson:** a portfolio README is a claim surface — re-derive every headline from the artifacts, and describe
  what *executes*, not what's aspirational. The 0.98 is real but narrow; saying exactly that is the honest move.

### Detection mAP — archived a docs-only figure as a real artifact; fixed athlete/ball + the caveat placement (2026-07-28)
`detection_mAP` is null in every tracking eval JSON — the **0.987 mAP@50 existed only in the docs**, a
training-console figure. Chose to **re-run, not delete**: built `detect/eval.py` (`YOLO.val()` on the existing
fine-tuned weights + the basketball-val YOLO labels — no fine-tuning) + `make detect-eval`, which **reproduced
the figure exactly** (mAP@50 0.9872 / mAP50-95 0.7951 / P 0.971 / R 0.958) into a committed
`eval_results/detection_*.json`. Corrected two labels everywhere the number appears: it is **single-class
`athlete`** (`finetune.py names={0:'athlete'}`), **not "player/ball" — the ball is not detected** (the
design-doc's "Player/ball detection" was wrong); and the **model-selection caveat now sits next to the number**
(best.pt was selected on the same basketball-val split HOTA is scored on → mildly optimistic), not three
sections away, and baked into the artifact's caveats.
- **Lesson:** "committed number" means an *artifact*, not a console line — a headline that lives only in prose
  is unverifiable. And a metric's label (which class, which split, selected how) is part of the number.

### Semantic transfer probe — does the augmentation-SSL encoder capture play *content*? (no) (2026-07-28)
**What was done:** the committed retrieval eval's positive is the query's own index → it scores instance-level
*invariance*, not *content*. Built `retrieve/semantic_probe.py`: coarse buckets derived from the (T,11,2)
tensors themselves (no external labels; thresholds as levers in `configs/semantic_probe.yaml`) —
transition/halfcourt (ball along-court advance), initiation side L/M/R (ball across-court entry),
ball-handler-change low/high (nearest-player-to-ball transitions). Metric = **precision@5** (fraction of a
query's top-5 sharing its bucket; relevant = same-bucket, **|relevant|>1** — not the index-identity; random ≈
bucket prevalence). Scored on the held-out val split (858) for **floor / trained / random**.

**Outcome (the finding, reported as-is): floor > trained ≈ random on ALL three schemes.** trained precision@5:
transition **0.513** (rand 0.496, floor 0.597), initiation-side **0.382 ≈ rand 0.383 (at chance)**, handler
**0.565** (rand 0.511, floor 0.616). The encoder that scores **recall@1 0.98** on instance-invariance encodes
little play content. The side result is the sharpest tell: it is *exactly* at chance because the encoder is
court-mirror-invariant (inc-06b) → left/right is destroyed **by design**, while the un-invariant floor recovers
it (0.49). All buckets came out ≥10% (no merge needed). Thresholds picked on the definitions, run once, not
tuned.
- **Alternatives:** pick a flattering metric (hit-rate@5 ≈ 0.9), or tune thresholds to lift the number.
  Rejected — precision@5 with a prevalence baseline is the honest frame (literal recall@5 ≈ 0.006 is reported
  too, and is degenerate).
- **Honest note:** the trained encoder isn't persisted, so it's reproduced in-process with the committed
  inc-06b recipe (same as study.py); the "don't train anything" tension was flagged openly, not silently.
- **The depth-round lesson:** an SSL eval whose positives are the query's own augmentations measures what you
  *trained for* (invariance) — it can look excellent (0.98) while the representation is semantically thin.
  Probe content with a *different* relevant set and let the number come out mediocre; that mediocrity **is** the
  finding, and it retires any "the embedding understands plays" over-claim.

### Court-keypoint front-end — a learned detector beats the 503px floor (503 -> 40px) (2026-07-28)
**What was done:** built the deferred homography front-end (inc-05 built + measured the solver; this supplies
its correspondences). Harness-before-modeling: 7 canonical court line-intersections, GT image locations from
DeepSportradar calibration, and a reprojection-error eval for any detector (inc-05's metric). Trivial floor
(global-mean keypoints) = **503px**. Then trained a KaliCalib-lite detector — resnet18(pretrained) encoder +
upsample decoder -> 7 heatmaps, masked MSE, **split by ARENA** (must generalize to unseen cameras, not memorize
one). Wired the trained detector into the `CourtHomography` stage (`learned_register`) so the pipeline
auto-registers a frame -> `court_xy`.
- **Outcome:** on **held-out arenas**, reprojection error **503px -> 40px median**, 99% of frames registered
  (508s / 40 epochs on MPS). Verified end to end (an unseen-arena image -> court coords). inc-04 MPS discipline
  held: the 1-epoch probe caught the undertrained 0-solve state, and a nan-median bug (a degenerate H from
  imperfect keypoints made `best < nan` always false, so best-weights never saved) — fixed by dropping
  non-finite reprojection errors.
- **Alternatives:** classical line-detection + template matching (finicky correspondence — the design-doc's
  deferral reason); a per-arena mean H (aces fixed cameras, useless for broadcast — rejected as leakage).
- **Tradeoff / honesty:** DeepSportradar is FIXED arena cameras, not moving broadcast — 40px is on unseen
  *arenas* but NOT proven on broadcast (a stated boundary). Weights gitignored (a local-only model artifact).
- **The depth-round lesson:** harness-before-modeling again — the 503px floor + the reprojection metric made
  the detector's job falsifiable, and the arena-split kept the number honest (generalization, not camera memory).
- **Broadcast generalization (the honest negative):** added domain augmentation (perspective + photometric)
  and retrained — it *improved* held-out-arena accuracy **40px -> 16px** (100% solve; augmentation forces
  viewpoint/lighting robustness that helps even on unseen arena cameras). But it **did NOT bridge broadcast**:
  on SportsMOT both models produce ~no confident keypoints (mean peak ~0.12, <=2/35 above 0.3, need >=4 to
  register). The measured conclusion: broadcast registration is a **data** limitation, not a modelling one —
  DeepSportradar's arena cameras are too far from broadcast, and no augmentation crosses that gap without
  broadcast training data (aligned court GT, which doesn't exist for this project). Reported as-is; the
  augmented model is now the default (strictly better within the arena domain).

### Jersey-OCR coverage — the win was evidence, not enhancement (0.175 -> 0.73), and CLAHE hurt (2026-07-29)
- **The problem:** jersey number is re-ID's only *individual*-identity signal (OSNet appearance can't split
  same-uniform teammates — cosines smear 0.32–0.87). The initial per-track coverage was ~40%, measured ad-hoc.
- **Harness first (again):** built `reid.eval_jersey` (`make jersey-eval`) — run easyocr over **GT-boxed
  athletes** (SportsMOT basketball-val), so the number is the OCR stage's own capability *given correct
  tracking*, with tracker error factored out. SportsMOT has no jersey labels, so I measure **coverage** (does a
  track get *a* confident majority number) not accuracy, using **cross-frame vote consensus** as the precision
  proxy (a real number reads consistently across frames; noise doesn't). Committed timestamped JSON.
- **The anti-cheat:** the precision thresholds (`min_conf` 0.4, `min_votes` 2) were **held fixed** across the
  whole ablation, so coverage can only rise from better *evidence*, never a lowered bar. Cheap coverage (drop
  the thresholds) is exactly the trap here.
- **Additive ablation + attribution (one variable at a time — like inc-03).** The big jump came from a config
  that changed two knobs at once (crop count 15→40 **and** even-sampling), and CLAHE contrast-normalization was
  carried along — so I disambiguated: **more crops** alone 0.175→0.35, **even-sampling** alone (spread the 15
  crops across the whole possession so a camera-facing frame is caught) 0.175→0.30 — two *independent evidence
  levers* that combine to 0.60. Then **removing CLAHE** lifted 0.60→**0.70** (and crop-read-rate to the best in
  the sweep). CLAHE **hurt** both alone (0.175→0.125) and in-regime: it manufactures false digit-like reads.
- **Result (winner on all 15 seqs, 150 GT tracks):** coverage **0.733**, high-consensus coverage **0.467**,
  per-crop read rate 0.162; numbers read are plausible jerseys (15, 6, 5, 32, 30, 23…). The simpler no-CLAHE
  config won and is promoted to the `JerseyOCR` defaults — **rule #7 (simpler-first) vindicated by measurement**.
- **Honest framing:** 0.73 is coverage on **GT boxes** — the OCR ceiling given correct tracking. It's coverage,
  not verified accuracy. The old "~40%" was ad-hoc on tracker output — a different, dirtier basis; the new
  number is reproducible and isolates the stage. The depth-round lesson repeats: *measure the stage in
  isolation, hold the precision bar fixed, and attribute the win to the actual lever.*
- **Then measured the real operating point (`make jersey-eval-tracker`, `bytetrack_ft` HOTA 0.473, all 15
  seqs):** raw per-id coverage **0.73 → 0.365** — but the drop is **association, not OCR**. The tracker
  fragments 150 GT ids into **949 tracker-ids** (6.3×), so most ids are short fragments that never accumulate
  `min_votes`; the per-crop read rate is **unchanged (0.162 → 0.161)**, and among **substantial tracks (≥10
  crops) coverage is 0.56**. Added `coverage_substantial` precisely to separate fragmentation from OCR
  capability. This is the honest re-ID operating point, and it **re-derives inc-07's headline from a different
  stage**: reconstruction cost concentrates in **tracking association**, not the per-frame perception — better
  association (fewer fragments/id-swaps) recovers most of the jersey-coverage gap, not a better OCR model.
  Committed `eval_results/jersey_ocr_tracker_*.json`.
- **Then built the recovery lever the diagnosis implied — fragment stitching (`reid/stitch.py`,
  `make jersey-eval-stitch`).** Since the loss is fragmentation, gap-close a player's split ids so jersey votes
  pool: link id B to id A when B *starts* just after A *ends* (gap ≤30f) and B's first box is spatially near
  A's last box (≤2 box-diagonals), greedy earliest-first + union-find. Chose spatiotemporal linking over
  appearance clustering deliberately: it only links **temporally disjoint** fragments (never co-present, so it
  can't merge same-uniform teammates — OSNet's fatal weakness here). Measured on all 15 seqs: **949 → 480 ids**
  (2× consolidation), raw coverage **0.365 → 0.479** (≈⅓ of the gap to the 0.73 ceiling). The **precision check
  that makes it credible: consensus ROSE 0.252 → 0.294** — if stitching were over-merging teammates, two numbers
  would compete and consensus would fall; it rose because correct same-player merges reinforce the majority.
  Honest limits: a heuristic gap-closer recovers a *third*, not all — long gaps and true id-swaps need real
  re-ID or a less-fragmenting tracker; and it's still coverage, not accuracy. Closes the arc: **diagnosis
  (association is the cost) → lever (fix association) → measured recovery, with a precision guard.**

### End-to-end wire — REAL pipeline output through the retrieval core, at last (2026-07-29)
- **The gap it closes:** the centerpiece was fed *only* SportVU GT; the inc-07 headline **simulated** perception
  error with an order-of-magnitude budget. The two halves never touched, so a reviewer could fairly say the
  degradation numbers were a *model* of reality, not reality.
- **What I built:** `reconstruct.tracks_to_tensor` (pipeline `Track` output → the (T,11,2) tensor) +
  `end2end.py` — run the **actual** tracker output (`bytetrack_ft`, HOTA 0.473) through the adapter and the
  **real FAISS index**, measuring reconstructed-vs-GT retrieval per 48-frame window (query = tracker tensor,
  gallery = GT tensors, relevant = same window). 255 windows over the 15 basketball-val seqs.
- **Result: floor recall@1 0.80** (vs GT-self 1.0, chance 0.004). Real perception error costs ~20 points of
  recall@1 — and it's a **harder hit than inc-07's simulated budget implied** (floor stayed ~0.99 there): the
  proxy was optimistic, which is exactly why wiring the real path mattered. (Different domain/coords, so it's a
  qualitative comparison, not a number-for-number one — stated.)
- **Blockers made concrete (not hand-waved) by actually running it:** no ball (entity 0 zeroed — single-class
  detector); no broadcast homography (image-coordinate foot-points, not court — so the SportVU-trained
  transformer is **out-of-domain**, and the coordinate-agnostic **floor** is the only valid encoder here); tracker
  fragmentation (top-10 most-present ids, canonical x-order — a dropped/duplicated player is real error that
  shows as tensor mismatch). The single missing piece the number points at: a **broadcast-domain encoder**
  (needs working homography or broadcast court GT).
- **The honesty:** I led with the floor because using the domain-mismatched trained transformer would produce a
  misleading number; the FAISS index is the real one the committed evals use; the sanity check (GT self-retrieval
  = 1.0) confirms the windows are distinguishable, so 0.80 is reconstruction cost, not window ambiguity.

### Broadcast-domain encoder — the trained encoder made valid end-to-end (2026-07-29)
- **What the 0.80 pointed at:** the trained transformer was out-of-domain on image-coord broadcast tracks, so
  end2end had to fall back to the floor. The fix is not a new architecture — it's the training DOMAIN.
- **What I did:** `broadcast_encoder.py` trains the SAME inc-06b architecture/objective on the **image-coord
  broadcast window tensors** (SportsMOT GT), augmentations retargeted to the reconstruction error that dominates
  (jitter + order-perturbation + crop, `p_mirror=0` since image space isn't court-symmetric). Held out **BY
  GAME** — train on 3 games' GT windows, evaluate reconstructed-vs-GT on the 4th game (never trained on it).
- **Result: the in-domain encoder beats the floor** on the held-out game — recall@1 0.857 → **0.881**, recall@5
  0.857 → **0.905**, recall@10 0.857 → **0.952**, MRR 0.862 → **0.901**. Consistent across every k (most at
  higher k), so it's not a single-window fluke. Trained self-retrieval = 1.0 (sanity).
- **Honest limits:** tiny broadcast supervision (213 windows / 3 games) and a small val set (42 windows on one
  game), so this is a **proof-of-concept + honest measurement**, not a production encoder — the r@1 margin
  (+0.024) is a few windows, but r@5/r@10/MRR all improve, which is more robust. The lesson: the trained
  centerpiece CAN be made valid end-to-end; the blocker was domain/data, not the method — more broadcast data
  (and working homography for court coords) extends it. Only variable changed vs inc-06b is the training domain.

### Semantic retrieval — validated as ACHIEVABLE; the SSL objective was the limitation (2026-07-29)
- **The gap:** recall@1 0.98 is instance-invariance self-retrieval; the probe showed the SSL encoder ≈ random
  on play-type buckets. So "find similar plays" — the actual product — had no positive evidence. A fair
  reviewer's #1 shot: the centerpiece isn't shown to do semantic retrieval.
- **The test (one variable):** train the SAME architecture with a **supervised-contrastive (SupCon)** loss on a
  real play-type axis (transition vs halfcourt = fast break vs set offense), everything else identical to the
  SSL recipe (arch, augmentation, by-game split, seed) — so the ONLY change is the objective. Score held-out-
  **game** semantic precision@5.
- **Result: supervised 0.942** vs SSL 0.513 (≈ random 0.496) vs floor 0.597. Semantic play retrieval **is
  achievable** and generalizes to unseen games; the eval **is sensitive** (it cleanly separates 0.94 from 0.50);
  and the **SSL objective was the limitation, not the task** — instance-invariance training simply doesn't
  encode content, a supervised signal does.
- **Honesty:** the bucket is a *derived* proxy (early ball advance), not an annotated set-play, so this validates
  *achievability + eval sensitivity + generalization to held-out games*, not discovery of real designed plays.
  Stated as such. `make retrieve-semantic-validate`; +2 SupCon tests. The path forward for the real product:
  real play-type labels (or richer proxies) + this supervised objective.

### Jersey-OCR ACCURACY — resolution isn't the bottleneck; pose/blur/occlusion is (2026-07-29)
- **The gap #5:** coverage says how often a track gets *a* number, never whether it's *right* (no jersey
  labels). The study leaned on coverage as a lower bound on identity; accuracy was unmeasured.
- **How, without labels:** `eval_jersey_accuracy.py` (`make jersey-accuracy`) renders digits with KNOWN labels,
  degrades them to broadcast crop heights, and measures easyocr accuracy vs height — anchored to the REAL
  jersey-region height distribution from 118,825 GT boxes. Synthetic clean font = an OPTIMISTIC bound (stated).
- **Accuracy vs height:** 48px **1.0**, 32px 0.94, 24px 0.87, 20px 0.63, 16px 0.50, 12px 0 (unreadable). The
  **real** jersey-region heights are **median 63px** (p10 50, p90 83) — i.e. mostly in the near-perfect regime.
- **The reframing (the actual finding):** at real broadcast resolution the OCR is **not** resolution-limited —
  synthetic accuracy is ~1.0 at 63px, yet real coverage is only ~24%. So the gap is **pose (number not facing
  camera), motion blur, and occlusion**, not crop size or OCR capability. Actionable: prioritize frontal-frame
  selection / deblurring / a pose-robust number model — **not** upscaling. Honest limit: real accuracy needs a
  labeled broadcast set (SoccerNet jersey GT) and will sit below the synthetic ceiling, but the
  resolution-is-sufficient conclusion is robust (real heights >> the resolution-limited regime).

### Ball tracking — COCO 'sports ball', no training; measured ~24% coverage → true possessions where visible (2026-07-29)
- **The gap:** single-class athlete detector has no ball, so analytics could only do spacing/phases — true
  possessions/shots need ball control. The pragmatic path needs **no new training**: COCO yolov8m already has a
  'sports ball' class (32).
- **Measured, not assumed:** `detect/ball.py` (BallDetector) + `eval_ball.py` (`make ball-eval`) — coverage on
  2516 sampled SportsMOT frames: **24% @ conf 0.25** (15% @ 0.35, 8% @ 0.5), mean conf 0.41. So COCO finds the
  basketball in ~a quarter of frames — broadcast basketballs are small/fast/occluded, and there's no ball GT so
  it's coverage, not accuracy (same honest framing as jersey OCR). A single ball-friendly clip read 45%; the
  full-set 24% is the honest number.
- **What it unlocks (and doesn't):** `analytics.ball_possession` attributes the ball-handler (nearest player to
  the ball) and segments **true possessions** (handler runs; a change = pass/turnover) on the frames the ball is
  visible — wired into `POST /track` behind `detect.ball`. Shots still need a hoop/trajectory model (stated, not
  faked). The honest bound: a *quarter*-coverage ball supports partial possession analytics; robust ball
  tracking needs a basketball-specific detector + annotations. Reported as-is.

### Serving latency baseline — the missing operating point for the V2 Pareto (2026-07-29)
- **Why:** "serving optimization + Pareto frontier" is the headline V2 item, but there was no *before* number —
  you can't build a Pareto without a baseline on the real hardware. `serve/bench.py` (`make serve-bench`) times
  the deployed `detect → track` path on real frames, after a warmup pass (so model-load / first-call MPS
  compile don't pollute steady state).
- **Result (MPS, 200 frames, yolov8m @ imgsz 1280):** end-to-end **9.9 fps (101 ms/frame)** — **detection is
  94% of it** (94.5 ms/frame), tracking is ~free (6.5 ms/frame, 155 fps). Below the 30 fps real-time bar
  (~33 ms/frame), so the concrete lever is the *detector* (imgsz ↓, smaller model, quantization — each trades
  mAP for fps), not the tracker. The jersey-OCR re-ID stage is a separate, far heavier cost (easyocr,
  minutes/clip; measured in the jersey runs).
- **Honesty:** stage-level throughput, not per-frame percentiles (the detector batches internally); encode/FAISS
  is sub-ms and not the serving cost. This is a baseline to optimize against, stated as such — the number is
  config-bound (imgsz 1280 is deliberately large for accuracy), which is exactly the knob V2 would sweep.

### Headline metric — HOTA (not MOTA or IDF1)
- **Outcome:** **HOTA 0.301** (√(DetA·AssA), averaged over localization thresholds). The project *earned*
  the choice: **MOTA came out at −0.395**, because a detector that finds every athlete plus the crowd has
  FP > GT. A metric that goes negative on a partially-working system is a bad headline.
- **Alternatives:** MOTA (detection-dominated), IDF1 (identity-dominated).
- **Tradeoff:** HOTA balances detection and association explicitly and correlates with human judgment; it's
  the SportsMOT / modern-MOT standard. We still report MOTA/IDF1 — the *split* is the story.

### Benchmark split — SportsMOT val (not test)
- **Outcome:** a locally reproducible number.
- **Alternatives:** test (leaderboard).
- **Tradeoff:** test GT is withheld behind Codalab, so a committed, one-command-reproducible HOTA needs
  val. Comparable to *published val baselines*, not the leaderboard — stated in the JSON `caveats`.

### One shared pipeline — eval and serve call the same path
- **Outcome:** the committed 0.301 describes the exact code a deployed API would run.
- **Alternatives:** a separate eval-only path (faster to hack).
- **Tradeoff:** less flexibility (eval can't shortcut a stage); worth it — this is the single most
  important honesty decision in the codebase (again, the SEC lesson).

### `DO_PREPROC=False` in TrackEval
- **Outcome:** all GT boxes scored as-is.
- **Alternatives:** the MOT17 default `True` (distractor-class removal + zero-marked filtering).
- **Tradeoff:** SportsMOT has no distractor classes; `False` is the authors' comparable setting. Verified
  against the installed TrackEval source, not assumed.

### Frames as file paths (not decoded pixels)
- **Outcome:** a 1500-frame 720p sequence costs kilobytes in the pipeline; the detector reads images lazily
  in mini-batches. No OOM on long sequences.
- **Alternatives:** load whole sequences into memory (a 1500-frame clip ≈ 4 GB decoded).
- **Tradeoff:** two disk reads for a few stages vs. blowing up memory. Easy call.

### Blank canvas for the motion-only tracker
- **Outcome:** ByteTrack runs correctly without ever touching pixels; boxmot's shape-validation is
  satisfied by a zero canvas sized to the sequence.
- **Alternatives:** thread real frames through (needed for appearance trackers, wasteful for ByteTrack).
- **Tradeoff:** honest (ByteTrack genuinely ignores pixels) and fast. The `track(dets, frames)` signature
  already anticipates BoT-SORT needing real pixels — the interface didn't have to change for the next step.

### Eval scores exactly what was tracked
- **Outcome:** the harness writes a seqmap from the produced tracker files, so it's robust to smoke caps
  and partial runs; for the full run it's all 15. The JSON records `n_sequences_evaluated` so a capped run
  can never be misread as the full split.
- **Alternatives:** always eval the full GT seqmap (crashes if any sequence is missing).
- **Tradeoff:** none worth mentioning — it's strictly more honest and more robust.

---

## Part 3 — The questions they'd actually ask (and my answers)

### "What was your baseline, and is 0.30 good?"
It's a **floor, not a target** — and low *on purpose*. The story is in the split: **LocA 0.84** (matched
boxes localize well — geometry works), **DetA 0.33** (dragged by ~14 crowd/bench false positives per
frame from a COCO detector that was never told what an "athlete" is), **AssA 0.28** (motion-only
association fragments identity on fast basketball). The public leaderboard is far higher, but those use
detectors *trained on SportsMOT* and score the *test* set. The distance from the leaderboard isn't failure
— it's the exact headroom the next two measured steps exist to close.

### "Tell me about a failure mode you didn't expect and how you debugged it."
The manifest said 25,114 frames; the tracking run's own totals said 12,557. I didn't trust either — I
looked for *why they disagreed*. macOS had written `._`-prefixed AppleDouble twins next to every image
inside the tar; `pathlib.glob("*.jpg")` counted them (doubling the manifest) while `ls` hid them, and the
image decoder choked on them. Fixed at the source (skip hidden files on extract + read), cleaned 12,617
files, corrected the manifest. The tell was an **internal-consistency check**, not a crash — the dangerous
bugs are the ones that produce a plausible wrong number.

### "You depend on TrackEval — what happened when you first ran it?"
It crashed on `np.float`, which NumPy removed in 1.24 (we're on 2.4.6). The wrong fix is to downgrade NumPy
(breaks torch/ultralytics) or fork TrackEval. The right fix: a small compat shim that restores the exact
builtin aliases (`np.float = float`, etc.) — behavior-preserving, keeps the pinned TrackEval running
unmodified, documented as such. Rule #1 is "make the real API work," not "avoid the hard API."

### "What's your validation strategy, and how do you know the harness itself is right?"
- **Public benchmark (SportsMOT), scored through the same pipeline the API would use** — never a
  second eval-only path.
- **A wiring check that can't lie:** GT-as-tracker → HOTA 1.000. If perfect input doesn't score perfectly,
  nothing downstream is trustworthy.
- **The harness can't fabricate:** unrun stages are written as `null` + a `status` string; a scaffold run
  can't be misread as a result.
- **Reproducible:** fixed seed, pinned lockfile (TrackEval by git SHA), every knob a config lever,
  timestamped JSON committed per run, one-command rerun.

### "What would you do next, and why that order?"
The *data* dictated V0's order and delivered: tracker ablation (0.375) → attribution (the win is
confirmation, not appearance/CMC) → **detector fine-tune, the lever the numbers pointed at → HOTA 0.473,
DetA doubled**, and fine-tuned+ByteTrack beat the expensive BoT-SORT bundle. V0 is done (detection mAP
0.987 + tracking HOTA 0.473, both committed). Next is **V1**: homography (court coords, reprojection error)
→ re-ID (player identity) → the **embedding core** (the trained centerpiece, recall@k) → the
**reconstructed-vs-GT degradation study** (the finding the whole project exists for), each measured before
the next begins. One nuance the data just surfaced: on clean detections, IDSW rose — so association
(dead weight on noisy boxes) may now matter, a hypothesis for a future ablation. The through-line: I don't
guess the next lever, I let the committed numbers point at it.

---

## Meta-answer — "how did you keep yourself honest?"
A written engineering contract with non-negotiable rules — no fake APIs, never cherry-pick numbers, pair
every choice with its reason, one variable per ablation, reproducible by default — and a design doc where
scope deviations need a dated amendment. Concretely: I verified every library API against its installed
version before building on it, proved the metric harness with a perfect-input check before trusting a real
number, and committed the honest 0.301 (with a *negative* MOTA) rather than a tuned figure. The number is
a floor that names its own levers; that's the point.
