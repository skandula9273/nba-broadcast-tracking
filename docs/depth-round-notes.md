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
to the committed numbers via a dated amendment: **detection mAP50 0.987** (fine-tuned yolov8m) and
**tracking HOTA 0.301 baseline → 0.473 best**, all on SportsMOT basketball-val.
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
- **Player-identity (re-ID) is the dominant *unmeasured* risk** — 2 wrong players → r@1 0.81, full scramble →
  0.02. Sensitivity only (stage unbuilt); it's why re-ID is next.
- **Alternatives:** claim reconstruction "barely hurts" from the jitter/dropout results alone (misleading —
  omits the order-corruption stages); or wait for aligned broadcast GT that doesn't exist. Rejected both.
- **Tradeoff / honesty:** a controlled proxy with stated px→ft / IDSW→swaps mapping assumptions, not the real
  end-to-end pipeline; the sweeps (not one point) carry the finding; the re-ID axis is labelled unmeasured.
- **The depth-round lesson:** the study connected the perception error budgets to the retrieval cost they
  impose, localized it to one stage, and *surprised* me (the fancy encoder is less robust than the baseline
  where it wasn't trained to be) — then named the two next levers. Measuring beats assuming, again.

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
