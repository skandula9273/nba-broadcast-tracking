# hooptrack — technical writeup

Reconstruct player/ball tracking ("moving dots") from ordinary **NBA broadcast video**, train a
**play-embedding model** on the trajectories for similarity retrieval, and wrap both in a **measured,
reproducible eval platform**. The deliverable is the eval rigor and an honest research question — not a tracking
demo. This is the **narrative synthesis** — the argument and the findings, in order; the blow-by-blow lives in
`docs/depth-round-notes.md` and the `docs/increment-0N-*.md` files. Every number below is a committed timestamped
JSON in `eval_results/`, but the **`README` results table and the JSON are the numbers of record** — if a figure
here ever drifts from them, trust the artifact.

> **The research question:** how much do downstream basketball analytics (play retrieval) degrade when computed
> on CV-reconstructed tracking vs ground-truth tracking, and *which* perception errors matter most?

---

## The three findings (the intellectual core)

**1. Reconstruction cost concentrates in ONE stage — tracking association — not per-frame perception.**
In the controlled degradation study, positional jitter (homography + detection localization) and
dropout/fragmentation cost ~nothing; **ID-switches** dominate. At a realistic combined budget the trained
encoder falls to recall@1 **0.68** (floor 0.99) — and once the *measured* re-ID error is folded in (jersey
coverage on real tracker output → ~4 of 10 players unresolved per possession), to **0.27**. This was predicted
in simulation and then **confirmed empirically from three independent angles**: jersey coverage collapses
0.73 → 0.37 on real tracker output purely from fragmentation (949 track-ids for 150 players; per-crop read rate
unchanged); the end-to-end wire on real tracker output scores floor r@1 **0.80**; and the one-clip end-to-end
run returns arbitrary neighbours precisely because of arbitrary player slots. The lever to trust off broadcast
is the **association stage**, not sub-pixel registration.

**2. Order-invariance and temporal-crop robustness are entangled — you cannot have both in this
representation.** Making the encoder order-robust (so it survives association error) collapses temporal-crop
recall from **0.94 → ~0.45**. This holds whether the invariance comes from augmentation or from an *exactly*
permutation-invariant architecture — two independent routes pay the *same* crop cost, so it tracks the
invariance itself, not the method. (Whether more model capacity buys it back is GPU-gated: the d128
set-transformer needs CUDA; a ready harness is committed, and the equal-epoch signal so far shows no obvious
capacity effect.)

**3. Semantic play retrieval is achievable — but needs a semantic objective; the headline recall number is
instance-invariance, not semantics.** The augmentation-SSL encoder scores recall@1 **0.62 → 0.98**, but that
measures whether an augmented possession retrieves *its own original* — instance-level invariance, not "similar
plays." On an actual play-type axis the SSL encoder sits at **~random** (precision@5 0.51 vs 0.50). Changing
**only the objective** to supervised-contrastive (same architecture, augmentation, by-game split, seed) lifts
held-out-*game* semantic precision@5 to **0.942**. So the task and the eval are sound; the SSL *objective* was
the limitation. This directly answers the sharpest critique of the centerpiece.

---

## What runs, and the numbers

| Stage | Result (committed) | Honest status |
|---|---|---|
| **Detection** | mAP@50 **0.987** (fine-tuned yolov8m, single class); per-game **0.970–0.991** (std 0.009) | val-selected; ball not detected |
| **Tracking** | HOTA **0.301 → 0.473 → 0.525** (ByteTrack → BoT-SORT → detector fine-tune → **imgsz 640**) | real, on SportsMOT basketball-val |
| **Homography** | 503px floor → **16px** on held-out arenas (learned keypoint net + domain aug) | **does NOT transfer to broadcast** (data limit) |
| **Ball** | COCO 'sports ball', **~24%** frame coverage (no training) | sparse; enables partial possessions |
| **re-ID** | jersey coverage **0.73** (GT) / **0.37 → 0.48** (tracker + stitching) | coverage not accuracy; OSNet can't split teammates |
| **Retrieval (instance)** | recall@1 floor **0.62 → 0.98** trained; court-mirror 0.004 → 0.999 | on SportVU GT tracks |
| **Retrieval (semantic)** | supervised precision@5 **0.942** vs SSL 0.51 ≈ random | derived buckets, not annotated set-plays |
| **End-to-end wire** | real tracker → FAISS, floor r@1 **0.80**; broadcast-domain encoder **beats** floor (0.86 → 0.88) | image coords, no ball, fragmentation |
| **Degradation study** | association is the cost; combined trained **0.68** (0.27 with re-ID) vs floor 0.99 | controlled proxy + real confirmation |
| **Serving** | detect→track **9.9 → 21.5 fps** (COCO/1280 → deployed fine-tuned/640); yolov8n on the frontier | measured deployment menu |
| **Uncertainty** | top-1 similarity **calibrated** (corr 0.74, ECE 0.06); selective → 0.82 → **~1.0** | augmentation-SSL correctness label |

Reproducibility: encoders are **checkpointed** (`{state_dict, config, seed, corpus fingerprint, git sha}`), so
the retrieval and degradation artifacts provably describe **one model**, not two same-seed instances. Both the
eval harness and the FastAPI service call the one shared `pipeline.py`, so committed numbers describe the
deployed system. 106 tests, fixed seeds, one-command reruns via `make`.

---

## Honest map — real / simulated / gated

- **Real, end-to-end:** detect → track; the retrieval core + FAISS; reconstructed-vs-GT retrieval on *real*
  tracker output; the degradation study; serving detect→track; the demo (`make demo`) on GT.
- **Real but limited:** homography (arena only), re-ID (coverage not accuracy), ball (~24%), analytics
  (possessions only where the ball is visible).
- **Simulated (clearly labelled):** the degradation study's per-stage error budgets are a proxy; the real
  end-to-end number (0.80) is the honest anchor, and it was a *harder* hit than the proxy implied.
- **Gated (not faked):** the definitive d128 set-arch needs a CUDA GPU (harness ready); real broadcast
  homography needs aligned court GT; re-ID *accuracy* and semantic play-type *labels* need annotation.
  Monocular-3D is explicitly out of scope.

The most important honesty note is the **one-clip end-to-end run** (inc-10): the whole path executes on a
broadcast clip, but the retrieved neighbours are **not meaningful** — a hand-clicked homography (~30 ft off) +
arbitrary player slots + no ball = the degradation study's prediction made empirical. A clean-looking top-5
there would have been *less* honest than reporting that it doesn't work yet.

---

## V2 upside (measured, not a checklist)

- **Serving Pareto:** the deployed imgsz 1280 was strictly **dominated** — 640 gives higher mAP *and* 2.6× fps
  (a train/infer mismatch was silently costing HOTA); a fine-tuned **yolov8n** (3.0M params, 6 MB) sits on the
  frontier at 44.7 fps. Applied to the serve default.
- **Inference formats — a negative that flipped on inspection.** The naive exports first *looked* slower:
  full-predict ONNX and CoreML both trailed PyTorch/MPS, which read as an honest negative for the format knob.
  But the variable was the **execution provider**, not the format — `onnxruntime` had silently defaulted to its
  **CPU** provider. Routing the *same* ONNX graph through the **CoreML EP** (Apple Neural Engine) runs the
  forward pass in **17.7 ms — 2.47× faster than MPS PyTorch** (43.8 ms). Catching that mislabelled negative is
  worth more than the speedup itself (`make onnx-providers`).
- **End-to-end uncertainty:** the retrieval confidence is calibrated → selective prediction recovers 0.82 → ~1.0.
- **Observability:** real serving metrics + a detections/frame drift signal at `/metrics`.
- **NL query:** structured text → semantic-constraint retrieval over the *validated* axes (not a fake LLM).

---

## What I'd do next

1. **Association is the bottleneck** (finding 1) → a stronger tracker / real re-ID is the highest-leverage
   perception work, worth more than any detector or homography gain.
2. **Semantic retrieval** (finding 3) → real play-type labels + the supervised objective + a learned text
   encoder turn the NL demo into a product.
3. **Broadcast homography** → the one hard data blocker; a broadcast court-GT set unlocks the top-down path.
4. **The d128 set-arch on a GPU** (finding 2) → settle whether the crop cost is truly intrinsic.

The through-line: every heavier method had to beat a simpler baseline, every win was attributed to its actual
cause, and the negatives were reported as loudly as the wins. That discipline — not any single number — is the
deliverable.
