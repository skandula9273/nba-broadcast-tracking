# Increment 03 — attributing the BoT-SORT win (thresholds vs appearance vs CMC)

Increment-02 showed BoT-SORT beats ByteTrack by **+0.074 HOTA** (0.301 → 0.375) — but as a *method
bundle* (higher thresholds + confirmation cascade + appearance ReID + camera-motion compensation), so it
didn't say *why*. This increment decomposes it with four one-variable ablations, each on **byte-identical
detections** (guaranteed by the detection cache), so the accuracy comparison is clean.

Configs: `v0_bytetrack_hi` (ByteTrack, `track_thresh` 0.45→0.60), `v0_botsort_noreid` (`use_reid: false`),
`v0_botsort_nocmc` (`use_cmc: false`). Sources: the five committed `eval_results/eval_*.json`.

## Results (SportsMOT basketball-val, 15 seqs, identical detections)

| variant | HOTA | DetA | AssA | LocA | MOTA | IDF1 | IDSW | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ByteTrack (v0 baseline) | 0.301 | 0.325 | 0.279 | 0.837 | −0.395 | 0.281 | 901 | 2017 |
| ByteTrack @thr 0.60 | 0.336 | 0.395 | 0.287 | 0.839 | +0.036 | 0.333 | 736 | 1909 |
| BoT-SORT − ReID | 0.370 | 0.441 | 0.311 | 0.890 | +0.127 | 0.347 | 670 | 2562 |
| BoT-SORT − CMC | 0.375 | 0.448 | 0.314 | 0.888 | +0.156 | 0.351 | 769 | 2694 |
| **BoT-SORT (full)** | **0.375** | 0.439 | 0.320 | 0.890 | +0.113 | 0.351 | 728 | 2648 |

## Attribution of the +0.074 HOTA

| component | HOTA | how isolated |
|---|---:|---|
| Detection **threshold** (0.45→0.60, motion-only ByteTrack) | **+0.035 (~47%)** | baseline → ByteTrack@0.60 |
| BoT-SORT's fuller **confirmation cascade** (its tuned thresholds + `min_hits` + buffer) | ~+0.034 (~46%) | residual (ByteTrack@0.60 → BoT-SORT) |
| **Appearance / OSNet ReID** | **+0.005 (~7%)** | full − (−ReID) |
| **Camera-motion compensation (CMC)** | **~0.000 (~0%)** | full − (−CMC) |

## The finding

**BoT-SORT's two marquee features — appearance ReID and CMC — contribute essentially nothing on
basketball broadcast. The entire win is stricter detection confirmation.** The confirmation policy (higher
start thresholds, `min_hits`, the association cascade) acts as a *precision filter*: it stops promoting
low-confidence referee/bench/crowd boxes to output tracks, so **DetA jumps (0.325→0.44) and MOTA flips
negative→positive**. And **~half of that is reachable by simply raising ByteTrack's threshold** — a
motion-only tracker, no ReID, no CMC.

It's sharper than "no benefit": **removing** ReID or CMC actually *improved* MOTA (−CMC 0.156, −ReID 0.127
vs full 0.113), and −ReID had the fewest ID-switches (670 vs 728). So the expensive features are
net-neutral-to-mildly-harmful here — the near-identical-uniforms hypothesis (appearance can't tell
teammates apart) and minor CMC matching noise, now quantified rather than assumed.

**Cost (clean cached-vs-cached comparison):** with detections cached, association-only throughput was
~95 fps without ReID vs ~22 fps with ReID — **ReID is ~4× the cost for +0.005 HOTA**, a strictly bad
trade on this data. (Whole-pipeline fps isn't comparable across rows here because the baseline and full
BoT-SORT runs pre-date the cache and include detection; that confound is why fps is discussed separately,
not tabled.)

## What this changes

- **Tracking is "done enough" for V0.** The best HOTA is 0.375, and we now understand its structure: the
  gain is a precision effect, appearance/CMC are dead weight on this domain, and a cheaper motion-only
  tracker with a better threshold recovers most of it. No further tracker gold-plating (the enabler is not
  the headline).
- **The ceiling is upstream.** Even the best DetA is **0.44** — the COCO-pretrained "person" detector,
  never told what an athlete is, is the binding constraint. So the next increment is **detector
  fine-tuning on SportsMOT**, the single biggest remaining HOTA lever, measured against the 0.375 floor.

## Method note — the detection cache (deferred, then built when it paid off)

In increment-02 I explicitly *declined* to build a detection cache (YAGNI — a single ablation's
deterministic re-run is cheap enough). This increment's four tracker-only runs crossed that threshold, so
I built `CachingDetector` (config-keyed, transparent — it implements the `Detector` protocol, so the
shared pipeline is unchanged). It makes tracker ablations cheap **and** guarantees identical detections
across variants. Validated by a reproduce-check: cached ByteTrack reproduced the committed baseline
tracker output **byte-for-byte**.
