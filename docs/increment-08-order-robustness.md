# Increment 08 — order-robustness augmentation: closing the inc-07 gap, and the tradeoff it exposes

Increment-07 found that broadcast reconstruction costs retrieval almost entirely through **tracking
association** (ID-switches / arbitrary track order), and that the trained encoder was *more fragile* to that
than the naive floor — because it's order-sensitive and was trained only with order-*preserving* augmentations.
It named a falsifiable next lever: **add order-perturbation augmentation** (just as mirror-augmentation bought
mirror-invariance in inc-06b) and re-measure. This increment does exactly that — one variable, default-off so
the inc-06b/07 baselines are preserved — and measures both axes (clean recall@k *and* the degradation study).

## What was done

`possessions.order_perturb` composes the degradation study's **exact** order-corruptions (`permute_players`,
`id_swap` from `degrade.py`) as training augmentations — *train on the same corruption you measure*. It's
wired into the shared `train_encoder` view generator behind `p_permute` / `p_swap` levers (0.0 = off = the
inc-06b model). Two strengths were run end-to-end (train → recall@k; study → degradation), against the
order-blind baseline: **mild** (p=0.25) and **aggressive** (p=0.5, full 10-player shuffle).

## Results — recall@1

**Clean augmentation-SSL eval** (held-out val; the inc-06b metric):

| order-aug | jitter | crop | mirror | overall |
|---|---:|---:|---:|---:|
| **off** (inc-06b) | 1.000 | **0.944** | **0.999** | **0.981** |
| mild p=0.25 | 1.000 | 0.446 | 0.976 | 0.807 |
| aggressive p=0.5 | 1.000 | 0.386 | 0.944 | 0.777 |

**Degradation study** (query = degraded GT, gallery = clean GT; the inc-07 metric):

| order-aug | id_swap=2 | permute=4 | **combined realistic** | dropout@0.4 |
|---|---:|---:|---:|---:|
| **off** (inc-07) | 0.681 | 0.406 | **0.677** | 0.918 |
| mild p=0.25 | 1.000 | 1.000 | **1.000** | 0.397 |
| aggressive p=0.5 | 1.000 | 1.000 | **0.999** | 0.377 |
| (hand-feature floor) | 0.988 | 0.950 | 0.992 | 1.000 |

## The findings

1. **The lever works — the inc-07 hypothesis is confirmed exactly.** Order-aug drives id-swap and full-permute
   robustness to a **perfect 1.000 at every severity** (baseline decayed to 0.44 / 0.02), and the combined
   realistic operating point from **0.68 → 1.0**, now *beating* the floor (0.99). The inc-06b trick — train on
   a transform to gain invariance to it — generalizes from court-mirror to player-order.

2. **It's a regime switch, not a smooth dial.** Even **mild** (p=0.25) order-aug flips the encoder fully into
   permutation-invariance: mild and aggressive are near-identical on *both* axes. There is no cheap partial
   order-robustness here — a little augmentation crosses the threshold.

3. **The cost is specifically TEMPORAL robustness, and it's steep.** Crop recall collapses **0.944 → ~0.45**
   and high-dropout robustness falls **0.92 → ~0.39**, while the *spatial* invariances survive (jitter 1.000;
   mirror only 0.999 → ~0.95). Interpretation: the order-blind encoder used **player-slot identity as an
   anchor** to re-match a temporally-degraded play; permutation-invariance removes that anchor, so temporal
   information-loss (crop, dropout) — which the anchor used to paper over — now bites. **Order-invariance and
   temporal-crop robustness are entangled at this model's fixed capacity.**

4. **The right regime is deployment-driven.** Off broadcast, association is the *weakest* stage (AssA 0.317,
   inc-04), so ID instability is the real threat — and the order-invariant regime turns the inc-07 realistic
   degradation (0.68) into ~1.0, beating the floor. If a deployment's tracker has stable IDs (e.g. a good
   re-ID), the order-*sensitive* regime keeps the far better crop/clean recall. It's a regime **choice**, made
   against the tracker you actually have.

## Honest boundaries

- **Two points, a sharp threshold — not a fully-resolved frontier.** Mild and aggressive both sit in the
  invariant regime; the switch happens somewhere below p=0.25 (unprobed). The claim is "a little flips it,"
  not a precise threshold.
- **Train-on-what-you-measure** is legitimate here (same paradigm as mirror in inc-06b: random draws,
  held-out val possessions) but means the study measures robustness to the *modelled* corruption, not a real
  tracker's error distribution.
- **Same proxy caveats** as inc-06b/07 (augmentation-SSL invariance, not set-play semantics; val is selection
  + reporting; MPS not bit-exact — effect sizes dwarf it).

## Next

The augmentation fix buys order-robustness by *spending capacity*, which is why crop pays. Two better routes
the study points at: (1) **a permutation-invariant architecture** — per-player tokens + a symmetric pool over
players (DeepSets / set-transformer) — to build the invariance into the model instead of forcing a fixed,
order-sensitive model to approximate it, which could get order-robustness *without* the crop cost; (2) a
**team-structured** permutation (within-team only, using cheap jersey-colour that re-ID gives for free) as a
more targeted, less costly invariance. Either way, **build + measure re-ID** to replace inc-07's re-ID
sensitivity axis with a real operating point.
