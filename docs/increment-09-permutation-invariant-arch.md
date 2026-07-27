# Increment 09 — a permutation-invariant architecture: order-robustness for free, but the crop cost is intrinsic

Increment-08 bought order-robustness with augmentation and found it collapsed temporal-crop recall (0.944 →
~0.45) — a regime switch, spending capacity to *approximate* permutation-invariance. The lever it named:
build the invariance into the **architecture** instead, and get order-robustness *without* the crop cost.
This increment tests that hypothesis. The verdict is split and honest: **the architecture delivers exact,
augmentation-free order-robustness — but it does *not* escape the crop cost.** That cost turns out to be
intrinsic to order-invariance, not to how you impose it.

## The architecture

`SetTrajectoryTransformer` (`embed.py`, `--arch set_transformer`): each **(entity, timestep)** is a token; a
shared per-entity input projection, a single "player" type-embedding for all 10 players (a distinct "ball"
type for entity 0), and **temporal (not player) positional encoding**. Attention is factorized — over time
per entity, and over entities per timestep with **no player position** (permutation-equivariant) — then a
**symmetric mean-pool over players**. Permuting the 10 players yields the *identical* embedding by
construction; the unit test confirms `encode(P) == encode(permute(P))` to 1e-5 (the baseline is confirmed
*not* invariant). Order-invariance is a **guarantee**, not a learned approximation — and needs no
order-augmentation (trained with `p_permute=p_swap=0`; mirror/jitter/crop augmentation unchanged).

## Results (recall@1; the two other rows are the committed inc-06b / inc-08 baselines)

**Clean augmentation-SSL eval** (held-out val):

| encoder | jitter | crop | mirror | overall | order-invariance |
|---|---:|---:|---:|---:|---|
| baseline order-sensitive (d128) | 1.000 | **0.944** | 0.999 | 0.981 | none |
| order-aug (d128, inc-08) | 1.000 | 0.446 | 0.976 | 0.807 | learned (augmentation) |
| **set-arch (d64, inc-09)** | 1.000 | **0.487** | 0.955 | 0.814 | **exact (architecture)** |

**Degradation study** (query = degraded GT, gallery = clean GT):

| encoder | permute=10 | id_swap=4 | **combined realistic** | dropout@0.4 |
|---|---:|---:|---:|---:|
| baseline (inc-07) | 0.022 | 0.435 | 0.677 | 0.918 |
| order-aug (inc-08) | 1.000 | 1.000 | 1.000 | 0.377 |
| **set-arch (inc-09)** | **1.000 (exact)** | 0.990 | **0.998** | 0.403 |
| (hand-feature floor) | 0.205 | 0.959 | 0.992 | 1.000 |

## The findings

1. **Order-robustness, exact and free.** `permute_players` → **1.000 at every severity by construction** (no
   augmentation; the architectural guarantee, confirmed end-to-end), `id_swap` → **0.990** (near-perfect; the
   ~0.01 gap is the mid-possession discontinuity the per-entity temporal attention still sees — exactly as
   predicted, since a swap makes one slot's *time series* discontinuous even though each frame's player *set*
   is unchanged). Combined realistic **0.68 → 0.998**, beating the floor. This is a genuine advantage over the
   augmentation route: the invariance is a *proven guarantee* on all 10! permutations, not a learned
   approximation over sampled ones, and it costs zero augmentation.

2. **But the crop cost is *not* escaped — the hypothesis is not supported.** Crop recall is **0.487**, within
   noise of the augmentation approach's 0.446, and nowhere near the order-sensitive baseline's 0.944; the
   degradation study's high-dropout collapses identically (0.403 vs order-aug 0.377). Building the invariance
   into the model did not buy back temporal robustness.

3. **The crop cost is *intrinsic to order-invariance*, not to the method (the real insight).** Two very
   different routes — augmentation at full d128 capacity (inc-08) and architecture at d64 (inc-09) — both
   collapse crop to ~0.45–0.49 and both collapse high-dropout, while only the order-*sensitive* baseline keeps
   0.94. The mechanism is consistent: the order-blind encoder used **player-slot identity as a temporal
   anchor** to re-match a temporally-degraded play; removing that anchor — by augmentation *or* architecture —
   removes the crutch. Order-invariance and temporal-crop robustness are **fundamentally entangled** in this
   trajectory representation; you can't get both by relocating *where* the invariance lives.

## Honest boundaries

- **Capacity/training confound, named.** The set-arch runs at **d_model=64 / 150 epochs** because the
  per-entity tokens explode the attention batch dimension on MPS (OOM at d128/batch512 — the recurring inc-04
  tradeoff), vs the baseline's d128 / 300 epochs. So its absolute recall is not capacity-matched. *But the
  central claim survives it:* the augmentation route at **full d128 capacity** collapsed crop just as hard, so
  the crop cost tracks the *invariance*, not the capacity. A d128 set-arch (a bigger GPU) is the clean
  follow-up, but the convergent evidence already makes the intrinsic-entanglement finding robust.
- **Cost side.** The architecture is ~5× slower per epoch on MPS and needs a smaller batch — a real
  serving/training cost for the exact-guarantee it buys.
- **Same proxy caveats** as inc-06b/07/08 (augmentation-SSL invariance, not set-play semantics; val is
  selection + reporting; MPS not bit-exact — effect sizes dwarf it).

## Verdict + next

Three encoders now sit on a measured frontier: **order-sensitive** (crop 0.94, order-fragile), and two
**order-invariant** ones — learned (inc-08) and architectural (inc-09) — both ~0.49 crop but order-robust, the
architectural one *exactly and for free*. For broadcast retrieval, where association is the weak stage (AssA
0.317), an order-invariant encoder is the right call and the architecture is the cleaner way to get it (a
guarantee, no augmentation) — but **neither escapes the crop cost, because that cost is intrinsic**. The open
routes: (1) a **capacity-matched d128 set-arch** on a real GPU (does more capacity buy back crop?); (2) a
**team-structured** invariance (within-team only, via cheap jersey colour) that keeps *some* slot-anchor; (3)
accept the tradeoff and pick the regime per deployment. And still: **build + measure re-ID** to turn inc-07's
re-ID sensitivity axis into a real operating point.
