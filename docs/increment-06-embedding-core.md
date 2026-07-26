# Increment 06 — the embedding core (V1 centerpiece): the recall@k floor

The centerpiece: a **play/possession embedding** for similar-play retrieval — map a possession (players +
ball trajectories) to a vector so similar plays land near each other; "find similar plays" = nearest
neighbour over the vectors. This increment builds the **hand-feature retrieval floor** (simpler-first, the
contract) that the trained encoder must beat. No learning yet — this sets the bar.

## Locked decisions

- **Labeling scheme:** augmentation-SSL recall@k — query with an augmented possession (jitter /
  temporal-crop / court-mirror), the "relevant" gallery item is the original. Honest self-constructed
  proxy (measures invariance to structure-preserving transforms; a mirrored/cropped play *is* the same
  play). PBP-semantic labels are a later refinement.
- **Encoder architecture (next step):** compact trajectory transformer, self-supervised contrastive.
- **Baseline first:** a hand feature — the flattened, court-normalized possession trajectory, L2-normalized
  → cosine nearest-neighbour.

## Data

`linouk23/NBA-Player-Movements` — 2015-16 SportVU, the **last publicly-available NBA tracking season**
(25 Hz, ball + 10 players, court feet), merged with play-by-play. Each event → a possession tensor
**(T=48 × 11 entities × 2)**, entities ordered (ball, then players by id), court-normalized. Corpus:
**1,336 possessions from 6 games** (bounded; `data/` is gitignored).

## Results — the floor (hand feature, no learning)

| augmentation | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|
| jitter | 0.665 | **1.000** | 1.000 | 0.833 |
| temporal crop | 0.563 | 0.914 | 0.938 | 0.734 |
| **court mirror** | **0.001** | 0.005 | 0.007 | 0.005 |
| **overall** | **0.410** | 0.640 | 0.648 | 0.524 |
| random baseline | 0.0008 | 0.004 | 0.007 | — |

## The finding (and the target for the trained encoder)

The raw-trajectory feature is **~550× better than random overall**, and it **essentially solves jitter and
crop** (recall@5 = 1.0 and 0.91) — small noise and temporal reframing barely move a normalized trajectory.
But it is **at chance on court-mirror (recall@1 0.001)**, by construction: mirroring flips the coordinates,
so the flattened trajectory of a mirrored play is far from the original even though it's the *same play*.

That is the precise, measured job of the trained trajectory transformer: **add mirror-invariance** (and
tighten crop) via contrastive training with these augmentations, while keeping jitter robustness — i.e.,
beat **overall recall@1 0.41**, and specifically drag **mirror off the floor (0.001)**. A concrete,
falsifiable target, not a vibe.

## Honest boundaries

- **Self-constructed proxy.** The augmentation-SSL eval measures invariance to structure-preserving
  transforms, not full semantic "same set-play" similarity (the design doc's stated limitation). A
  PBP-outcome or hand-labeled set-play eval is a later, more semantic upgrade.
- **Bounded corpus** (6 games / 1,336 possessions) — enough for a clean floor; scale up for the trained
  encoder. Near-duplicate SportVU events are de-duplicated cheaply.
- **The floor is honestly easy where it should be** (jitter/crop) and honestly hard where the value is
  (mirror) — so "beat the floor" is a meaningful claim, not a rigged one.

## Next

The trained **compact trajectory transformer** (contrastive InfoNCE over the same augmentations) → must
beat this floor, reported as-is. **Done in increment-06b** (`docs/increment-06b-embedding-core-trained.md`):
overall recall@1 0.62 → **0.98**, court-mirror 0.004 → **0.999**, with a one-variable ablation proving the
mirror augmentation (not the transformer capacity) is what buys the invariance. Then the FAISS-indexed
retrieval and, downstream, the **reconstructed-vs-GT degradation study** — the finding the whole platform
exists for.
