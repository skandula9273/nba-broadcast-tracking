# Increment 06b — the trained embedding core: a compact trajectory transformer beats the floor

The learned half of the centerpiece. Increment-06a built the **hand-feature recall@k floor** and localized
exactly where it fails: **court-mirror recall@1 ≈ 0.001** (at chance), because a flattened trajectory can't
see that a mirrored play is the *same* play. This increment trains a **compact trajectory transformer** with
**contrastive InfoNCE** over jitter/crop/mirror augmentations, and asks the falsifiable question the floor
posed: does it **beat the floor overall, and specifically drag court-mirror off the floor?** It does —
mirror recall@1 **0.004 → 0.999** on a held-out set the encoder never trained on.

## Locked decisions

- **Architecture — compact trajectory transformer.** Each of `T=48` timesteps is one token: the flattened
  (ball + 10 players) × (x, y) = 22-dim court-normalized positions, centered to `[-0.5, 0.5]` so a
  court-mirror is a coordinate **sign flip**. Tokens → `d_model=128`, + sinusoidal positional encoding over
  time, a 2-layer **pre-LN** `TransformerEncoder` (4 heads), temporal mean-pool, a small projection head →
  128-d **L2-normalized** embedding. **301k params** — small by design (the increment-04 MPS caution).
- **Objective — InfoNCE / NT-Xent (SimCLR).** A positive pair is two independently-augmented views of one
  possession; every other view in the batch is a negative (batch 512 → 1022 negatives), temperature 0.1.
- **The mechanism for mirror-invariance:** each view composes the same structure-preserving augmentations
  the eval uses, applying a court-mirror with `p=0.5`. So positive pairs frequently differ by a mirror →
  the contrastive loss is *forced* to map a play and its reflection to the same point. This is the exact,
  deliberate training signal aimed at the floor's one weakness.

## Data + the honest eval

- **Scaled the corpus beyond 6 games** → `linouk23` SportVU 2015-16, **12 games / 2,612 possessions**
  (2× the floor's 1,336), 99.6% unique.
- **Split by game (no leakage):** train = 8 games (1,754 possessions), **val = 4 held-out games (858)**.
  A game-level split matters because SportVU events overlap — a random split would leak near-duplicate
  possessions across train/val. The encoder is trained only on train games and scored only on val games.
- **One-variable comparison.** The hand-feature floor is recomputed on the **identical val gallery** with the
  **identical augmented queries** (same seed) as the trained encoder. So "trained beats floor" isolates the
  feature method — nothing else changes. (This is why floor_on_val here — r@1 0.618 — is higher than the
  committed 06a floor of 0.41: a smaller 858-gallery is easier, and the corpus is clean. The fair baseline
  is always the recomputed floor_on_val, reported alongside.)

## A data-contamination bug found + fixed (the internal-consistency catch)

Scaling to 12 games surfaced a latent bug in the corpus builder. `_game_json` selected the extracted JSON by
**newest mtime**, but `py7zr` preserves each file's stored **2016** mtime — so once the cache held many game
archives, every call returned the *same* global-newest JSON. The first scaled build produced **2,994
"possessions" with only 499 unique** — effectively two games duplicated six times each, which would have put
identical plays in both train and val. Caught by a **unique-possession ratio check**, not a crash (the same
"plausible wrong number" class as the increment-01 AppleDouble bug). Fixed at the root: read the JSON member
that *this* archive actually contains via `getnames()`. Rebuilt clean (2,612 / 99.6% unique), and the check
is now a guard in the training run (`unique_ratio < 0.9` raises). *The committed 06a floor was computed on a
corpus with this latent duplication — noted plainly; 06b's floor_on_val is the clean baseline.*

## Results — trained encoder vs the floor (identical held-out val set)

| augmentation | floor r@1 | **trained r@1** | floor r@5 | trained r@5 | floor MRR | trained MRR |
|---|---:|---:|---:|---:|---:|---:|
| jitter | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal crop | 0.851 | 0.944 | 0.929 | 0.997 | 0.888 | 0.968 |
| **court mirror** | **0.004** | **0.999** | 0.020 | 0.997 | 0.010 | 0.998 |
| **overall** | **0.618** | **0.981** | 0.645 | 0.998 | 0.633 | 0.989 |

Overall recall@1 **0.618 → 0.981 (+0.363)**; the win is almost entirely the mirror axis: **0.004 → 0.999**.
Jitter stays saturated (1.000) and crop improves (0.851 → 0.944) — the encoder added mirror-invariance
*without* trading away what the floor already did well. No embedding collapse (a collapsed space would score
random ≈ 1/858 ≈ 0.001). The learning curve is the story: mirror climbs 0.005 → 0.11 (ep 25) → **0.99
(ep 50)** as the InfoNCE loss drops 6.69 → 0.63.

## Attribution — is it the mirror augmentation or the transformer? (the one-variable ablation)

Following the increment-03 discipline (isolate the driver, don't just bank the win): retrain **identically
except `p_mirror=0`**.

| augmentation | floor r@1 | full model r@1 | **no-mirror r@1** |
|---|---:|---:|---:|
| jitter | 1.000 | 1.000 | 1.000 |
| temporal crop | 0.851 | 0.944 | **0.981** |
| **court mirror** | **0.004** | **0.999** | **0.019** |
| **overall** | **0.618** | **0.981** | 0.667 |

**Without mirror-augmented positives, mirror stays at the floor (0.019).** The transformer's *capacity* buys
no mirror-invariance on its own — the entire +0.98 mirror gain is attributable to the augmentation. Two
honest nuances fall out: (1) the no-mirror model gets *better* crop (0.981 vs 0.944) — enforcing
mirror-invariance costs a little temporal discrimination, a real tradeoff; (2) no-mirror converges to a
**lower** training loss (0.168 vs 0.632) yet is far worse on the metric that matters — a clean reminder that
the contrastive loss is a proxy, not the retrieval objective.

## The retrieval index (FAISS) — and a wiring check that can't lie

The embeddings are indexed for real nearest-neighbour search with **`faiss.IndexIDMap(IndexFlatIP)`** — an
**exact, flat, inner-product** index (`retrieve/index.py`). Because the encoder emits L2-normalized vectors,
inner product *is* cosine, so the flat index reproduces the brute-force `q @ gallery.T` retrieval **bit-for-bit**.
Exact-first (rule #7): an approximate index (IVF/HNSW, recall-vs-latency) is a later *measured* ablation, not a
prerequisite — and there's no point trading recall for speed before there's a reason to.

That equivalence is the honesty hook, not a footnote: the training run scores the val set **twice** — once
brute-force, once through the FAISS index over the *same* embeddings — and **asserts they match** (recorded as
`results.index.reproduces_bruteforce: true`, index overall r@1 0.981 == brute-force 0.981). Same discipline as
the GT-as-tracker → HOTA 1.000 check: if swapping the index in changed the committed recall, the run would fail.

**A real bug found here (the "tell me about a bug" story):** on macOS, torch and faiss-cpu each link their own
`libomp`. `KMP_DUPLICATE_LIB_OK=TRUE` lets them *import* together, but under actual OpenMP compute (torch MPS +
faiss) the process **segfaults (exit 139)** — verified by reproduction. The robust fix is pinning faiss to a
single OpenMP thread (`faiss.omp_set_num_threads(1)`); import-reordering and `OMP_NUM_THREADS=1` also work but
are fragile/global. Negligible cost at this index size, and correct for the eventual **serving** path too, where
one process encodes a query with torch and searches with faiss. And the brute-force assertion doubles as the
guard: any OpenMP-induced corruption would *raise*, not slip through silently.

## Honest boundaries

- **Still the self-constructed augmentation-SSL proxy** (the design-doc limitation): this measures invariance
  to structure-preserving transforms, not full semantic "same set-play" similarity. mirror r@1 0.999 is a
  near-solved *invariance*, not evidence that the embedding captures play semantics. A PBP-outcome /
  hand-labeled set-play eval is the next, more meaningful bar.
- **MPS non-determinism:** seeded (numpy + torch + mps), but MPS kernels aren't bit-exact run-to-run. The
  effect size (0.004 → 0.999) dwarfs any such noise, so the finding is robust; exact digits may drift.
- **Mildly optimistic knobs:** no held-out *test* games (only 12 games total), and hyperparameters weren't
  tuned on a separate split — the val set is both the model-selection and the reporting set. Stated plainly.
- **Bounded corpus** (12 games) — enough to learn the invariance cleanly; not a generalization study.

## Next

The FAISS index is in (above), so the retrieval identity is complete and verified. Next is the
**reconstructed-vs-GT degradation study** — run this same retrieval on tracks reconstructed by the perception
pipeline vs SportVU ground truth, and measure how much recall the reconstruction costs. That study is the
finding the whole platform exists for. (The open design question there: there's no dataset with *aligned*
broadcast + SportVU truth, so the honest version is a **controlled degradation** — perturb GT tracks with the
error budgets already measured per stage and measure the recall falloff.)
