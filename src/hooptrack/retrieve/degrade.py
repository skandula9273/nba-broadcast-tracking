"""Reconstruction-error model for the degradation study. (increment-07)

The platform's headline question: *how much retrieval accuracy does broadcast reconstruction cost vs
ground-truth tracks?* There is no dataset with aligned broadcast + SportVU truth, so we run a **controlled
degradation**: take clean SportVU GT possessions and inject the error budgets THIS project already measured
at each perception stage, then measure the recall@k falloff (study.py). Each knob maps to a measured number
(rule #5 — anchor to a real metric, not an invented one), and each is swept one-variable-at-a-time.

Perturbations (all operate on a possession tensor P: (T, 11, 2), court-normalized [0,1], entity 0 = ball):

  jitter_ft      positional Gaussian noise    <- homography registration (inc-05: ~2px @ 3px keypoints)
                                                 + detection localization (LocA 0.841)
  dropout_interp drop a per-entity span, then <- detection recall / fragmentation (DetA 0.707, Frag 1847)
                 linear-interpolate the gap      (what analytics does with an occlusion gap)
  id_swap        swap two players' tracks     <- tracking association (AssA 0.317, IDSW 955) — the
                 from a random frame onward       retrieval-critical error: it scrambles per-entity slots
  permute_players shuffle k players' slots     <- re-ID / player identity (NOT yet built/measured): a real
                                                 tracker emits arbitrary track order; canonical order needs
                                                 re-ID. SENSITIVITY ONLY — no measured operating point.
"""
from __future__ import annotations

import numpy as np

COURT_L, COURT_W = 94.0, 50.0  # NBA court, feet — for the ft <-> court-normalized conversion


def jitter_ft(P: np.ndarray, rng: np.random.Generator, sigma_ft: float) -> np.ndarray:
    """Add isotropic Gaussian position noise of std `sigma_ft` FEET (converted to normalized units)."""
    scale = np.array([sigma_ft / COURT_L, sigma_ft / COURT_W])
    return np.clip(P + rng.normal(0.0, 1.0, P.shape) * scale, 0.0, 1.0)


def dropout_interp(P: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Occlusion / missed detection: blank a contiguous span of `rate`*T frames per entity, fill by linear
    interpolation between the surrounding known positions (a tracker/analytics would bridge the gap)."""
    T, E, _ = P.shape
    g = int(round(rate * T))
    if g <= 0:
        return P.copy()
    out = P.copy()
    for e in range(E):
        s = int(rng.integers(0, T - g + 1)) if T > g else 0
        lo, hi = s - 1, s + g                      # anchors just outside the gap
        idx = np.arange(s, s + g)
        for c in range(2):
            if lo < 0 and hi >= T:                 # whole track missing -> hold its mean
                out[s:s + g, e, c] = P[:, e, c].mean()
            elif lo < 0:
                out[s:s + g, e, c] = P[hi, e, c]   # gap at start -> hold first known
            elif hi >= T:
                out[s:s + g, e, c] = P[lo, e, c]   # gap at end -> hold last known
            else:
                out[s:s + g, e, c] = np.interp(idx, [lo, hi], [P[lo, e, c], P[hi, e, c]])
    return out


def id_swap(P: np.ndarray, rng: np.random.Generator, n_swaps: int) -> np.ndarray:
    """Tracking ID switch: swap two PLAYER entities' trajectories from a random frame onward (the ball,
    entity 0, is excluded). The play as a whole is unchanged, but the per-entity slots get scrambled."""
    out = P.copy()
    T, E, _ = P.shape
    players = np.arange(1, E)
    for _ in range(int(n_swaps)):
        i, j = rng.choice(players, size=2, replace=False)
        t = int(rng.integers(1, T))
        out[t:, [i, j]] = out[t:, [j, i]]
    return out


def permute_players(P: np.ndarray, rng: np.random.Generator, n_wrong: int) -> np.ndarray:
    """Re-ID error (sensitivity only): shuffle the canonical slots of `n_wrong` players for the WHOLE
    possession — a real tracker's arbitrary track order that re-ID would have to recover."""
    out = P.copy()
    n = int(n_wrong)
    if n < 2:
        return out
    players = np.arange(1, P.shape[1])
    chosen = rng.choice(players, size=min(n, len(players)), replace=False)
    out[:, chosen] = P[:, rng.permutation(chosen)]
    return out


def reconstruct(P: np.ndarray, rng: np.random.Generator,
                sigma_ft: float, drop_rate: float, n_swaps: int) -> np.ndarray:
    """The combined perception-stage error: id-swap, then dropout+interp, then positional jitter."""
    out = id_swap(P, rng, n_swaps)
    out = dropout_interp(out, rng, drop_rate)
    return jitter_ft(out, rng, sigma_ft)
