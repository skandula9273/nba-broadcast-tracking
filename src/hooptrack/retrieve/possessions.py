"""SportVU possession extraction -> (T x 11 x 2) player+ball trajectories. (increment-06, the centerpiece)

Data: linouk23/NBA-Player-Movements (2015-16 SportVU — the last public NBA tracking season). Each game
log (.7z) has `events`; each event's `moments` are 25Hz samples of the ball + 10 players in court feet.
A possession tensor = one event resampled to T timesteps, entities ordered (ball first, then players by
id) and normalized to the 94x50 ft court. Augmentations (court-mirror, temporal crop, jitter) are the
self-supervised positives AND the retrieval eval's 'relevant' set (the augmentation-SSL labeling scheme).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

COURT_L, COURT_W = 94.0, 50.0  # NBA court, feet
N_ENTITIES = 11  # ball + 10 players
_GH = "https://github.com/linouk23/NBA-Player-Movements/raw/master/data/2016.NBA.Raw.SportVU.Game.Logs"
_API = "https://api.github.com/repos/linouk23/NBA-Player-Movements/contents/data/2016.NBA.Raw.SportVU.Game.Logs"


def list_game_names(n: int) -> list[str]:
    with urllib.request.urlopen(_API, timeout=30) as r:
        items = json.load(r)
    names = [it["name"][:-3] for it in items if it["name"].endswith(".7z")]
    return sorted(names)[:n]


def _game_json(name: str, cache: Path) -> dict | None:
    import py7zr

    cache.mkdir(parents=True, exist_ok=True)
    z = cache / f"{name}.7z"
    if not z.exists():
        urllib.request.urlretrieve(f"{_GH}/{name}.7z", z)
    # Read the json THIS archive contains (unique gameid filename), not the newest in the cache dir.
    # py7zr preserves each file's stored 2016 mtime, so an mtime sort returns the wrong game once the
    # cache holds many archives — it silently duplicated one game across every slot (data-contamination
    # bug caught by a unique-possession consistency check, increment-06b).
    with py7zr.SevenZipFile(z, "r") as a:
        members = [n for n in a.getnames() if n.endswith(".json")]
        a.extractall(cache)
    for jn in members:
        try:
            d = json.loads((cache / jn).read_text())
        except Exception:
            continue
        if isinstance(d, dict) and "events" in d:
            return d
    return None


def _event_to_possession(event: dict, T: int, min_moments: int) -> np.ndarray | None:
    """One event -> (T, 11, 2) normalized trajectory, or None if too short / malformed."""
    frames = []
    for m in event.get("moments", []):
        ents = m[5] if len(m) > 5 else None
        if not ents or len(ents) != N_ENTITIES:
            continue
        # order: ball (teamid == -1) first, then players by (teamid, playerid) -> stable across moments
        ents = sorted(ents, key=lambda e: (e[0] != -1, e[0], e[1]))
        frames.append([[e[2], e[3]] for e in ents])  # (11, 2) = (x, y)
    if len(frames) < min_moments:
        return None
    arr = np.asarray(frames, float)  # (M, 11, 2)
    idx = np.linspace(0, len(arr) - 1, T).round().astype(int)  # uniform resample to T
    poss = arr[idx]
    poss[..., 0] = np.clip(poss[..., 0] / COURT_L, 0, 1)  # normalize to court
    poss[..., 1] = np.clip(poss[..., 1] / COURT_W, 0, 1)
    return poss


def build_corpus(
    n_games: int = 6, T: int = 48, max_possessions: int = 1500, min_moments: int = 40,
    cache_dir: str = "data/sportvu",
) -> tuple[np.ndarray, list[dict]]:
    """Download n_games SportVU logs -> a corpus of possession tensors. Returns (corpus (N,T,11,2), meta)."""
    cache = Path(cache_dir)
    corpus, meta = [], []
    for name in list_game_names(n_games):
        d = _game_json(name, cache)
        if d is None:
            continue
        seen = set()
        for ev in d["events"]:
            p = _event_to_possession(ev, T, min_moments)
            if p is None:
                continue
            key = round(float(p.sum()), 2)  # cheap near-duplicate guard (SportVU events overlap)
            if key in seen:
                continue
            seen.add(key)
            corpus.append(p)
            meta.append({"game": name, "eventId": ev.get("eventId")})
            if len(corpus) >= max_possessions:
                return np.asarray(corpus), meta
    return np.asarray(corpus), meta


# ---- augmentations (SSL positives + eval 'relevant' set) ----

def mirror(poss: np.ndarray, axis: str) -> np.ndarray:
    out = poss.copy()
    if axis == "length":       # swap baskets (same play, other end)
        out[..., 0] = 1.0 - out[..., 0]
    elif axis == "width":      # left-right symmetry
        out[..., 1] = 1.0 - out[..., 1]
    return out


def temporal_crop(poss: np.ndarray, rng: np.random.Generator, frac: float = 0.7) -> np.ndarray:
    T = len(poss)
    w = max(4, int(T * frac))
    s = rng.integers(0, T - w + 1)
    idx = np.linspace(s, s + w - 1, T).round().astype(int)  # crop then resample back to T
    return poss[idx]


def jitter(poss: np.ndarray, rng: np.random.Generator, sigma: float = 0.01) -> np.ndarray:
    return np.clip(poss + rng.normal(0, sigma, poss.shape), 0, 1)


def augment(poss: np.ndarray, rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "jitter":
        return jitter(poss, rng)
    if kind == "crop":
        return temporal_crop(poss, rng)
    if kind == "mirror":
        return mirror(poss, "length" if rng.random() < 0.5 else "width")
    raise ValueError(kind)


def augment_view(
    poss: np.ndarray, rng: np.random.Generator,
    jitter_sigma: float = 0.01, p_mirror: float = 0.5, p_crop: float = 0.5,
) -> np.ndarray:
    """A stochastic COMPOSITION of the three augmentations — one contrastive view (increment-06b).

    Composes the same structure-preserving transforms the eval uses. Mirror lands in ~p_mirror of views,
    so a positive pair (two views of one possession) frequently differs by a court-mirror — that is the
    exact training signal that forces the mirror-invariance the raw-trajectory floor lacks (r@1 ~0.001).
    """
    out = poss
    if rng.random() < p_mirror:
        out = mirror(out, "length" if rng.random() < 0.5 else "width")
    if rng.random() < p_crop:
        out = temporal_crop(out, rng)
    return jitter(out, rng, sigma=jitter_sigma)  # a little jitter always
