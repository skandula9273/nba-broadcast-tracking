"""Encoder checkpointing — save -> load -> encode is bit-identical, and load uses the checkpoint's own config."""

import numpy as np
import pytest

pytest.importorskip("torch")  # torch is in the [cv] extra, not the CI [dev] gate — skip if absent

import torch  # noqa: E402

from hoopvec.config import EmbeddingConfig  # noqa: E402
from hoopvec.retrieve.checkpoint import corpus_fingerprint, load_checkpoint, save_checkpoint  # noqa: E402
from hoopvec.retrieve.embed import PlayEmbedder  # noqa: E402


def _small_embedder():
    cfg = EmbeddingConfig(enabled=True, arch="trajectory_transformer", dim=32, d_model=32, n_heads=2,
                          n_layers=1, ff_dim=64, dropout=0.1, temperature=0.1)
    torch.manual_seed(0)
    emb = PlayEmbedder(cfg, device="cpu", T=16)
    emb.build()
    return emb


def test_save_load_encode_is_bit_identical(tmp_path):
    emb = _small_embedder()
    x = np.random.default_rng(0).random((5, 16, 11, 2)).astype(np.float32)
    before = emb.encode_batch(x)
    path = save_checkpoint(emb, tmp_path / "ck.pt", seed=0, corpus_fp="abc123",
                           final_loss=0.5, val_metrics={"recall@1": 0.9})
    emb2, ckpt = load_checkpoint(path, device="cpu")
    after = emb2.encode_batch(x)
    assert np.array_equal(before, after)                 # bit-identical, not just close
    assert ckpt["seed"] == 0 and ckpt["corpus_fingerprint"] == "abc123"
    assert ckpt["val_metrics"] == {"recall@1": 0.9}


def test_load_reconstructs_from_checkpoint_config_not_current(tmp_path):
    # a checkpoint stays loadable after config drift: reload rebuilds at the SAVED dims regardless of any
    # 'current' config. Save a d_model=32 model; the reloaded encoder is d_model=32.
    emb = _small_embedder()
    path = save_checkpoint(emb, tmp_path / "c.pt", seed=1, corpus_fp="x", final_loss=None, val_metrics=None)
    emb2, ckpt = load_checkpoint(path, device="cpu")
    assert emb2.cfg.d_model == 32 and emb2.cfg.dim == 32 and emb2.T == 16
    assert ckpt["config"]["d_model"] == 32               # the config that built it travels with the weights


def test_corpus_fingerprint_stable_and_content_sensitive():
    a = np.zeros((3, 4, 11, 2), np.float32)
    b = a.copy()
    b[0, 0, 0, 0] = 1.0
    assert corpus_fingerprint(a) == corpus_fingerprint(a.copy())   # deterministic
    assert corpus_fingerprint(a) != corpus_fingerprint(b)          # sensitive to content
