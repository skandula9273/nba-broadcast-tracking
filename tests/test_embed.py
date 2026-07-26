"""Trajectory-transformer + InfoNCE tests (increment-06b). torch-gated so the suite stays CI-safe."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hooptrack.retrieve.embed import TrajectoryTransformer, info_nce_loss  # noqa: E402


def _model(dim=32, T=16):
    return TrajectoryTransformer(dim=dim, d_model=32, n_heads=4, n_layers=2, ff_dim=64, dropout=0.0, T=T)


def test_forward_shape_and_l2_normalized():
    m = _model(dim=32, T=16).eval()
    x = torch.rand(5, 16, 11, 2)
    z = m(x)
    assert z.shape == (5, 32)
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-5)  # cosine-ready


def test_info_nce_rewards_alignment():
    """The metric-harness analog of 'GT-as-tracker -> 1.0': perfectly aligned positives score a LOW loss,
    and lower than random (unaligned) views. If alignment isn't rewarded, the objective is miswired."""
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(16, 64), dim=1)
    aligned = info_nce_loss(z, z.clone(), temperature=0.1)          # view2 == view1: positives perfect
    z2 = torch.nn.functional.normalize(torch.randn(16, 64), dim=1)
    unaligned = info_nce_loss(z, z2, temperature=0.1)
    assert torch.isfinite(aligned) and aligned < 1.0
    assert aligned < unaligned


def test_build_is_deterministic_under_seed():
    torch.manual_seed(7)
    a = _model()
    torch.manual_seed(7)
    b = _model()
    x = torch.rand(3, 16, 11, 2)
    with torch.no_grad():
        assert torch.allclose(a.eval()(x), b.eval()(x), atol=1e-6)  # same seed -> same weights -> same out


def test_compact_param_budget():
    # "compact by design" (MPS caution) — the default 128-d model stays small.
    m = TrajectoryTransformer(dim=128, d_model=128, n_heads=4, n_layers=2, ff_dim=256, T=48)
    assert sum(p.numel() for p in m.parameters()) < 500_000


def test_encode_batch_roundtrips_numpy():
    from hooptrack.config import EmbeddingConfig
    from hooptrack.retrieve.embed import PlayEmbedder

    cfg = EmbeddingConfig(dim=16, d_model=32, n_heads=4, n_layers=1, ff_dim=32)
    emb = PlayEmbedder(cfg, device="cpu", T=12)
    emb.build()
    arr = np.random.rand(7, 12, 11, 2).astype(np.float32)
    out = emb.encode_batch(arr, batch=4)
    assert out.shape == (7, 16)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
