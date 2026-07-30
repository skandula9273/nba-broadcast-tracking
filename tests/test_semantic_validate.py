"""SupCon loss for the semantic-retrieval validation — torch, tiny tensors; skipped on the torch-less CI gate."""

import pytest

pytest.importorskip("torch")  # torch is in the [cv] extra, not the CI [dev] gate — skip if absent

import torch  # noqa: E402

from hooptrack.retrieve.semantic_validate import supcon_loss  # noqa: E402


def _norm(x):
    return x / x.norm(dim=1, keepdim=True)


def test_supcon_rewards_label_aligned_clusters():
    # rows 0,1 near [1,0]; rows 2,3 near [0,1]. Same embeddings, two label assignments:
    z = _norm(torch.tensor([[1.0, 0.02], [1.0, -0.02], [0.02, 1.0], [-0.02, 1.0]]))
    aligned = torch.tensor([0, 0, 1, 1])       # positives = the spatially-near pairs -> low loss
    crossed = torch.tensor([0, 1, 0, 1])       # positives = the far pairs -> high loss
    assert supcon_loss(z, aligned, 0.1) < supcon_loss(z, crossed, 0.1)


def test_supcon_returns_finite_scalar():
    z = _norm(torch.randn(8, 4))
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    loss = supcon_loss(z, labels, 0.1)
    assert loss.ndim == 0 and torch.isfinite(loss)
