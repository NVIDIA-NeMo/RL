"""Tests for Multi-LoRA DP rank-striped row placement."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.models.multi_lora.sharding import rank_striped_indices


def test_rank_stripes_equal_adapter_blocks():
    ids = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [3] * 4)
    idx = rank_striped_indices(ids, dp_size=2)
    assert idx == [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]
    striped = ids[idx]
    assert torch.equal(striped[:8], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    assert torch.equal(striped[8:], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))


def test_rank_stripes_production_shape():
    ids = torch.tensor([0] * 16 + [1] * 16 + [2] * 16 + [3] * 16)
    striped = ids[rank_striped_indices(ids, dp_size=8)]
    for rank in range(8):
        assert torch.equal(
            striped[rank * 8 : (rank + 1) * 8],
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        )


def test_single_adapter_is_identity():
    ids = torch.zeros(16, dtype=torch.long)
    assert rank_striped_indices(ids, dp_size=8) == list(range(16))


def test_rejects_unequal_counts():
    ids = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError, match="equal rows"):
        rank_striped_indices(ids, dp_size=1)


def test_rejects_noncontiguous_blocks():
    ids = torch.tensor([0, 1, 0, 1])
    with pytest.raises(ValueError, match="block-contiguous"):
        rank_striped_indices(ids, dp_size=1)


def test_rejects_indivisible_adapter_block():
    ids = torch.tensor([0] * 3 + [1] * 3)
    with pytest.raises(ValueError, match="divisible"):
        rank_striped_indices(ids, dp_size=2)


def test_rejects_wrong_dtype():
    with pytest.raises(TypeError, match="torch.long"):
        rank_striped_indices(torch.tensor([0.0, 1.0]), dp_size=1)
