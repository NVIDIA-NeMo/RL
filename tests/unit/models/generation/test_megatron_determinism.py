# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify the upstream request-seed contract used by strict GRPO."""

import pytest
import torch
from megatron.core.inference.sampling.torch_sampling import TorchSampling
from megatron.core.inference.sampling_params import SamplingParams

from nemo_rl.utils.determinism import keyed_seed

pytestmark = pytest.mark.mcore


def test_native_seed_serialization():
    seed = keyed_seed(42, "train", 2, 3, "turn", 0)
    params = SamplingParams(seed=seed)
    assert params.serialize()["seed"] == seed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA seeded sampling")
@pytest.mark.parametrize("top_k", [0, 8])
def test_native_draws_ignore_batch_order_and_shared_rng(top_k):
    generator = torch.Generator(device="cuda").manual_seed(123)
    logits = torch.randn(8, 4096, device="cuda", generator=generator)
    seeds = [keyed_seed(42, "train", 0, i, "turn", 0) for i in range(8)]
    positions = list(range(32, 40))
    state = generator.get_state().clone()

    def draw(rows, row_seeds, row_positions):
        return TorchSampling.sample_from_logits(
            rows,
            0.8,
            top_k,
            1.0,
            generator=generator,
            row_seeds=row_seeds,
            row_positions=row_positions,
        )

    expected = draw(logits, seeds, positions)
    assert torch.equal(generator.get_state(), state)
    generator.manual_seed(987)
    order = [7, 0, 5, 2, 6, 1, 3, 4]
    actual = draw(
        logits[order], [seeds[i] for i in order], [positions[i] for i in order]
    )
    assert torch.equal(actual, expected[order])
    for i in range(8):
        assert torch.equal(
            draw(logits[i : i + 1], seeds[i : i + 1], positions[i : i + 1]),
            expected[i : i + 1],
        )
