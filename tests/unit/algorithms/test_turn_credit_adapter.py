# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay-branch integration: metadata and data-plane reads match the shaping contract."""

import json
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils import turn_credit_adapter as adapter
from nemo_rl.algorithms.single_controller_utils.turn_credit import (
    RolloutTurnLabels,
    TurnCreditConfig,
)


class Plane:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def get_samples(self, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": self.ids}


@pytest.mark.asyncio
@pytest.mark.parametrize("shadow", [False, True])
async def test_adapter_maps_sample_id_and_preserves_mass(shadow):
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0]])
    advantages = mask * 2
    meta = SimpleNamespace(
        sample_ids=["s0"],
        tags=[{"rollout_debug": json.dumps({"trace_metadata": {"turns": 2}})}],
        partition_id="p",
    )
    labels = {
        "s0": RolloutTurnLabels(
            "g", 0, 2, ((0, 1), (1, -1)), aligned=True, step_tokens=(2, 2)
        )
    }
    cfg = TurnCreditConfig(
        enabled=True, apply=not shadow, judge={"endpoint_file": "/unused"}
    )
    plane = Plane(None)
    shaped, stats, _ = await adapter.apply_turn_credit(
        plane,
        meta,
        advantages,
        train_mask=mask,
        token_mask=mask,
        labels=labels,
        config=cfg,
        tokenizer_config={},
    )
    assert plane.calls == []
    assert stats.shaped_rows == 1
    assert torch.allclose((shaped * mask).sum(), (advantages * mask).sum())
    assert torch.equal(shaped, advantages) if shadow else shaped[0, 1] > shaped[0, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("has_tool", [False, True])
async def test_trailing_call_requires_no_tool_and_exact_token_counts(
    monkeypatch, has_tool
):
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
    meta = SimpleNamespace(
        sample_ids=["s0"],
        tags=[{"rollout_debug": {"trace_metadata": {"turns": 3}}}],
        partition_id="p",
    )
    labels = {
        "s0": RolloutTurnLabels(
            "g",
            0,
            2,
            ((0, 1),),
            aligned=True,
            trailing_empty_calls=1,
            step_tokens=(2, 2, 1),
        )
    }
    ids = torch.tensor([[0, 8, 8, 0, 8, 8, 0, 99 if has_tool else 8]])
    plane = Plane(ids)
    monkeypatch.setattr(adapter, "tool_call_token_id", lambda _: 99)
    cfg = TurnCreditConfig(enabled=True, judge={"endpoint_file": "/unused"})
    shaped, stats, _ = await adapter.apply_turn_credit(
        plane,
        meta,
        mask,
        train_mask=mask,
        token_mask=mask,
        labels=labels,
        config=cfg,
        tokenizer_config={},
    )
    assert plane.calls == [
        {"sample_ids": ["s0"], "partition_id": "p", "select_fields": ["input_ids"]}
    ]
    assert stats.shaped_rows == (0 if has_tool else 1)
    if has_tool:
        assert torch.equal(shaped, mask)
