# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Behavioral contract tests for vectorized DFlash batch planning."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor

pytestmark = pytest.mark.mcore

_PLAN_MODULE = "nemo_rl.models.megatron.draft.block_plan"


def _load_plan_module() -> ModuleType:
    try:
        return importlib.import_module(_PLAN_MODULE)
    except ModuleNotFoundError as error:
        pytest.fail(
            f"DFlash batch-plan production contract is missing: {error}",
            pytrace=False,
        )


def _build_plan(
    token_valid_mask: Tensor,
    sample_ids: Tensor,
    *,
    anchors_per_sample: int,
    gamma: int,
    optimizer_step: int,
    seed: int,
) -> Any:
    module = _load_plan_module()
    return module.build_dflash_batch_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=anchors_per_sample,
        gamma=gamma,
        optimizer_step=optimizer_step,
        seed=seed,
    )


def test_plan_has_fixed_shapes_and_excludes_conditioning_slot_from_loss() -> None:
    """Catches reordered slots, mutable metadata, and loss on the anchor slot."""
    token_valid_mask = torch.ones((2, 17), dtype=torch.bool)
    sample_ids = torch.tensor([101, 102], dtype=torch.int64)

    plan = _build_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=3,
        gamma=3,
        optimizer_step=7,
        seed=11,
    )

    assert dataclasses.is_dataclass(plan)
    assert type(plan).__dataclass_params__.frozen
    assert not hasattr(plan, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.gamma = 9

    assert plan.batch_size == 2
    assert plan.sequence_length == 17
    assert plan.anchors_per_sample == 3
    assert plan.gamma == 3
    assert plan.block_size == 4
    assert plan.token_valid_mask.shape == (2, 17)
    assert plan.sample_rows.shape == (6,)
    assert plan.anchor_ids.shape == (6,)
    assert plan.anchor_positions.shape == (6,)
    assert plan.trunk_lengths.shape == (6,)
    assert plan.query_positions.shape == (6, 4)
    assert plan.label_positions.shape == (6, 4)
    assert plan.block_valid.shape == (6,)
    assert plan.slot_valid.shape == (6, 4)
    assert plan.loss_mask.shape == (6, 4)

    assert torch.equal(plan.sample_rows, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert torch.equal(
        plan.anchor_ids,
        torch.tensor([2407, 2466, 2525, 2424, 2483, 2542]),
    )
    assert torch.equal(plan.anchor_positions, torch.tensor([7, 4, 6, 7, 1, 7]))
    assert torch.equal(plan.trunk_lengths, plan.anchor_positions)
    assert torch.equal(
        plan.query_positions,
        torch.tensor(
            [
                [7, 8, 9, 10],
                [4, 5, 6, 7],
                [6, 7, 8, 9],
                [7, 8, 9, 10],
                [1, 2, 3, 4],
                [7, 8, 9, 10],
            ]
        ),
    )
    assert torch.equal(plan.label_positions, plan.query_positions)
    assert plan.block_valid.all()
    assert plan.slot_valid.all()
    assert torch.equal(
        plan.loss_mask,
        torch.tensor([[False, True, True, True]]).expand(6, -1),
    )
    assert plan.anchor_positions.device == token_valid_mask.device


def test_anchor_identity_is_stable_across_batch_permutation_and_slicing() -> None:
    """Catches accidental use of transient batch-row indices as sample identity."""
    token_valid_mask = torch.tensor(
        [
            [True] * 12,
            [True] * 10 + [False] * 2,
            [True] * 11 + [False],
        ]
    )
    sample_ids = torch.tensor([101, 303, 202], dtype=torch.int64)
    full = _build_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=2,
        gamma=2,
        optimizer_step=19,
        seed=5,
    )

    permutation = torch.tensor([2, 0, 1])
    permuted = _build_plan(
        token_valid_mask[permutation],
        sample_ids[permutation],
        anchors_per_sample=2,
        gamma=2,
        optimizer_step=19,
        seed=5,
    )
    sliced = _build_plan(
        token_valid_mask[1:],
        sample_ids[1:],
        anchors_per_sample=2,
        gamma=2,
        optimizer_step=19,
        seed=5,
    )

    full_ids = full.anchor_ids.reshape(3, 2)
    full_positions = full.anchor_positions.reshape(3, 2)
    assert torch.equal(permuted.anchor_ids.reshape(3, 2), full_ids[permutation])
    assert torch.equal(
        permuted.anchor_positions.reshape(3, 2),
        full_positions[permutation],
    )
    assert torch.equal(sliced.anchor_ids.reshape(2, 2), full_ids[1:])
    assert torch.equal(sliced.anchor_positions.reshape(2, 2), full_positions[1:])


def test_duplicate_sample_ids_intentionally_repeat_anchor_schedule() -> None:
    """Catches hidden row-dependent entropy for duplicate stable sample IDs."""
    token_valid_mask = torch.ones((3, 9), dtype=torch.bool)
    sample_ids = torch.tensor([44, 9, 44], dtype=torch.int64)

    plan = _build_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=4,
        gamma=2,
        optimizer_step=3,
        seed=8,
    )

    anchor_ids = plan.anchor_ids.reshape(3, 4)
    anchors = plan.anchor_positions.reshape(3, 4)
    assert torch.equal(anchor_ids[0], anchor_ids[2])
    assert torch.equal(anchors[0], anchors[2])
    assert not torch.equal(anchor_ids[0], anchor_ids[1])


def test_optimizer_step_and_seed_change_stable_anchor_identity() -> None:
    """Catches a supposedly random schedule that ignores seed or training step."""
    token_valid_mask = torch.ones((1, 17), dtype=torch.bool)
    sample_ids = torch.tensor([101], dtype=torch.int64)
    base = _build_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=3,
        gamma=3,
        optimizer_step=7,
        seed=11,
    )
    next_step = _build_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=3,
        gamma=3,
        optimizer_step=8,
        seed=11,
    )
    next_seed = _build_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=3,
        gamma=3,
        optimizer_step=7,
        seed=12,
    )

    assert torch.equal(next_step.anchor_ids - base.anchor_ids, torch.full((3,), 31))
    assert torch.equal(next_seed.anchor_ids - base.anchor_ids, torch.full((3,), 43))
    assert not torch.equal(next_step.anchor_positions, base.anchor_positions)
    assert not torch.equal(next_seed.anchor_positions, base.anchor_positions)


def test_anchor_schedule_does_not_alias_when_candidate_count_divides_stride() -> None:
    """Catches a linear modulo schedule collapsing every anchor to one position."""
    gamma = 3
    block_size = gamma + 1
    candidate_count = 59
    sequence_length = candidate_count + block_size - 1

    plan = _build_plan(
        torch.ones((1, sequence_length), dtype=torch.bool),
        torch.tensor([101], dtype=torch.int64),
        anchors_per_sample=8,
        gamma=gamma,
        optimizer_step=7,
        seed=11,
    )

    assert plan.anchor_positions.unique().numel() > 1


def test_tail_and_empty_rows_emit_only_safe_invalid_slots() -> None:
    """Catches tail-crossing anchors and out-of-range gathers for short rows."""
    token_valid_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
            [False, False, False, False, False, False],
        ]
    )
    sample_ids = torch.tensor([1, 2, 3], dtype=torch.int64)

    plan = _build_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=2,
        gamma=3,
        optimizer_step=0,
        seed=0,
    )

    assert torch.equal(
        plan.block_valid,
        torch.tensor([True, True, False, False, False, False]),
    )
    assert torch.equal(
        plan.query_positions[:2],
        torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]]),
    )
    assert torch.equal(plan.query_positions[2:], torch.zeros((4, 4), dtype=torch.int64))
    assert torch.equal(plan.label_positions[2:], torch.zeros((4, 4), dtype=torch.int64))
    assert torch.equal(plan.trunk_lengths, torch.tensor([1, 0, 0, 0, 0, 0]))
    assert plan.slot_valid[:2].all()
    assert not plan.slot_valid[2:].any()
    assert not plan.loss_mask[:, 0].any()
    assert not plan.loss_mask[2:].any()
    assert torch.all(plan.query_positions >= 0)
    assert torch.all(plan.query_positions < token_valid_mask.shape[1])


def test_noncontiguous_masks_schedule_only_full_valid_windows() -> None:
    token_valid_mask = torch.tensor(
        [[False, True, True, True, False, True, True, True]]
    )

    plan = _build_plan(
        token_valid_mask,
        torch.tensor([7], dtype=torch.int64),
        anchors_per_sample=8,
        gamma=2,
        optimizer_step=0,
        seed=0,
    )

    assert plan.block_valid.all()
    assert set(plan.anchor_positions.tolist()) <= {1, 5}
    assert plan.slot_valid.all()
    assert torch.equal(
        plan.trunk_lengths,
        torch.where(plan.anchor_positions == 1, 0, 3),
    )


def test_noncontiguous_mask_without_full_window_emits_safe_invalid_blocks() -> None:
    token_valid_mask = torch.tensor([[True, False, True]])

    plan = _build_plan(
        token_valid_mask,
        torch.tensor([7], dtype=torch.int64),
        anchors_per_sample=2,
        gamma=1,
        optimizer_step=0,
        seed=0,
    )

    assert not plan.block_valid.any()
    assert torch.equal(plan.anchor_positions, torch.zeros(2, dtype=torch.int64))
    assert torch.equal(plan.query_positions, torch.zeros((2, 2), dtype=torch.int64))
    assert not plan.slot_valid.any()
    assert not plan.loss_mask.any()


def test_plan_builder_has_no_host_sync_or_python_row_anchor_loops() -> None:
    """Catches capture-hostile scalar extraction, host copies, and Python loops."""
    module = _load_plan_module()
    source = inspect.getsource(module.build_dflash_batch_plan)
    tree = ast.parse(source)

    forbidden_attributes = {"cpu", "item", "tolist"}
    used_forbidden_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
    }
    loop_nodes = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    assert not used_forbidden_attributes
    assert not any(isinstance(node, loop_nodes) for node in ast.walk(tree))
