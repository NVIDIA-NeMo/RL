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

"""CP ownership and coordinate contracts for DSpark sampled blocks."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.models.megatron.draft.block_plan import build_dspark_batch_plan
from nemo_rl.models.megatron.draft.sequence_layout import build_draft_sequence_layout

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize("cp_size", [2, 4])
def test_dspark_plan_owns_complete_query_label_windows_exactly_once(
    cp_size: int,
) -> None:
    padded_length = 8 * cp_size
    logical_length = padded_length - 3
    token_valid_mask = torch.ones((1, logical_length), dtype=torch.bool)
    sample_ids = torch.tensor([91], dtype=torch.int64)
    plans = []

    for cp_rank in range(cp_size):
        layout = build_draft_sequence_layout(
            logical_sample_ids=sample_ids,
            cu_seqlens_q=torch.tensor([0, logical_length], dtype=torch.int64),
            cu_seqlens_q_padded=torch.tensor([0, padded_length], dtype=torch.int64),
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=0,
            tp_size=2,
            device=torch.device("cpu"),
        )
        plan = build_dspark_batch_plan(
            token_valid_mask,
            sample_ids,
            anchors_per_sample=4,
            block_size=3,
            optimizer_step=7,
            seed=11,
            sequence_layout=layout,
        )
        plans.append(plan)

        assert plan.query_positions.shape == (plan.sample_rows.numel(), 3)
        assert plan.label_positions.shape == plan.query_positions.shape
        assert torch.equal(
            plan.local_query_positions,
            layout.cp_global_to_local[plan.global_query_positions],
        )
        assert torch.equal(
            plan.local_label_positions,
            layout.cp_global_to_local[plan.global_label_positions],
        )
        assert torch.equal(
            plan.owner_cp_ranks,
            torch.full_like(plan.owner_cp_ranks, cp_rank),
        )
        assert torch.equal(
            plan.global_label_positions,
            plan.global_query_positions + 1,
        )
        assert torch.equal(
            plan.packed_rope_positions,
            plan.global_query_positions,
        )
        assert torch.equal(
            plan.packed_label_rope_positions,
            plan.global_label_positions,
        )

    owned_anchor_ids = torch.cat([plan.anchor_ids for plan in plans])
    assert owned_anchor_ids.unique().numel() == owned_anchor_ids.numel()


def test_dspark_plan_drops_a_query_label_window_crossing_cp_ownership() -> None:
    layout = build_draft_sequence_layout(
        logical_sample_ids=torch.tensor([5], dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, 7], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([0, 8], dtype=torch.int64),
        cp_rank=0,
        cp_size=2,
        tp_rank=0,
        tp_size=1,
        device=torch.device("cpu"),
    )
    plan = build_dspark_batch_plan(
        torch.ones((1, 7), dtype=torch.bool),
        torch.tensor([5], dtype=torch.int64),
        anchors_per_sample=8,
        block_size=3,
        optimizer_step=0,
        seed=0,
        sequence_layout=layout,
    )

    assert plan.excluded_window_count > 0
    assert plan.boundary_valid_mask.all()
    assert torch.all(plan.owner_cp_ranks[:, None] == layout.owner_cp_rank[plan.global_label_positions])
