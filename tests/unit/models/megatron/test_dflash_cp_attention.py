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

"""Packed-boundary and communication-width tests for CP DFlash attention."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module(name: str) -> ModuleType:
    module_name = f"nemo_rl.models.megatron.draft.{name}"
    path = _REPO_ROOT / f"nemo_rl/models/megatron/draft/{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_cp2_plan() -> tuple[object, object]:
    layout_module = _load_module("sequence_layout")
    plan_module = _load_module("block_plan")
    sample_ids = torch.tensor([101, 303], dtype=torch.int64)
    layout = layout_module.build_draft_sequence_layout(
        logical_sample_ids=sample_ids,
        cu_seqlens_q=torch.tensor([0, 4, 8], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int64),
        cp_rank=1,
        cp_size=2,
        tp_rank=0,
        tp_size=1,
        device=torch.device("cpu"),
    )
    plan = plan_module.build_dflash_batch_plan(
        torch.ones(2, 4, dtype=torch.bool),
        sample_ids,
        anchors_per_sample=1,
        gamma=1,
        optimizer_step=0,
        seed=0,
        sequence_layout=layout,
    )
    return layout, plan


def test_cp_attention_gathers_projected_width_without_cross_segment_visibility() -> (
    None
):
    layout, plan = _make_cp2_plan()
    _load_module("context_parallel")
    attention = _load_module("block_attention")
    full_key = torch.arange(16, dtype=torch.float32).reshape(1, 8, 1, 2) / 10
    full_value = torch.arange(16, dtype=torch.float32).reshape(1, 8, 1, 2)
    local_key = full_key[:, layout.cp_global_positions].clone()
    local_value = full_value[:, layout.cp_global_positions].clone()
    block_query = torch.ones(2, 2, 1, 2)
    block_key = torch.zeros(2, 2, 1, 2)
    block_value = torch.zeros(2, 2, 1, 2)
    cp_group = MagicMock(name="cp_group")
    gathered_inputs: list[torch.Tensor] = []

    def run(candidate_value: torch.Tensor) -> torch.Tensor:
        outputs = iter((full_key, candidate_value))

        def record_gather(projected: torch.Tensor, **kwargs: object) -> torch.Tensor:
            gathered_inputs.append(projected)
            assert kwargs["sequence_layout"] is layout
            assert kwargs["cp_group"] is cp_group
            assert kwargs["sequence_dim"] == 1
            return next(outputs)

        attention.gather_projected_kv = record_gather
        return attention.dflash_block_only_attention(
            plan=plan,
            trunk_k=local_key,
            trunk_v=local_value,
            block_q=block_query,
            block_k=block_key,
            block_v=block_value,
            sequence_layout=layout,
            context_parallel_group=cp_group,
        )

    reference = run(full_value)
    mutated_value = full_value.clone()
    mutated_value[:, 4:] += 1000
    mutated = run(mutated_value)

    assert len(gathered_inputs) == 4
    assert all(tensor.shape[-2:] == (1, 2) for tensor in gathered_inputs)
    assert all(
        tensor.shape[1] == layout.cp_global_positions.numel()
        for tensor in gathered_inputs
    )
    torch.testing.assert_close(reference[0], mutated[0])
    assert not torch.equal(reference[1], mutated[1])
