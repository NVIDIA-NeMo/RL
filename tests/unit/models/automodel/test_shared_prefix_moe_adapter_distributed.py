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

"""Distributed oracle for the EP-free Qwen3-MoE checkpoint adapter.

Run with either of the following commands::

    torchrun --standalone --nproc-per-node=2 -m pytest -q \
        tests/unit/models/automodel/test_shared_prefix_moe_adapter_distributed.py
    torchrun --standalone --nproc-per-node=4 -m pytest -q \
        tests/unit/models/automodel/test_shared_prefix_moe_adapter_distributed.py

The production adapter deliberately stages checkpoint values in ordinary
rank-local placeholders.  DCP fills those placeholders, and ``from_hf`` copies
the fully validated values into the original FSDP DTensor storage.  This test
simulates that DCP write without needing a filesystem checkpoint.
"""

from __future__ import annotations

import math
import os
import re
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = [
    pytest.mark.automodel,
    pytest.mark.skipif(
        _WORLD_SIZE not in {2, 4} or "RANK" not in os.environ,
        reason="requires torchrun with two or four ranks",
    ),
]

_N_EXPERTS = 7
_HIDDEN = 6
_INTER = 2
_GATE_UP_KEY = "model.layers.0.mlp.experts.gate_and_up_projs"
_DOWN_KEY = "model.layers.0.mlp.experts.down_projs"
_EXPERT_KEY = re.compile(
    r"model\.layers\.0\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def _adapter():
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import (
        Qwen3MoeStateDictAdapter,
    )
    from nemo_automodel.components.moe.config import MoEConfig

    moe_config = MoEConfig(
        n_routed_experts=_N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=_HIDDEN,
        inter_dim=2 * _INTER,
        moe_inter_dim=_INTER,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.float32,
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )
    return Qwen3MoeStateDictAdapter(
        config=SimpleNamespace(),
        moe_config=moe_config,
        backend=backend,
        dtype=torch.float32,
    )


def _grouped_values(offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up = torch.arange(
        _N_EXPERTS * _HIDDEN * 2 * _INTER,
        dtype=torch.float32,
    ).reshape(_N_EXPERTS, _HIDDEN, 2 * _INTER)
    down = torch.arange(
        _N_EXPERTS * _INTER * _HIDDEN,
        dtype=torch.float32,
    ).reshape(_N_EXPERTS, _INTER, _HIDDEN)
    return gate_up + offset, down + 2.0 * offset


def _local_expert_ids(rank: int, world_size: int) -> tuple[int, ...]:
    # DTensor's Shard placement follows torch.chunk rather than balanced
    # divmod slices. Seven experts deliberately exercises uneven shards in
    # both supported world sizes.
    chunk_size = math.ceil(_N_EXPERTS / world_size)
    start = min(rank * chunk_size, _N_EXPERTS)
    stop = min(start + chunk_size, _N_EXPERTS)
    return tuple(range(start, stop))


def _expert_ids(hf_state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    ids = {
        int(match.group("expert"))
        for key in hf_state_dict
        if (match := _EXPERT_KEY.fullmatch(key)) is not None
    }
    return tuple(sorted(ids))


def _copy_checkpoint_values_into_placeholders(
    hf_state_dict: dict[str, torch.Tensor],
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> None:
    """Simulate DCP's in-place writes into adapter-created destinations."""
    for key, placeholder in hf_state_dict.items():
        match = _EXPERT_KEY.fullmatch(key)
        if match is None:
            continue
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if projection == "gate_proj":
            checkpoint_value = gate_up[expert, :, :_INTER].T
        elif projection == "up_proj":
            checkpoint_value = gate_up[expert, :, _INTER:].T
        else:
            checkpoint_value = down[expert].T
        placeholder.copy_(checkpoint_value)


def _assert_all_placeholders_are_contiguous(
    hf_state_dict: dict[str, torch.Tensor],
) -> None:
    expert_placeholders = [
        value for key, value in hf_state_dict.items() if _EXPERT_KEY.fullmatch(key)
    ]
    assert expert_placeholders
    assert all(value.is_contiguous() for value in expert_placeholders)


def _assert_hf_expert_values(
    hf_state_dict: dict[str, torch.Tensor],
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> None:
    """Verify native-to-HF save conversion before simulating a DCP load."""
    for key, value in hf_state_dict.items():
        match = _EXPERT_KEY.fullmatch(key)
        if match is None:
            continue
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if projection == "gate_proj":
            expected = gate_up[expert, :, :_INTER].T
        elif projection == "up_proj":
            expected = gate_up[expert, :, _INTER:].T
        else:
            expected = down[expert].T
        torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)


def _assert_regular_refit_conversion_does_not_stage(adapter=None) -> None:
    """A refit converts a gathered ordinary Tensor, not an FSDP local shard."""
    adapter = _adapter() if adapter is None else adapter
    source_gate_up, source_down = _grouped_values(offset=300.0)

    hf_state_dict = dict(
        adapter.convert_single_tensor_to_hf(_GATE_UP_KEY, source_gate_up)
    )
    hf_state_dict.update(adapter.convert_single_tensor_to_hf(_DOWN_KEY, source_down))

    assert _expert_ids(hf_state_dict) == tuple(range(_N_EXPERTS))
    _assert_all_placeholders_are_contiguous(hf_state_dict)

    target_gate_up, target_down = _grouped_values(offset=700.0)
    _copy_checkpoint_values_into_placeholders(
        hf_state_dict,
        target_gate_up,
        target_down,
    )
    restored = adapter.from_hf(hf_state_dict)

    torch.testing.assert_close(
        restored[_GATE_UP_KEY], target_gate_up, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(restored[_DOWN_KEY], target_down, rtol=0.0, atol=0.0)


def test_ep_free_checkpoint_adapter_staged_copy_oracle() -> None:
    from torch.distributed._tensor import Shard, distribute_tensor
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_ep_free_checkpoint_adapter,
    )

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group("gloo", timeout=timedelta(minutes=3))

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        mesh = DeviceMesh(
            "cpu",
            torch.arange(world_size),
            mesh_dim_names=("dp",),
        )
        enable_qwen3_moe_ep_free_checkpoint_adapter()

        initial_gate_up, initial_down = _grouped_values(offset=10.0)
        target_gate_up, target_down = _grouped_values(offset=1000.0)

        native_gate_up = distribute_tensor(initial_gate_up, mesh, [Shard(0)])
        native_down = distribute_tensor(initial_down, mesh, [Shard(0)])
        adapter = _adapter()
        hf_state_dict = adapter.to_hf(
            {
                _GATE_UP_KEY: native_gate_up,
                _DOWN_KEY: native_down,
            }
        )

        assert _expert_ids(hf_state_dict) == _local_expert_ids(rank, world_size)
        _assert_all_placeholders_are_contiguous(hf_state_dict)
        _assert_hf_expert_values(hf_state_dict, initial_gate_up, initial_down)

        # The staged-copy design must not mutate model storage until every
        # checkpoint key has been validated by from_hf.
        local_gate_up_before = native_gate_up.to_local().clone()
        local_down_before = native_down.to_local().clone()
        _copy_checkpoint_values_into_placeholders(
            hf_state_dict,
            target_gate_up,
            target_down,
        )
        torch.testing.assert_close(
            native_gate_up.to_local(), local_gate_up_before, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            native_down.to_local(), local_down_before, rtol=0.0, atol=0.0
        )

        restored = adapter.from_hf(hf_state_dict)
        assert restored[_GATE_UP_KEY] is native_gate_up
        assert restored[_DOWN_KEY] is native_down
        torch.testing.assert_close(
            native_gate_up.full_tensor(), target_gate_up, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            native_down.full_tensor(), target_down, rtol=0.0, atol=0.0
        )

        # A rollout refit can reuse the checkpoint adapter after a previous
        # bulk conversion was abandoned before ``from_hf``.  The first regular
        # full-Tensor conversion must discard that stale rank-local staging.
        adapter.to_hf(
            {
                _GATE_UP_KEY: native_gate_up,
                _DOWN_KEY: native_down,
            }
        )
        _assert_regular_refit_conversion_does_not_stage(adapter)

        # Missing-key validation must happen before any projection is copied.
        failing_gate_up = distribute_tensor(initial_gate_up, mesh, [Shard(0)])
        failing_down = distribute_tensor(initial_down, mesh, [Shard(0)])
        failing_adapter = _adapter()
        incomplete = failing_adapter.to_hf(
            {
                _GATE_UP_KEY: failing_gate_up,
                _DOWN_KEY: failing_down,
            }
        )
        _copy_checkpoint_values_into_placeholders(
            incomplete,
            target_gate_up,
            target_down,
        )
        missing_expert = _local_expert_ids(rank, world_size)[0]
        del incomplete[f"model.layers.0.mlp.experts.{missing_expert}.up_proj.weight"]
        failing_gate_up_before = failing_gate_up.to_local().clone()
        failing_down_before = failing_down.to_local().clone()

        with pytest.raises(
            (RuntimeError, ValueError),
            match=r"(?i)(missing|expected|checkpoint|expert)",
        ):
            failing_adapter.from_hf(incomplete)

        torch.testing.assert_close(
            failing_gate_up.to_local(),
            failing_gate_up_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            failing_down.to_local(), failing_down_before, rtol=0.0, atol=0.0
        )

        # NeMo-RL's refit path calls full_tensor() before adapting each tensor.
        # Its ordinary full-Tensor conversion must therefore retain all experts
        # and the original from_hf rebuild semantics on every DP rank.
        _assert_regular_refit_conversion_does_not_stage()
    finally:
        if owns_process_group:
            dist.destroy_process_group()
