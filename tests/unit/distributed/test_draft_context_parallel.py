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

"""Distributed contracts for draft TP/SP and CP sequence communication."""

from __future__ import annotations

import importlib.util
import math
import os
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.mcore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAYOUT_MODULE = "nemo_rl.models.megatron.draft.sequence_layout"
_LAYOUT_PATH = _REPO_ROOT / "nemo_rl/models/megatron/draft/sequence_layout.py"
_SP_MODULE = "nemo_rl.models.megatron.draft.sequence_parallel"
_SP_PATH = _REPO_ROOT / "nemo_rl/models/megatron/draft/sequence_parallel.py"
_CP_MODULE = "nemo_rl.models.megatron.draft.context_parallel"
_CP_PATH = _REPO_ROOT / "nemo_rl/models/megatron/draft/context_parallel.py"
_PLAN_MODULE = "nemo_rl.models.megatron.draft.block_plan"
_PLAN_PATH = _REPO_ROOT / "nemo_rl/models/megatron/draft/block_plan.py"
_ATTENTION_MODULE = "nemo_rl.models.megatron.draft.block_attention"
_ATTENTION_PATH = _REPO_ROOT / "nemo_rl/models/megatron/draft/block_attention.py"


def _load_module(module_name: str, path: Path) -> ModuleType:
    if not path.is_file():
        pytest.fail(f"Production contract is missing: {path}", pytrace=False)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise RuntimeError(f"Unable to create a module spec for {path}")
    if spec.loader is None:
        raise RuntimeError(f"Module spec has no loader for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_layout(*, tp_rank: int, tp_size: int) -> Any:
    module = _load_module(_LAYOUT_MODULE, _LAYOUT_PATH)
    return module.build_draft_sequence_layout(
        logical_sample_ids=torch.tensor([101, 303], dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, 5, 8], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([0, 8, 16], dtype=torch.int64),
        cp_rank=1,
        cp_size=2,
        tp_rank=tp_rank,
        tp_size=tp_size,
        device=torch.device("cpu"),
    )


def _build_cp_layout(*, cp_rank: int, cp_size: int) -> Any:
    module = _load_module(_LAYOUT_MODULE, _LAYOUT_PATH)
    padding_multiple = 2 * cp_size
    first_padded = ((5 + padding_multiple - 1) // padding_multiple) * padding_multiple
    second_padded = ((3 + padding_multiple - 1) // padding_multiple) * padding_multiple
    return module.build_draft_sequence_layout(
        logical_sample_ids=torch.tensor([101, 303], dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, 5, 8], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor(
            [0, first_padded, first_padded + second_padded],
            dtype=torch.int64,
        ),
        cp_rank=cp_rank,
        cp_size=cp_size,
        tp_rank=0,
        tp_size=1,
        device=torch.device("cpu"),
    )


def _build_tp_cp_layout(
    *,
    tp_rank: int,
    tp_size: int,
    cp_rank: int,
    cp_size: int,
) -> Any:
    module = _load_module(_LAYOUT_MODULE, _LAYOUT_PATH)
    padding_multiple = 2 * cp_size
    first_padded = ((5 + padding_multiple - 1) // padding_multiple) * padding_multiple
    second_padded = ((3 + padding_multiple - 1) // padding_multiple) * padding_multiple
    return module.build_draft_sequence_layout(
        logical_sample_ids=torch.tensor([101, 303], dtype=torch.int64),
        cu_seqlens_q=torch.tensor([0, 5, 8], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor(
            [0, first_padded, first_padded + second_padded],
            dtype=torch.int64,
        ),
        cp_rank=cp_rank,
        cp_size=cp_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        device=torch.device("cpu"),
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_tp_reconstruction(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    interface_names = {name for _, name in socket.if_nameindex()}
    for loopback_name in ("lo", "lo0"):
        if loopback_name in interface_names:
            os.environ["GLOO_SOCKET_IFNAME"] = loopback_name
            break
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        module = _load_module(_SP_MODULE, _SP_PATH)
        layout = _build_layout(tp_rank=rank, tp_size=world_size)
        full = torch.arange(8, dtype=torch.float64).reshape(8, 1)
        local = full[layout.sp_local_positions].clone().requires_grad_(True)

        reconstructed = module.reconstruct_tp_sequence(
            local,
            sequence_layout=layout,
            tp_group=dist.group.WORLD,
            sequence_dim=0,
        )

        assert torch.equal(reconstructed, full)
        (reconstructed.square().sum() / world_size).backward()
        assert local.grad is not None
        assert torch.equal(local.grad, 2 * local.detach())
    finally:
        dist.destroy_process_group()


def test_tp_sequence_reconstruction_is_ordered_and_autograd_safe() -> None:
    """Catches detached, rank-reordered, or backward-incorrect TP gathers."""
    world_size = 2
    mp.spawn(
        _run_tp_reconstruction,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )


def test_tp1_sequence_reconstruction_is_identity() -> None:
    """Catches needless copies or collectives on the dense TP1 path."""
    module = _load_module(_SP_MODULE, _SP_PATH)
    layout = _build_layout(tp_rank=0, tp_size=1)
    local = torch.arange(8, dtype=torch.float32).reshape(8, 1)

    reconstructed = module.reconstruct_tp_sequence(
        local,
        sequence_layout=layout,
        tp_group=None,
        sequence_dim=0,
    )

    assert reconstructed is local


def _run_projected_cp_gather(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    interface_names = {name for _, name in socket.if_nameindex()}
    for loopback_name in ("lo", "lo0"):
        if loopback_name in interface_names:
            os.environ["GLOO_SOCKET_IFNAME"] = loopback_name
            break
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        module = _load_module(_CP_MODULE, _CP_PATH)
        layout = _build_cp_layout(cp_rank=rank, cp_size=world_size)
        total_length = int(layout.cu_seqlens_q_padded[-1])
        full = torch.arange(total_length * 2, dtype=torch.float64).reshape(
            total_length,
            1,
            1,
            2,
        )
        local = full[layout.cp_global_positions].clone().requires_grad_(True)

        gathered = module.gather_projected_kv(
            local,
            sequence_layout=layout,
            cp_group=dist.group.WORLD,
            sequence_dim=0,
        )

        assert torch.equal(gathered, full)
        (gathered.square().sum() / world_size).backward()
        assert local.grad is not None
        assert torch.equal(local.grad, 2 * local.detach())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_projected_kv_gather_restores_packed_order_with_autograd(cp_size: int) -> None:
    """Catches rank-major output, per-pack reordering, and broken gradients."""
    mp.spawn(
        _run_projected_cp_gather,
        args=(cp_size, _find_free_port()),
        nprocs=cp_size,
        join=True,
    )


def test_projected_kv_cp1_is_identity() -> None:
    module = _load_module(_CP_MODULE, _CP_PATH)
    layout = _build_cp_layout(cp_rank=0, cp_size=1)
    local = torch.randn(layout.cp_global_positions.numel(), 1, 2, 3)

    gathered = module.gather_projected_kv(
        local,
        sequence_layout=layout,
        cp_group=None,
        sequence_dim=0,
    )

    assert gathered is local


def _run_tp_cp_reconstruction_and_gather(
    rank: int,
    tp_size: int,
    cp_size: int,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    interface_names = {name for _, name in socket.if_nameindex()}
    for loopback_name in ("lo", "lo0"):
        if loopback_name in interface_names:
            os.environ["GLOO_SOCKET_IFNAME"] = loopback_name
            break
    world_size = tp_size * cp_size
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        tp_groups = [
            dist.new_group(
                ranks=list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size))
            )
            for cp_rank in range(cp_size)
        ]
        cp_groups = [
            dist.new_group(
                ranks=[cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)]
            )
            for tp_rank in range(tp_size)
        ]
        tp_rank = rank % tp_size
        cp_rank = rank // tp_size
        layout = _build_tp_cp_layout(
            tp_rank=tp_rank,
            tp_size=tp_size,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )
        sequence_parallel_module = _load_module(_SP_MODULE, _SP_PATH)
        context_parallel_module = _load_module(_CP_MODULE, _CP_PATH)
        total_length = int(layout.cu_seqlens_q_padded[-1])
        full = torch.arange(total_length * 2, dtype=torch.float64).reshape(
            total_length,
            1,
            1,
            2,
        )
        cp_local = full[layout.cp_global_positions]
        sp_local = cp_local[layout.sp_local_positions].clone().requires_grad_(True)

        reconstructed = sequence_parallel_module.reconstruct_tp_sequence(
            sp_local,
            sequence_layout=layout,
            tp_group=tp_groups[cp_rank],
            sequence_dim=0,
        )
        assert torch.equal(reconstructed, cp_local)
        lane_offset = tp_rank * 1_000
        projected = reconstructed + lane_offset

        gathered = context_parallel_module.gather_projected_kv(
            projected,
            sequence_layout=layout,
            cp_group=cp_groups[tp_rank],
            sequence_dim=0,
        )
        assert torch.equal(gathered, full + lane_offset)
        (((gathered - lane_offset).square().sum()) / world_size).backward()
        assert sp_local.grad is not None
        assert torch.equal(sp_local.grad, 2 * sp_local.detach())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_tp2_cp_reconstruction_and_projected_gather_are_group_isolated(
    cp_size: int,
) -> None:
    """Target-SP and CP gathers must compose on orthogonal process groups."""
    tp_size = 2
    mp.spawn(
        _run_tp_cp_reconstruction_and_gather,
        args=(tp_size, cp_size, _find_free_port()),
        nprocs=tp_size * cp_size,
        join=True,
    )


def _run_cuda_cp_zero_owner_attention(
    rank: int,
    world_size: int,
    port: int,
    all_zero: bool,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        layout_module = _load_module(_LAYOUT_MODULE, _LAYOUT_PATH)
        plan_module = _load_module(_PLAN_MODULE, _PLAN_PATH)
        _load_module(_CP_MODULE, _CP_PATH)
        attention_module = _load_module(_ATTENTION_MODULE, _ATTENTION_PATH)
        device = torch.device("cuda", rank)
        sample_ids = torch.tensor([101], dtype=torch.int64, device=device)
        layout = layout_module.build_draft_sequence_layout(
            logical_sample_ids=sample_ids,
            cu_seqlens_q=torch.tensor([0, 16], dtype=torch.int64, device=device),
            cu_seqlens_q_padded=torch.tensor(
                [0, 16], dtype=torch.int64, device=device
            ),
            cp_rank=rank,
            cp_size=world_size,
            tp_rank=0,
            tp_size=1,
            device=device,
        )
        token_valid = torch.zeros if all_zero else torch.ones
        plan = plan_module.build_dflash_batch_plan(
            token_valid(1, 16, dtype=torch.bool, device=device),
            sample_ids,
            anchors_per_sample=1,
            gamma=2,
            optimizer_step=0,
            seed=0,
            sequence_layout=layout,
        )
        local_blocks = torch.tensor(
            [plan.sample_rows.numel()], dtype=torch.int64, device=device
        )
        owner_counts = [torch.zeros_like(local_blocks) for _ in range(world_size)]
        dist.all_gather(owner_counts, local_blocks)
        expected_owners = 0 if all_zero else 1
        assert sum(int(count.item()) for count in owner_counts) == expected_owners
        assert any(int(count.item()) == 0 for count in owner_counts)

        full_shape = (1, 16, 1, 16)
        full_k_value = torch.arange(
            math.prod(full_shape), dtype=torch.float32, device=device
        ).reshape(full_shape) / 100
        full_v_value = torch.flip(full_k_value, dims=(1,))
        local_positions = layout.cp_global_positions
        trunk_k = (
            full_k_value[:, local_positions].detach().clone().requires_grad_(True)
        )
        trunk_v = (
            full_v_value[:, local_positions].detach().clone().requires_grad_(True)
        )
        block_shape = (plan.sample_rows.numel(), plan.block_size, 1, 16)
        full_block_shape = (expected_owners, plan.block_size, 1, 16)
        full_block_value = torch.arange(
            math.prod(full_block_shape), dtype=torch.float32, device=device
        ).reshape(full_block_shape) / 50
        block_q = full_block_value[: block_shape[0]].clone().requires_grad_(True)
        block_k = (full_block_value[: block_shape[0]] + 0.25).requires_grad_(True)
        block_v = (full_block_value[: block_shape[0]] - 0.25).requires_grad_(True)

        output = attention_module.dflash_block_only_attention(
            plan=plan,
            trunk_k=trunk_k,
            trunk_v=trunk_v,
            block_q=block_q,
            block_k=block_k,
            block_v=block_v,
            sequence_layout=layout,
            context_parallel_group=dist.group.WORLD,
        )
        assert output.shape == block_shape
        output.float().sum().backward()
        assert trunk_k.grad is not None
        assert trunk_v.grad is not None
        assert block_q.grad is not None
        assert block_k.grad is not None
        assert block_v.grad is not None

        oracle_layout = layout_module.build_draft_sequence_layout(
            logical_sample_ids=sample_ids,
            cu_seqlens_q=torch.tensor([0, 16], dtype=torch.int64, device=device),
            cu_seqlens_q_padded=torch.tensor(
                [0, 16], dtype=torch.int64, device=device
            ),
            cp_rank=0,
            cp_size=1,
            tp_rank=0,
            tp_size=1,
            device=device,
        )
        oracle_plan = plan_module.build_dflash_batch_plan(
            token_valid(1, 16, dtype=torch.bool, device=device),
            sample_ids,
            anchors_per_sample=1,
            gamma=2,
            optimizer_step=0,
            seed=0,
            sequence_layout=oracle_layout,
        )
        oracle_k = full_k_value.detach().clone().requires_grad_(True)
        oracle_v = full_v_value.detach().clone().requires_grad_(True)
        oracle_q = full_block_value.detach().clone().requires_grad_(True)
        oracle_block_k = (full_block_value.detach().clone() + 0.25).requires_grad_(
            True
        )
        oracle_block_v = (full_block_value.detach().clone() - 0.25).requires_grad_(
            True
        )
        oracle_output = attention_module.dflash_block_only_attention(
            plan=oracle_plan,
            trunk_k=oracle_k,
            trunk_v=oracle_v,
            block_q=oracle_q,
            block_k=oracle_block_k,
            block_v=oracle_block_v,
            sequence_layout=oracle_layout,
        )
        oracle_output.float().sum().backward()
        assert oracle_k.grad is not None
        assert oracle_v.grad is not None
        torch.testing.assert_close(trunk_k.grad, oracle_k.grad[:, local_positions])
        torch.testing.assert_close(trunk_v.grad, oracle_v.grad[:, local_positions])
        if block_shape[0]:
            assert oracle_q.grad is not None
            assert oracle_block_k.grad is not None
            assert oracle_block_v.grad is not None
            torch.testing.assert_close(output, oracle_output)
            torch.testing.assert_close(block_q.grad, oracle_q.grad)
            torch.testing.assert_close(block_k.grad, oracle_block_k.grad)
            torch.testing.assert_close(block_v.grad, oracle_block_v.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("all_zero", [False, True], ids=["one_owner", "all_zero"])
def test_cuda_cp_zero_owner_runs_flex_attention_forward_backward(
    all_zero: bool,
) -> None:
    """Both CP ranks must enter K/V collectives when one owns no draft blocks."""
    world_size = 2
    mp.spawn(
        _run_cuda_cp_zero_owner_attention,
        args=(world_size, _find_free_port(), all_zero),
        nprocs=world_size,
        join=True,
    )
