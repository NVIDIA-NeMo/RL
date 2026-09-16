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

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _load_symbols():
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[3] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.eagle_ttt")
    return (
        module.EagleTTTAttentionPlan,
        module.EagleTTTState,
        module.eagle_ttt_attention,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(2026)
    shape = (1, 4, 7, 8)
    tensors = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(9)]
    return tuple(tensors)


def _run_attention(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    EagleTTTAttentionPlan, EagleTTTState, eagle_ttt_attention = _load_symbols()
    differentiable = tuple(tensor.detach().requires_grad_(True) for tensor in tensors)
    query, trunk_key, trunk_value, *branches = differentiable
    branch_keys = tuple(branches[:3])
    branch_values = tuple(branches[3:])
    state = EagleTTTState.from_trunk(
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        pass_count=4,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for key, value in zip(branch_keys, branch_values, strict=True):
        state = state.append_branch(branch_key=key, branch_value=value)
    plan = EagleTTTAttentionPlan(
        pass_index=3,
        pass_count=4,
        max_passes=8,
        sequence_length=query.shape[2],
    )
    output = eagle_ttt_attention(query=query, state=state, plan=plan)
    gradients = torch.autograd.grad(output.float().square().sum(), differentiable)
    return output.detach(), tuple(gradient.detach() for gradient in gradients)


def _tp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        full_inputs = _inputs()
        local_inputs = tuple(
            tensor.chunk(world_size, dim=1)[rank].contiguous() for tensor in full_inputs
        )
        local_output, local_gradients = _run_attention(local_inputs)
        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output)
        gathered_gradients: list[list[torch.Tensor]] = []
        for gradient in local_gradients:
            gathered = [torch.empty_like(gradient) for _ in range(world_size)]
            dist.all_gather(gathered, gradient)
            gathered_gradients.append(gathered)

        if rank == 0:
            expected_output, expected_gradients = _run_attention(full_inputs)
            actual_output = torch.cat(gathered_outputs, dim=1)
            torch.testing.assert_close(
                actual_output,
                expected_output,
                atol=5e-2,
                rtol=5e-2,
            )
            for gathered, expected in zip(
                gathered_gradients,
                expected_gradients,
                strict=True,
            ):
                actual = torch.cat(gathered, dim=1)
                torch.testing.assert_close(actual, expected, atol=8e-2, rtol=8e-2)
                assert torch.count_nonzero(actual) > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Gloo distributed support is required",
)
def test_tp2_bfloat16_matches_dense_output_and_all_gradients(tmp_path: Path) -> None:
    init_file = tmp_path / "eagle-ttt-tp2-init"
    mp.spawn(
        _tp_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )


class _SessionModel(torch.nn.Module):
    def __init__(self, core_attention: torch.nn.Module) -> None:
        super().__init__()
        self.core_attention = core_attention

    def rotary_pos_emb(self, sequence_length: int) -> None:
        del sequence_length
        return None

    def forward(
        self,
        *,
        embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rotary_pos_emb: torch.Tensor | None = None,
        packed_seq_params: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del embeddings, rotary_pos_emb
        sequence, batch, hidden = hidden_states.shape
        query = hidden_states.reshape(sequence, batch, 2, hidden // 2)
        output = self.core_attention(
            query,
            query,
            query,
            attention_mask,
            attn_mask_type=None,
            attention_bias=None,
            packed_seq_params=packed_seq_params,
        )
        return output, output


def _tp_layout_agreement_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _load_symbols()
        module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]
        core = module.EagleTTTCoreAttention(
            config=SimpleNamespace(context_parallel_size=1),
            layer_number=1,
            attn_mask_type=None,
            attention_type="self",
            cp_comm_type=None,
            softmax_scale=None,
            pg_collection=SimpleNamespace(tp=dist.group.WORLD),
        )
        session = module.MCoreEagleTTTSession(_SessionModel(core))
        storage = module.EagleTTTStoragePlan(
            batch_size=1,
            kv_heads=1,
            sequence_length=6,
            head_dim=4,
            dtype=torch.float32,
            pass_count=1,
            max_passes=8,
            activation_budget_bytes=1 << 20,
            layer_count=1,
            hidden_size=4,
            rope_dim=4,
        )
        agreed = module.EagleTTTSequenceLayout.from_cu_seqlens(
            cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32),
            sequence_length=6,
        )
        session.begin(
            layout=agreed,
            storage_plan=storage,
            excluded_tensors=(),
            resource_ledger=module.EagleTTTResourceLedger(limit_bytes=1 << 20),
        )
        session.reset()

        document_ids = agreed.document_ids.clone()
        if rank == 1:
            document_ids[0, -1] = 0
        mismatched = module.EagleTTTSequenceLayout(
            valid_tokens=agreed.valid_tokens,
            document_ids=document_ids,
        )
        with pytest.raises(ValueError, match="TP ranks must agree"):
            session.begin(
                layout=mismatched,
                storage_plan=storage,
                excluded_tensors=(),
                resource_ledger=module.EagleTTTResourceLedger(limit_bytes=1 << 20),
            )
        assert core.state is None
        assert core.plan is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Gloo distributed support is required",
)
def test_tp2_session_rejects_rank_mismatched_layout_before_forward(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "eagle-ttt-layout-tp2-init"
    mp.spawn(
        _tp_layout_agreement_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )


def _cuda_nccl_session_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _load_symbols()
        module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]
        core = module.EagleTTTCoreAttention(
            config=SimpleNamespace(context_parallel_size=1),
            layer_number=1,
            attn_mask_type=None,
            attention_type="self",
            cp_comm_type=None,
            softmax_scale=None,
            pg_collection=SimpleNamespace(tp=dist.group.WORLD),
        )
        session = module.MCoreEagleTTTSession(_SessionModel(core).cuda())
        sequence = 64
        storage = module.EagleTTTStoragePlan(
            batch_size=1,
            kv_heads=2,
            sequence_length=sequence,
            head_dim=16,
            dtype=torch.bfloat16,
            pass_count=2,
            max_passes=8,
            activation_budget_bytes=1 << 30,
            layer_count=1,
            hidden_size=32,
            rope_dim=16,
        )
        layout = module.EagleTTTSequenceLayout.from_cu_seqlens(
            cu_seqlens=torch.tensor(
                [0, 32, 60],
                dtype=torch.int32,
                device=device,
            ),
            sequence_length=sequence,
        )
        torch.manual_seed(2200)
        hidden = torch.randn(
            sequence,
            1,
            32,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        embeddings = torch.randn_like(hidden)
        ledger = module.EagleTTTResourceLedger(limit_bytes=1 << 30)
        session.begin(
            layout=layout,
            storage_plan=storage,
            excluded_tensors=(hidden, embeddings),
            resource_ledger=ledger,
        )
        current = hidden
        output = hidden
        for pass_index in range(2):
            plan = module.EagleTTTAttentionPlan(
                pass_index=pass_index,
                pass_count=2,
                max_passes=8,
                sequence_length=sequence,
            )
            output, branch = session(
                embeddings=embeddings,
                hidden_states=current,
                plan=plan,
                rope_positions=torch.arange(sequence, device=device),
            )
            current = (
                torch.cat(
                    (torch.zeros_like(branch[:1]), branch[:-1]),
                    dim=0,
                )
                .detach()
                .requires_grad_(True)
            )
        output.float().square().mean().backward()
        gathered = [torch.empty_like(output) for _ in range(world_size)]
        dist.all_gather(gathered, output)
        for peer_output in gathered:
            torch.testing.assert_close(peer_output, output, atol=5e-2, rtol=5e-2)
        assert output.isfinite().all()
        assert len(session.block_masks) == 2
        session.reset()
        assert core.state is None
        assert session.block_masks == []
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available()
    or torch.cuda.device_count() < 2,
    reason="two CUDA devices and NCCL are required",
)
def test_cuda_nccl_tp2_session_collectives_and_flex_attention(
    tmp_path: Path,
) -> None:
    init_file = tmp_path / "eagle-ttt-cuda-nccl-tp2-init"
    mp.spawn(
        _cuda_nccl_session_worker,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )
