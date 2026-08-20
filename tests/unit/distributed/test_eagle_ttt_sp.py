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

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


_TP_SIZE = 2
_GLOBAL_SEQUENCE = 16
_LOCAL_SEQUENCE = _GLOBAL_SEQUENCE // _TP_SIZE
_HIDDEN_SIZE = 64
_HEAD_DIM = 16


def _load_eagle_ttt_module() -> ModuleType:
    import importlib

    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[3] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.eagle_ttt")


class _ZeroRotary(torch.nn.Module):
    """Record requested geometry while keeping QKV/core geometry independently testable."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_lengths: list[int] = []

    def forward(self, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.requested_lengths.append(sequence_length)
        frequencies = torch.zeros(
            _GLOBAL_SEQUENCE,
            1,
            1,
            _HEAD_DIM,
            device=torch.cuda.current_device(),
        )
        return frequencies, frequencies


class _RealMCoreEagleModule(torch.nn.Module):
    def __init__(
        self,
        *,
        sequence_parallel: bool,
        eagle_ttt_module: ModuleType,
    ) -> None:
        super().__init__()
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from megatron.core.transformer.attention import (
            SelfAttention,
            SelfAttentionSubmodules,
        )
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=1,
            hidden_size=_HIDDEN_SIZE,
            num_attention_heads=_HIDDEN_SIZE // _HEAD_DIM,
            num_query_groups=_HIDDEN_SIZE // _HEAD_DIM,
            ffn_hidden_size=64,
            kv_channels=_HEAD_DIM,
            tensor_model_parallel_size=_TP_SIZE,
            sequence_parallel=sequence_parallel,
            attention_dropout=0.0,
            add_bias_linear=False,
            params_dtype=torch.float32,
            gradient_accumulation_fusion=False,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        assert pg_collection.tp is parallel_state.get_tensor_model_parallel_group()
        self.self_attention = SelfAttention(
            config=config,
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=eagle_ttt_module.EagleTTTCoreAttention,
                linear_proj=RowParallelLinear,
            ),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )
        self.rotary_pos_emb = _ZeroRotary()
        self.qkv_sequence_lengths: list[int] = []

        def record_qkv(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor | None],
        ) -> None:
            projected, _bias = output
            self.qkv_sequence_lengths.append(projected.shape[0])

        self.self_attention.linear_qkv.register_forward_hook(record_qkv)

    def forward(
        self,
        *,
        embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        rotary_pos_emb: torch.Tensor | None = None,
        packed_seq_params: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, bias = self.self_attention(
            hidden_states + embeddings,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        if bias is not None:
            output = output + bias
        return output, output


def _set_deterministic_weights(model: torch.nn.Module, *, rank: int) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters(), start=1):
            values = torch.arange(
                parameter.numel(),
                device=parameter.device,
                dtype=parameter.dtype,
            ).reshape_as(parameter)
            parameter.copy_(
                torch.sin(values * 0.017 + rank * 0.31 + parameter_index * 0.13) * 0.1
            )


def _global_layout(
    eagle_ttt_module: ModuleType,
    *,
    kind: str,
    device: torch.device,
) -> tuple[Any, Any | None]:
    if kind == "packed":
        cu_seqlens = torch.tensor([0, 5, 9, 16], dtype=torch.int32, device=device)
        layout = eagle_ttt_module.EagleTTTSequenceLayout.from_cu_seqlens(
            cu_seqlens=cu_seqlens,
            sequence_length=_GLOBAL_SEQUENCE,
        )
        from megatron.core.packed_seq_params import PackedSeqParams

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=7,
            max_seqlen_kv=7,
            total_tokens=_GLOBAL_SEQUENCE,
        )
        return layout, packed_seq_params

    valid_tokens = torch.tensor(
        [[True] * 6 + [False] * 2 + [True] * 6 + [False] * 2],
        device=device,
    )
    document_ids = torch.where(
        valid_tokens,
        torch.zeros((), dtype=torch.int64, device=device),
        torch.full((), -1, dtype=torch.int64, device=device),
    )
    return (
        eagle_ttt_module.EagleTTTSequenceLayout(
            valid_tokens=valid_tokens,
            document_ids=document_ids,
        ),
        None,
    )


def _local_layout(
    eagle_ttt_module: ModuleType,
    global_layout: Any,
    *,
    rank: int,
) -> Any:
    start = rank * _LOCAL_SEQUENCE
    stop = start + _LOCAL_SEQUENCE
    return eagle_ttt_module.EagleTTTSequenceLayout(
        valid_tokens=global_layout.valid_tokens[:, start:stop].contiguous(),
        document_ids=global_layout.document_ids[:, start:stop].contiguous(),
    )


def _storage_plan(eagle_ttt_module: ModuleType, *, sequence_length: int) -> Any:
    return eagle_ttt_module.EagleTTTStoragePlan(
        batch_size=1,
        kv_heads=2,
        sequence_length=sequence_length,
        head_dim=_HEAD_DIM,
        dtype=torch.float32,
        pass_count=2,
        max_passes=8,
        activation_budget_bytes=1 << 30,
        layer_count=1,
        hidden_size=_HIDDEN_SIZE,
        rope_dim=_HEAD_DIM,
    )


def _plan(
    eagle_ttt_module: ModuleType,
    *,
    pass_index: int,
    sequence_length: int,
) -> Any:
    return eagle_ttt_module.EagleTTTAttentionPlan(
        pass_index=pass_index,
        pass_count=2,
        max_passes=8,
        sequence_length=sequence_length,
    )


def _inputs(*, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(20260819)
    return tuple(
        torch.randn(
            _GLOBAL_SEQUENCE,
            1,
            _HIDDEN_SIZE,
            generator=generator,
            device=device,
        )
        for _ in range(4)
    )


def _run_two_passes(
    *,
    eagle_ttt_module: ModuleType,
    model: _RealMCoreEagleModule,
    layout: Any,
    packed_seq_params: Any | None,
    inputs: tuple[torch.Tensor, ...],
    sequence_length: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    trunk, draft, embeddings, upstream = inputs
    differentiable = tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in (trunk, draft, embeddings)
    )
    trunk, draft, embeddings = differentiable
    session = eagle_ttt_module.MCoreEagleTTTSession(model)
    ledger = eagle_ttt_module.EagleTTTResourceLedger(limit_bytes=1 << 30)
    session.begin(
        layout=layout,
        storage_plan=_storage_plan(
            eagle_ttt_module,
            sequence_length=sequence_length,
        ),
        excluded_tensors=differentiable,
        resource_ledger=ledger,
        packed_seq_params=packed_seq_params,
    )
    try:
        session(
            embeddings=embeddings,
            hidden_states=trunk,
            plan=_plan(
                eagle_ttt_module,
                pass_index=0,
                sequence_length=sequence_length,
            ),
            rope_positions=torch.arange(sequence_length, device=trunk.device),
        )
        output, _ = session(
            embeddings=embeddings,
            hidden_states=draft,
            plan=_plan(
                eagle_ttt_module,
                pass_index=1,
                sequence_length=sequence_length,
            ),
            rope_positions=torch.arange(sequence_length, device=trunk.device),
        )
        loss = (output * upstream).sum()
        loss.backward()
        input_gradients = tuple(
            tensor.grad.detach().clone() for tensor in differentiable
        )
        parameter_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        return output.detach().clone(), input_gradients, parameter_gradients
    finally:
        session.reset()
        assert all(layer.state is None for layer in session.layers)
        assert session.block_masks == []


def _real_mcore_sp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    layout_kind: str,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(20260819)
        eagle_ttt_module = _load_eagle_ttt_module()
        dense_model = _RealMCoreEagleModule(
            sequence_parallel=False,
            eagle_ttt_module=eagle_ttt_module,
        ).to(device)
        sp_model = _RealMCoreEagleModule(
            sequence_parallel=True,
            eagle_ttt_module=eagle_ttt_module,
        ).to(device)
        _set_deterministic_weights(dense_model, rank=rank)
        sp_model.load_state_dict(dense_model.state_dict())

        global_layout, packed_seq_params = _global_layout(
            eagle_ttt_module,
            kind=layout_kind,
            device=device,
        )
        local_layout = _local_layout(
            eagle_ttt_module,
            global_layout,
            rank=rank,
        )
        full_inputs = _inputs(device=device)
        start = rank * _LOCAL_SEQUENCE
        stop = start + _LOCAL_SEQUENCE
        local_inputs = tuple(
            tensor[start:stop].contiguous() for tensor in full_inputs[:-1]
        ) + (full_inputs[-1][start:stop].contiguous(),)

        expected_output, expected_input_gradients, expected_parameter_gradients = (
            _run_two_passes(
                eagle_ttt_module=eagle_ttt_module,
                model=dense_model,
                layout=global_layout,
                packed_seq_params=packed_seq_params,
                inputs=full_inputs,
                sequence_length=_GLOBAL_SEQUENCE,
            )
        )
        actual_output, actual_input_gradients, actual_parameter_gradients = (
            _run_two_passes(
                eagle_ttt_module=eagle_ttt_module,
                model=sp_model,
                layout=local_layout,
                packed_seq_params=packed_seq_params,
                inputs=local_inputs,
                sequence_length=_LOCAL_SEQUENCE,
            )
        )

        torch.testing.assert_close(
            actual_output,
            expected_output[start:stop],
            atol=2e-5,
            rtol=2e-5,
        )
        for actual, expected in zip(
            actual_input_gradients,
            expected_input_gradients,
            strict=True,
        ):
            torch.testing.assert_close(
                actual,
                expected[start:stop],
                atol=3e-5,
                rtol=3e-5,
            )
            assert torch.count_nonzero(actual) > 0
        assert actual_parameter_gradients.keys() == expected_parameter_gradients.keys()
        for name, actual in actual_parameter_gradients.items():
            torch.testing.assert_close(
                actual,
                expected_parameter_gradients[name],
                atol=4e-5,
                rtol=4e-5,
            )
            assert torch.count_nonzero(actual) > 0

        assert dense_model.qkv_sequence_lengths == [_GLOBAL_SEQUENCE] * 2
        assert sp_model.qkv_sequence_lengths == [_GLOBAL_SEQUENCE] * 2
        assert dense_model.rotary_pos_emb.requested_lengths == [_GLOBAL_SEQUENCE]
        assert sp_model.rotary_pos_emb.requested_lengths == [_GLOBAL_SEQUENCE]
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _sp_packed_mismatch_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(20260819)
        eagle_ttt_module = _load_eagle_ttt_module()
        model = _RealMCoreEagleModule(
            sequence_parallel=True,
            eagle_ttt_module=eagle_ttt_module,
        ).to(device)
        global_layout, packed_seq_params = _global_layout(
            eagle_ttt_module,
            kind="packed",
            device=device,
        )
        local_layout = _local_layout(
            eagle_ttt_module,
            global_layout,
            rank=rank,
        )
        if rank == 1:
            document_ids = local_layout.document_ids.clone()
            document_ids[0, -1] = 99
            local_layout = eagle_ttt_module.EagleTTTSequenceLayout(
                valid_tokens=local_layout.valid_tokens,
                document_ids=document_ids,
            )
        session = eagle_ttt_module.MCoreEagleTTTSession(model)
        with pytest.raises(
            ValueError,
            match="packed sequence parameters do not match",
        ):
            session.begin(
                layout=local_layout,
                storage_plan=_storage_plan(
                    eagle_ttt_module,
                    sequence_length=_LOCAL_SEQUENCE,
                ),
                excluded_tensors=(),
                resource_ledger=eagle_ttt_module.EagleTTTResourceLedger(
                    limit_bytes=1 << 30
                ),
                packed_seq_params=packed_seq_params,
            )
        assert session.layout is None
        assert all(
            layer.state is None and layer.layout is None for layer in session.layers
        )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _sp_packed_params_mismatch_worker(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(20260819)
        eagle_ttt_module = _load_eagle_ttt_module()
        model = _RealMCoreEagleModule(
            sequence_parallel=True,
            eagle_ttt_module=eagle_ttt_module,
        ).to(device)
        global_layout, packed_seq_params = _global_layout(
            eagle_ttt_module,
            kind="packed",
            device=device,
        )
        local_layout = _local_layout(
            eagle_ttt_module,
            global_layout,
            rank=rank,
        )
        if rank == 1:
            mismatched_cu_seqlens = torch.tensor(
                [0, 4, 9, 16],
                dtype=torch.int32,
                device=device,
            )
            packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=mismatched_cu_seqlens,
                cu_seqlens_kv=mismatched_cu_seqlens,
                max_seqlen_q=7,
                max_seqlen_kv=7,
                total_tokens=_GLOBAL_SEQUENCE,
            )
        session = eagle_ttt_module.MCoreEagleTTTSession(model)
        error = ""
        try:
            session.begin(
                layout=local_layout,
                storage_plan=_storage_plan(
                    eagle_ttt_module,
                    sequence_length=_LOCAL_SEQUENCE,
                ),
                excluded_tensors=(),
                resource_ledger=eagle_ttt_module.EagleTTTResourceLedger(
                    limit_bytes=1 << 30
                ),
                packed_seq_params=packed_seq_params,
            )
        except ValueError as exc:
            error = str(exc)
        error_flag = torch.tensor([bool(error)], dtype=torch.int32, device=device)
        gathered_flags = [torch.empty_like(error_flag) for _ in range(world_size)]
        dist.all_gather(gathered_flags, error_flag)
        assert [flag.item() for flag in gathered_flags] == [1] * world_size
        assert error == "TP ranks must agree on EAGLE TTT packed sequence parameters"
        session.reset()
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_nccl_available() or torch.cuda.device_count() < _TP_SIZE,
    reason="two CUDA devices and NCCL are required",
)
@pytest.mark.parametrize("layout_kind", ["padded", "packed"])
def test_real_mcore_tp2_sequence_parallel_matches_global_oracle(
    tmp_path: Path,
    layout_kind: str,
) -> None:
    """Catch local layout/RoPE geometry reaching MCore's global QKV sequence."""
    init_file = tmp_path / f"eagle-ttt-real-mcore-sp-{layout_kind}"
    mp.spawn(
        _real_mcore_sp_worker,
        args=(_TP_SIZE, str(init_file), layout_kind),
        nprocs=_TP_SIZE,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_nccl_available() or torch.cuda.device_count() < _TP_SIZE,
    reason="two CUDA devices and NCCL are required",
)
def test_real_mcore_tp2_sequence_parallel_packed_mismatch_agrees_and_resets(
    tmp_path: Path,
) -> None:
    """Catch rank-local packed metadata accepted as one inconsistent geometry."""
    init_file = tmp_path / "eagle-ttt-real-mcore-sp-mismatch"
    mp.spawn(
        _sp_packed_mismatch_worker,
        args=(_TP_SIZE, str(init_file)),
        nprocs=_TP_SIZE,
        join=True,
    )


@pytest.mark.skipif(
    not torch.distributed.is_nccl_available() or torch.cuda.device_count() < _TP_SIZE,
    reason="two CUDA devices and NCCL are required",
)
def test_real_mcore_tp2_sequence_parallel_packed_params_mismatch_agrees(
    tmp_path: Path,
) -> None:
    """Catch rank-local packed metadata errors before any rank enters MCore."""
    init_file = tmp_path / "eagle-ttt-real-mcore-sp-packed-params-mismatch"
    mp.spawn(
        _sp_packed_params_mismatch_worker,
        args=(_TP_SIZE, str(init_file)),
        nprocs=_TP_SIZE,
        join=True,
    )
