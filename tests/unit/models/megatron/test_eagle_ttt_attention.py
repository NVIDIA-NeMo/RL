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
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch


def _load_module() -> ModuleType:
    """Load the pure TTT module without importing optional Megatron dependencies."""
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[4] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.eagle_ttt")


def _load_symbols():
    module = _load_module()
    return (
        module.EagleTTTAttentionPlan,
        module.EagleTTTState,
        module.EagleTTTStoragePlan,
        module.eagle_ttt_attention,
    )


def _dense_reference(
    *,
    query: torch.Tensor,
    trunk_key: torch.Tensor,
    trunk_value: torch.Tensor,
    branch_keys: tuple[torch.Tensor, ...],
    branch_values: tuple[torch.Tensor, ...],
    pass_index: int,
) -> torch.Tensor:
    query_heads = query.shape[1]
    repeats = query_heads // trunk_key.shape[1]
    key = torch.cat((trunk_key, *branch_keys), dim=2).repeat_interleave(repeats, dim=1)
    value = torch.cat((trunk_value, *branch_values), dim=2).repeat_interleave(
        repeats, dim=1
    )
    sequence = query.shape[2]
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * query.shape[-1] ** -0.5
    query_positions = torch.arange(sequence)[:, None]
    key_positions = torch.arange(sequence)[None, :]
    trunk_visible = key_positions <= query_positions - pass_index
    branches_visible = tuple(
        key_positions == query_positions - (pass_index - branch_index - 1)
        for branch_index in range(len(branch_keys))
    )
    visible = torch.cat((trunk_visible, *branches_visible), dim=1)
    probabilities = scores.masked_fill(~visible[None, None], float("-inf")).softmax(
        dim=-1
    )
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, value)


def _packed_dense_reference(
    *,
    query: torch.Tensor,
    trunk_key: torch.Tensor,
    trunk_value: torch.Tensor,
    branch_keys: tuple[torch.Tensor, ...],
    branch_values: tuple[torch.Tensor, ...],
    pass_index: int,
    valid_tokens: torch.Tensor,
    document_ids: torch.Tensor,
) -> torch.Tensor:
    query_heads = query.shape[1]
    repeats = query_heads // trunk_key.shape[1]
    key = torch.cat((trunk_key, *branch_keys), dim=2).repeat_interleave(repeats, dim=1)
    value = torch.cat((trunk_value, *branch_values), dim=2).repeat_interleave(
        repeats, dim=1
    )
    batch, _, sequence, _ = query.shape
    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * query.shape[-1] ** -0.5
    query_positions = torch.arange(sequence, device=query.device)[:, None]
    key_positions = torch.arange(sequence, device=query.device)[None, :]
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    valid_pair = valid_tokens[:, :, None] & valid_tokens[:, None, :]
    trunk_visible = (
        (key_positions <= query_positions - pass_index)[None]
        & same_document
        & valid_pair
    )
    branches_visible = []
    for branch_index in range(len(branch_keys)):
        offset = pass_index - branch_index - 1
        aligned_key = key_positions == query_positions - offset
        branches_visible.append(aligned_key[None] & same_document & valid_pair)
    visible = torch.cat((trunk_visible, *branches_visible), dim=-1)
    masked_scores = scores.masked_fill(~visible[:, None], float("-inf"))
    safe_scores = torch.where(visible.any(dim=-1)[:, None, :, None], masked_scores, 0.0)
    probabilities = safe_scores.softmax(dim=-1)
    probabilities = probabilities.masked_fill(
        ~visible.any(dim=-1)[:, None, :, None], 0.0
    )
    output = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
    assert output.shape[:3] == (batch, query_heads, sequence)
    return output


@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
@pytest.mark.parametrize("query_heads,kv_heads", [(4, 4), (4, 2)])
def test_attention_matches_dense_output_and_every_input_gradient(
    pass_count: int,
    query_heads: int,
    kv_heads: int,
) -> None:
    (
        EagleTTTAttentionPlan,
        EagleTTTState,
        _,
        eagle_ttt_attention,
    ) = _load_symbols()
    torch.manual_seed(31 + pass_count)
    batch, sequence, head_dim = 1, 5, 8
    query = torch.randn(
        batch,
        query_heads,
        sequence,
        head_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    trunk_key = torch.randn(
        batch,
        kv_heads,
        sequence,
        head_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    trunk_value = torch.randn_like(trunk_key, requires_grad=True)
    branch_keys = tuple(
        torch.randn_like(trunk_key, requires_grad=True) for _ in range(pass_count - 1)
    )
    branch_values = tuple(
        torch.randn_like(trunk_value, requires_grad=True) for _ in range(pass_count - 1)
    )
    state = EagleTTTState.from_trunk(
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for branch_key, branch_value in zip(branch_keys, branch_values, strict=True):
        state = state.append_branch(branch_key=branch_key, branch_value=branch_value)
    plan = EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )

    actual = eagle_ttt_attention(query=query, state=state, plan=plan)
    expected = _dense_reference(
        query=query,
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        branch_keys=branch_keys,
        branch_values=branch_values,
        pass_index=plan.pass_index,
    )
    torch.testing.assert_close(actual, expected)

    upstream = torch.randn_like(actual)
    differentiable_inputs = (
        query,
        trunk_key,
        trunk_value,
        *branch_keys,
        *branch_values,
    )
    actual_gradients = torch.autograd.grad(
        actual, differentiable_inputs, upstream, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, differentiable_inputs, upstream)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("sequence_length", [8_192, 32_768, 262_144])
@pytest.mark.parametrize("pass_count", [1, 2, 4, 8])
def test_rope_positions_and_retained_storage_scale_linearly(
    sequence_length: int,
    pass_count: int,
) -> None:
    EagleTTTAttentionPlan, _, EagleTTTStoragePlan, _ = _load_symbols()
    plan = EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence_length,
    )
    positions = plan.rope_positions()
    assert positions.dtype == torch.int64
    assert positions.shape == (sequence_length,)
    assert positions[0].item() == 0
    assert positions[-1].item() == sequence_length - 1

    storage = EagleTTTStoragePlan(
        batch_size=2,
        kv_heads=4,
        sequence_length=sequence_length,
        head_dim=16,
        layer_count=3,
        hidden_size=32,
        rope_dim=16,
        dtype=torch.bfloat16,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 40,
    )
    kv_bytes_per_pass = 3 * 2 * 2 * 4 * sequence_length * 16 * 2
    hidden_bytes_per_pass = 2 * sequence_length * 32 * 2
    block_count = (sequence_length + 127) // 128
    mask_bytes = 2 * pass_count * 16 * (block_count + block_count * block_count)
    loss_bytes = sum(
        2 * max(sequence_length - pass_index - 1, 0) * 24
        for pass_index in range(pass_count)
    )
    assert storage.kv_bytes == pass_count * kv_bytes_per_pass
    assert storage.hidden_bytes == pass_count * hidden_bytes_per_pass
    assert storage.rope_bytes == sequence_length * 16 * 2
    assert storage.mask_bytes == mask_bytes
    assert storage.mask_bytes < pass_count * sequence_length * sequence_length
    assert storage.loss_bytes == loss_bytes
    assert storage.retained_bytes == (
        storage.kv_bytes
        + storage.hidden_bytes
        + storage.rope_bytes
        + storage.mask_bytes
        + storage.loss_bytes
    )
    assert all(
        not isinstance(getattr(plan, field.name), torch.Tensor)
        for field in fields(plan)
    )


def test_mask_bound_covers_unique_multibatch_block_mask_storage() -> None:
    module = _load_module()
    batch_size = 8
    sequence_length = 1_024
    layout = module.EagleTTTSequenceLayout.unpacked(
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    block_mask = module._layout_block_mask(layout=layout, pass_index=0)
    storage = module.EagleTTTStoragePlan(
        batch_size=batch_size,
        kv_heads=1,
        sequence_length=sequence_length,
        head_dim=1,
        dtype=torch.bfloat16,
        pass_count=1,
        max_passes=1,
        activation_budget_bytes=1 << 20,
    )
    mask_storages = {
        (
            tensor.untyped_storage().data_ptr(),
            tensor.untyped_storage().nbytes(),
        ): tensor.untyped_storage().nbytes()
        for tensor in (
            getattr(block_mask, name, None)
            for name in (
                "kv_num_blocks",
                "kv_indices",
                "full_kv_num_blocks",
                "full_kv_indices",
                "q_num_blocks",
                "q_indices",
                "full_q_num_blocks",
                "full_q_indices",
            )
        )
        if isinstance(tensor, torch.Tensor)
    }

    assert storage.mask_bytes >= sum(mask_storages.values()) > 0


def test_flex_attention_is_compiled_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "nemo_rl.models.megatron.draft"
    package = ModuleType(package_name)
    package.__path__ = [
        str(Path(__file__).parents[4] / "nemo_rl/models/megatron/draft")
    ]
    sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.eagle_ttt")
    module._compiled_flex_attention.cache_clear()
    compile_calls: list[tuple[Any, bool, bool]] = []

    def compile_stub(function: Any, *, dynamic: bool, fullgraph: bool) -> Any:
        compile_calls.append((function, dynamic, fullgraph))
        return function

    monkeypatch.setattr(torch, "compile", compile_stub)
    compiled = module._compiled_flex_attention()

    assert callable(compiled)
    assert len(compile_calls) == 1
    assert compile_calls[0][1] is True
    assert compile_calls[0][2] is True
    module._compiled_flex_attention.cache_clear()


def test_budget_and_maximum_are_rejected_before_state_construction() -> None:
    _, EagleTTTState, EagleTTTStoragePlan, _ = _load_symbols()
    with pytest.raises(ValueError, match="configured maximum"):
        EagleTTTStoragePlan(
            batch_size=1,
            kv_heads=1,
            sequence_length=32_768,
            head_dim=8,
            layer_count=1,
            hidden_size=8,
            rope_dim=8,
            dtype=torch.bfloat16,
            pass_count=9,
            max_passes=8,
            activation_budget_bytes=1 << 30,
        )
    with pytest.raises(ValueError, match="activation budget"):
        EagleTTTStoragePlan(
            batch_size=1,
            kv_heads=1,
            sequence_length=262_144,
            head_dim=128,
            layer_count=32,
            hidden_size=4096,
            rope_dim=128,
            dtype=torch.bfloat16,
            pass_count=8,
            max_passes=8,
            activation_budget_bytes=1,
        )

    key = torch.randn(1, 1, 2, 4)
    with pytest.raises(ValueError, match="configured maximum"):
        EagleTTTState.from_trunk(
            trunk_key=key,
            trunk_value=key,
            pass_count=9,
            max_passes=8,
            activation_budget_bytes=1 << 20,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_backward_does_not_save_sequence_square_mask_or_packed_pass_kv() -> None:
    (
        EagleTTTAttentionPlan,
        EagleTTTState,
        _,
        eagle_ttt_attention,
    ) = _load_symbols()
    sequence, head_dim, pass_count = 128, 16, 4
    query = torch.randn(
        1,
        2,
        sequence,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    state = EagleTTTState.from_trunk(
        trunk_key=key,
        trunk_value=value,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for _ in range(pass_count - 1):
        state = state.append_branch(
            branch_key=torch.randn_like(key, requires_grad=True),
            branch_value=torch.randn_like(value, requires_grad=True),
        )
    plan = EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )
    saved_shapes: list[torch.Size] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        eagle_ttt_attention(query=query, state=state, plan=plan).sum().backward()

    assert torch.Size((sequence, sequence)) not in saved_shapes
    assert not any(
        len(shape) == 4 and shape[-2] == sequence * pass_count for shape in saved_shapes
    )


@pytest.mark.parametrize("pass_index", [0, 1, 2, 3, 7])
def test_visibility_matches_pinned_modelopt_pass_geometry(pass_index: int) -> None:
    EagleTTTAttentionPlan, _, _, _ = _load_symbols()
    sequence = 9
    plan = EagleTTTAttentionPlan(
        pass_index=pass_index,
        pass_count=pass_index + 1,
        max_passes=8,
        sequence_length=sequence,
    )
    visible = plan.dense_visibility_mask()
    query_positions = torch.arange(sequence)[:, None]
    key_positions = torch.arange(sequence)[None, :]
    expected_trunk = key_positions <= query_positions - pass_index
    expected_branches = tuple(
        key_positions == query_positions - (pass_index - branch_index - 1)
        for branch_index in range(pass_index)
    )
    torch.testing.assert_close(
        visible,
        torch.cat((expected_trunk, *expected_branches), dim=1),
    )


@pytest.mark.parametrize("sequence", [64, 96])
def test_pass_zero_causal_predicate_keeps_every_query_row_safe(
    sequence: int,
) -> None:
    _load_symbols()
    module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]
    query_positions = torch.arange(sequence)[:, None]
    key_positions = torch.arange(sequence)[None, :]

    visible = module._causal_mask(
        torch.tensor(0),
        torch.tensor(0),
        query_positions,
        key_positions,
        pass_index=0,
    )

    torch.testing.assert_close(
        visible.sum(dim=-1),
        torch.arange(1, sequence + 1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fullgraph_flex_supports_two_dynamic_sequence_lengths() -> None:
    module = importlib.import_module("nemo_rl.models.megatron.draft.eagle_ttt")
    module._compiled_flex_attention.cache_clear()
    for sequence in (64, 96):
        query = torch.randn(
            1,
            2,
            sequence,
            16,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn_like(query, requires_grad=True)
        state = module.EagleTTTState.from_trunk(
            trunk_key=key,
            trunk_value=value,
            pass_count=1,
            max_passes=8,
            activation_budget_bytes=1 << 30,
        )
        plan = module.EagleTTTAttentionPlan(
            pass_index=0,
            pass_count=1,
            max_passes=8,
            sequence_length=sequence,
        )
        output = module.eagle_ttt_attention(query=query, state=state, plan=plan)
        assert output.isfinite().all()
        output.float().square().mean().backward()


def test_sequence_layout_from_cu_seqlens_is_linear_and_marks_padding() -> None:
    _load_symbols()
    module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]
    sequence = 262_144

    layout = module.EagleTTTSequenceLayout.from_cu_seqlens(
        cu_seqlens=torch.tensor([0, 131_072, 200_000], dtype=torch.int32),
        sequence_length=sequence,
    )

    assert layout.valid_tokens.shape == (1, sequence)
    assert layout.document_ids.shape == (1, sequence)
    assert layout.valid_tokens.numel() + layout.document_ids.numel() == 2 * sequence
    assert layout.valid_tokens[:, :200_000].all()
    assert not layout.valid_tokens[:, 200_000:].any()
    assert (layout.document_ids[:, :131_072] == 0).all()
    assert (layout.document_ids[:, 131_072:200_000] == 1).all()
    assert (layout.document_ids[:, 200_000:] == -1).all()


@pytest.mark.parametrize(
    "cu_seqlens,sequence_length,error",
    [
        ([1, 4], 4, "start at zero"),
        ([0, 4, 3], 4, "monotonic"),
        ([0, 5], 4, "sequence length"),
    ],
)
def test_sequence_layout_rejects_invalid_cumulative_lengths(
    cu_seqlens: list[int],
    sequence_length: int,
    error: str,
) -> None:
    _load_symbols()
    module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]

    with pytest.raises(ValueError, match=error):
        module.EagleTTTSequenceLayout.from_cu_seqlens(
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
            sequence_length=sequence_length,
        )


def test_sequence_layout_rejects_invalid_token_document_contract() -> None:
    _load_symbols()
    module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]

    with pytest.raises(ValueError, match="same shape"):
        module.EagleTTTSequenceLayout(
            valid_tokens=torch.ones(1, 4, dtype=torch.bool),
            document_ids=torch.zeros(1, 3, dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="sentinel"):
        module.EagleTTTSequenceLayout(
            valid_tokens=torch.tensor([[True, False]]),
            document_ids=torch.tensor([[0, 0]]),
        )


@pytest.mark.parametrize("pass_index", [0, 1, 2, 4])
def test_packed_padding_visibility_matches_dense_output_and_gradients(
    pass_index: int,
) -> None:
    _load_symbols()
    module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]
    torch.manual_seed(700 + pass_index)
    batch, heads, sequence, head_dim = 1, 2, 12, 4
    query = torch.randn(
        batch, heads, sequence, head_dim, dtype=torch.float64, requires_grad=True
    )
    trunk_key = torch.randn_like(query, requires_grad=True)
    trunk_value = torch.randn_like(query, requires_grad=True)
    branch_keys = tuple(
        torch.randn_like(query, requires_grad=True) for _ in range(pass_index)
    )
    branch_values = tuple(
        torch.randn_like(query, requires_grad=True) for _ in range(pass_index)
    )
    layout = module.EagleTTTSequenceLayout.from_cu_seqlens(
        cu_seqlens=torch.tensor([0, 6, 11], dtype=torch.int32),
        sequence_length=sequence,
    )
    state = module.EagleTTTState.from_trunk(
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        pass_count=pass_index + 1,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for branch_key, branch_value in zip(branch_keys, branch_values, strict=True):
        state = state.append_branch(
            branch_key=branch_key,
            branch_value=branch_value,
        )
    plan = module.EagleTTTAttentionPlan(
        pass_index=pass_index,
        pass_count=pass_index + 1,
        max_passes=8,
        sequence_length=sequence,
    )

    actual = module.eagle_ttt_attention(
        query=query,
        state=state,
        plan=plan,
        layout=layout,
    )
    expected = _packed_dense_reference(
        query=query,
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        branch_keys=branch_keys,
        branch_values=branch_values,
        pass_index=pass_index,
        valid_tokens=layout.valid_tokens,
        document_ids=layout.document_ids,
    )
    torch.testing.assert_close(actual, expected)

    document_b = torch.zeros_like(actual)
    document_b[:, :, 6:11] = torch.randn_like(document_b[:, :, 6:11])
    inputs = (query, trunk_key, trunk_value, *branch_keys, *branch_values)
    actual_gradients = torch.autograd.grad(
        actual, inputs, document_b, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, inputs, document_b)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert not actual_gradient[:, :, :6].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("pass_count", [1, 2, 4])
def test_cuda_packed_padding_output_and_gradients_match_dense_oracle(
    pass_count: int,
) -> None:
    _load_symbols()
    module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]
    torch.manual_seed(900 + pass_count)
    batch, heads, sequence, head_dim = 1, 2, 64, 16
    tensors = tuple(
        torch.randn(
            batch,
            heads,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3 + 2 * (pass_count - 1))
    )
    query, trunk_key, trunk_value, *branches = tensors
    branch_keys = tuple(branches[: pass_count - 1])
    branch_values = tuple(branches[pass_count - 1 :])
    layout = module.EagleTTTSequenceLayout.from_cu_seqlens(
        cu_seqlens=torch.tensor([0, 32, 60], dtype=torch.int32, device="cuda"),
        sequence_length=sequence,
    )
    state = module.EagleTTTState.from_trunk(
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for branch_key, branch_value in zip(branch_keys, branch_values, strict=True):
        state = state.append_branch(
            branch_key=branch_key,
            branch_value=branch_value,
        )
    plan = module.EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )

    actual = module.eagle_ttt_attention(
        query=query,
        state=state,
        plan=plan,
        layout=layout,
    )
    expected = _packed_dense_reference(
        query=query,
        trunk_key=trunk_key,
        trunk_value=trunk_value,
        branch_keys=branch_keys,
        branch_values=branch_values,
        pass_index=plan.pass_index,
        valid_tokens=layout.valid_tokens,
        document_ids=layout.document_ids,
    )
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)
    assert actual.isfinite().all()

    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, tensors, upstream)
    expected_gradients = torch.autograd.grad(expected, tensors, upstream)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=8e-2,
            rtol=8e-2,
        )
        assert actual_gradient.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("sequence", [8_192, 32_768])
@pytest.mark.parametrize("pass_count", [1, 4])
def test_cuda_long_context_multi_pass_has_no_saved_sequence_square_tensor(
    sequence: int, pass_count: int
) -> None:
    _load_symbols()
    module = sys.modules["nemo_rl.models.megatron.draft.eagle_ttt"]
    query = torch.randn(
        1,
        2,
        sequence,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    branch_keys = tuple(
        torch.randn_like(query, requires_grad=True) for _ in range(pass_count - 1)
    )
    branch_values = tuple(
        torch.randn_like(query, requires_grad=True) for _ in range(pass_count - 1)
    )
    layout = module.EagleTTTSequenceLayout.from_cu_seqlens(
        cu_seqlens=torch.tensor(
            [0, sequence // 2, sequence],
            dtype=torch.int32,
            device="cuda",
        ),
        sequence_length=sequence,
    )
    state = module.EagleTTTState.from_trunk(
        trunk_key=key,
        trunk_value=value,
        pass_count=pass_count,
        max_passes=8,
        activation_budget_bytes=1 << 30,
    )
    for branch_key, branch_value in zip(branch_keys, branch_values, strict=True):
        state = state.append_branch(
            branch_key=branch_key,
            branch_value=branch_value,
        )
    plan = module.EagleTTTAttentionPlan(
        pass_index=pass_count - 1,
        pass_count=pass_count,
        max_passes=8,
        sequence_length=sequence,
    )
    saved_shapes: list[torch.Size] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tensor.shape)
        return tensor

    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = module.eagle_ttt_attention(
            query=query,
            state=state,
            plan=plan,
            layout=layout,
        )
        output.float().square().mean().backward()
    incremental_peak = torch.cuda.max_memory_allocated() - baseline

    assert torch.Size((sequence, sequence)) not in saved_shapes
    assert incremental_peak < 1 << 30
