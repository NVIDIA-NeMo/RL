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

from dataclasses import replace

import pytest
import torch
from megatron.core.model_parallel_config import ModelParallelConfig
from torch import Tensor

from nemo_rl.models.megatron.draft.block_plan import (
    DFlashBatchPlan,
    build_dflash_batch_plan,
)
from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig


pytestmark = pytest.mark.mcore


def _tiny_config(*, num_hidden_layers: int = 2) -> DFlashBodyConfig:
    return DFlashBodyConfig(
        hidden_size=8,
        intermediate_size=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=num_hidden_layers,
        num_target_taps=2,
        rope_theta=10_000.0,
    )


def _fp32_parallel_config(
    *,
    tensor_parallel_size: int = 1,
    sequence_parallel: bool = False,
) -> ModelParallelConfig:
    return ModelParallelConfig(
        tensor_model_parallel_size=tensor_parallel_size,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        sequence_parallel=sequence_parallel,
    )


def _plan(token_valid_mask: Tensor, *, gamma: int) -> DFlashBatchPlan:
    sample_ids = torch.arange(token_valid_mask.shape[0], dtype=torch.int64)
    return build_dflash_batch_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=1,
        gamma=gamma,
        optimizer_step=3,
        seed=11,
    )


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _reference_rope(hidden: Tensor, positions: Tensor, *, theta: float) -> Tensor:
    head_dim = hidden.shape[-1]
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=hidden.device, dtype=torch.float32)
            / head_dim
        )
    )
    frequencies = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cosine = angles.cos().to(hidden.dtype).unsqueeze(-2)
    sine = angles.sin().to(hidden.dtype).unsqueeze(-2)
    return hidden * cosine + _rotate_half(hidden) * sine


def _reference_block_attention(
    *,
    query: Tensor,
    trunk_key: Tensor,
    trunk_value: Tensor,
    block_key: Tensor,
    block_value: Tensor,
    plan: DFlashBatchPlan,
) -> Tensor:
    outputs: list[Tensor] = []
    repeats = query.shape[2] // trunk_key.shape[2]
    scale = query.shape[-1] ** -0.5
    for block_index in range(query.shape[0]):
        sample_row = int(plan.sample_rows[block_index])
        anchor_position = int(plan.anchor_positions[block_index])
        visible_trunk = plan.token_valid_mask[sample_row].clone()
        visible_trunk[anchor_position:] = False
        visible_block = plan.slot_valid[block_index]
        key = torch.cat(
            (
                trunk_key[sample_row, visible_trunk],
                block_key[block_index, visible_block],
            ),
            dim=0,
        ).repeat_interleave(repeats, dim=1)
        value = torch.cat(
            (
                trunk_value[sample_row, visible_trunk],
                block_value[block_index, visible_block],
            ),
            dim=0,
        ).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query[block_index], key) * scale
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.einsum("hqk,khd->qhd", probabilities, value)
        outputs.append(
            torch.where(
                plan.slot_valid[block_index, :, None, None],
                output,
                torch.zeros_like(output),
            )
        )
    return torch.stack(outputs)


def _dense_reference(
    body: DFlashBody,
    *,
    target_taps: Tensor,
    block_embeddings: Tensor,
    plan: DFlashBatchPlan,
) -> Tensor:
    config = body.config
    batch_size, sequence_length = target_taps.shape[:2]
    target_hidden = body.hidden_norm(body.fc(target_taps.flatten(start_dim=2)))
    hidden = torch.where(
        plan.slot_valid[..., None],
        block_embeddings,
        torch.zeros_like(block_embeddings),
    )
    trunk_positions = torch.arange(
        sequence_length,
        dtype=torch.int64,
        device=target_taps.device,
    ).expand(batch_size, -1)

    for layer in body.layers:
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        trunk_key = layer.self_attn.k_norm(
            layer.self_attn.k_proj(target_hidden).view(
                batch_size,
                sequence_length,
                config.num_key_value_heads,
                config.head_dim,
            )
        )
        trunk_value = layer.self_attn.v_proj(target_hidden).view(
            batch_size,
            sequence_length,
            config.num_key_value_heads,
            config.head_dim,
        )
        block_query = layer.self_attn.q_norm(
            layer.self_attn.q_proj(normalized).view(
                *normalized.shape[:2],
                config.num_attention_heads,
                config.head_dim,
            )
        )
        block_key = layer.self_attn.k_norm(
            layer.self_attn.k_proj(normalized).view(
                *normalized.shape[:2],
                config.num_key_value_heads,
                config.head_dim,
            )
        )
        block_value = layer.self_attn.v_proj(normalized).view(
            *normalized.shape[:2],
            config.num_key_value_heads,
            config.head_dim,
        )
        trunk_key = _reference_rope(
            trunk_key,
            trunk_positions,
            theta=config.rope_theta,
        )
        block_query = _reference_rope(
            block_query,
            plan.query_positions,
            theta=config.rope_theta,
        )
        block_key = _reference_rope(
            block_key,
            plan.query_positions,
            theta=config.rope_theta,
        )
        attention = _reference_block_attention(
            query=block_query,
            trunk_key=trunk_key,
            trunk_value=trunk_value,
            block_key=block_key,
            block_value=block_value,
            plan=plan,
        )
        hidden = residual + layer.self_attn.o_proj(attention.flatten(start_dim=2))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    output = body.norm(hidden)
    return torch.where(
        plan.slot_valid[..., None],
        output,
        torch.zeros_like(output),
    )


def test_dflash_fp32_forward_and_input_gradients_match_dense_oracle() -> None:
    torch.manual_seed(2026)
    body = DFlashBody(
        _tiny_config(),
        parallel_config=_fp32_parallel_config(),
    )
    plan = _plan(torch.ones((2, 5), dtype=torch.bool), gamma=2)
    plan = replace(
        plan,
        anchor_positions=torch.tensor([2, 2], dtype=torch.int64),
        trunk_lengths=torch.tensor([2, 2], dtype=torch.int64),
        query_positions=torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.int64),
        label_positions=torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.int64),
    )
    target_actual = torch.randn(2, 5, 2, 8, requires_grad=True)
    blocks_actual = torch.randn(2, 3, 8, requires_grad=True)
    target_reference = target_actual.detach().clone().requires_grad_()
    blocks_reference = blocks_actual.detach().clone().requires_grad_()

    actual = body(
        target_taps=target_actual,
        block_embeddings=blocks_actual,
        plan=plan,
    )
    expected = _dense_reference(
        body,
        target_taps=target_reference,
        block_embeddings=blocks_reference,
        plan=plan,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    weights = torch.linspace(0.25, 1.25, actual.numel()).reshape_as(actual)
    actual_gradients = torch.autograd.grad(
        (actual * weights).sum(),
        (target_actual, blocks_actual),
    )
    expected_gradients = torch.autograd.grad(
        (expected * weights).sum(),
        (target_reference, blocks_reference),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.abs().sum() > 0
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=5e-5,
            atol=5e-6,
        )


def test_forward_builds_each_rope_table_once() -> None:
    torch.manual_seed(2027)
    body = DFlashBody(
        _tiny_config(num_hidden_layers=3),
        parallel_config=_fp32_parallel_config(),
    )
    plan = _plan(torch.ones((2, 5), dtype=torch.bool), gamma=2)
    target_taps = torch.randn(2, 5, 2, 8)
    block_embeddings = torch.randn(2, 3, 8)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        body(
            target_taps=target_taps,
            block_embeddings=block_embeddings,
            plan=plan,
        )

    operator_counts = {event.key: event.count for event in profile.key_averages()}
    assert operator_counts["aten::cos"] == 2
    assert operator_counts["aten::sin"] == 2


def test_repeated_block_embeddings_become_slot_distinct_through_rope() -> None:
    torch.manual_seed(7)
    body = DFlashBody(
        _tiny_config(num_hidden_layers=1),
        parallel_config=_fp32_parallel_config(),
    )
    plan = _plan(torch.ones((1, 5), dtype=torch.bool), gamma=3)
    plan = replace(
        plan,
        anchor_positions=torch.tensor([1], dtype=torch.int64),
        trunk_lengths=torch.tensor([1], dtype=torch.int64),
        query_positions=torch.arange(1, 5, dtype=torch.int64)[None],
        label_positions=torch.arange(1, 5, dtype=torch.int64)[None],
    )
    repeated_embedding = torch.randn(1, 1, 8).expand(1, 4, 8).clone()
    output = body(
        target_taps=torch.randn(1, 5, 2, 8),
        block_embeddings=repeated_embedding,
        plan=plan,
    )

    assert not torch.allclose(output[:, 0], output[:, 1])
    assert not torch.allclose(output[:, 1], output[:, 2])


def test_holes_left_padding_and_all_invalid_rows_are_zeroed() -> None:
    torch.manual_seed(19)
    body = DFlashBody(
        _tiny_config(num_hidden_layers=1),
        parallel_config=_fp32_parallel_config(),
    )
    token_valid_mask = torch.tensor(
        [
            [False, False, True, True, True, False],
            [False, False, False, False, False, False],
        ]
    )
    plan = _plan(token_valid_mask, gamma=2)
    target_taps = torch.randn(2, 6, 2, 8, requires_grad=True)
    block_embeddings = torch.randn(2, 3, 8, requires_grad=True)

    output = body(
        target_taps=target_taps,
        block_embeddings=block_embeddings,
        plan=plan,
    )
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~plan.slot_valid]) == 0
    assert torch.count_nonzero(output[1]) == 0
    output.sum().backward()
    assert target_taps.grad is not None
    assert block_embeddings.grad is not None
    assert torch.isfinite(target_taps.grad).all()
    assert torch.isfinite(block_embeddings.grad).all()


@pytest.mark.parametrize("position", [8_192, 32_768, 262_144])
def test_rope_positions_do_not_wrap(position: int) -> None:
    torch.manual_seed(29)
    body = DFlashBody(
        _tiny_config(num_hidden_layers=1),
        parallel_config=_fp32_parallel_config(),
    )
    base_plan = _plan(torch.ones((1, 4), dtype=torch.bool), gamma=2)
    base_plan = replace(
        base_plan,
        anchor_positions=torch.tensor([1], dtype=torch.int64),
        trunk_lengths=torch.tensor([1], dtype=torch.int64),
        query_positions=torch.arange(1, 4, dtype=torch.int64)[None],
        label_positions=torch.arange(1, 4, dtype=torch.int64)[None],
    )
    high_positions = torch.arange(position, position + 3, dtype=torch.int64)[None]
    high_plan = replace(base_plan, query_positions=high_positions)
    target_taps = torch.randn(1, 4, 2, 8)
    block_embeddings = torch.randn(1, 3, 8)

    baseline = body(
        target_taps=target_taps,
        block_embeddings=block_embeddings,
        plan=base_plan,
    )
    high_position_output = body(
        target_taps=target_taps,
        block_embeddings=block_embeddings,
        plan=high_plan,
    )

    assert high_plan.query_positions.dtype == torch.int64
    assert int(high_plan.query_positions.max()) == position + 2
    assert torch.isfinite(high_position_output).all()
    assert not torch.allclose(baseline, high_position_output)


def test_forward_rejects_mismatched_caller_owned_inputs() -> None:
    body = DFlashBody(
        _tiny_config(num_hidden_layers=1),
        parallel_config=_fp32_parallel_config(),
    )
    plan = _plan(torch.ones((1, 4), dtype=torch.bool), gamma=2)

    with pytest.raises(ValueError, match="target_taps"):
        body(
            target_taps=torch.randn(1, 4, 1, 8),
            block_embeddings=torch.randn(1, 3, 8),
            plan=plan,
        )
    with pytest.raises(ValueError, match="block_embeddings"):
        body(
            target_taps=torch.randn(1, 4, 2, 8),
            block_embeddings=torch.randn(1, 2, 8),
            plan=plan,
        )


def test_constructor_rejects_sequence_parallel_config_without_mutating_it() -> None:
    parallel_config = _fp32_parallel_config(
        tensor_parallel_size=2,
        sequence_parallel=True,
    )

    with pytest.raises(
        ValueError,
        match="DFlashBody does not support sequence_parallel=True",
    ):
        DFlashBody(
            _tiny_config(num_hidden_layers=1),
            parallel_config=parallel_config,
        )

    assert parallel_config.sequence_parallel is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bfloat16_cuda_forward_backward() -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    config = DFlashBodyConfig(
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=2,
        num_target_taps=2,
        rope_theta=10_000.0,
    )
    body = DFlashBody(config).to(device=device, dtype=torch.bfloat16)
    token_valid_mask = torch.ones((2, 5), dtype=torch.bool, device=device)
    sample_ids = torch.arange(2, dtype=torch.int64, device=device)
    plan = build_dflash_batch_plan(
        token_valid_mask,
        sample_ids,
        anchors_per_sample=1,
        gamma=2,
        optimizer_step=0,
        seed=5,
    )
    target_taps = torch.randn(
        2,
        5,
        2,
        32,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    block_embeddings = torch.randn(
        2,
        3,
        32,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )

    output = body(
        target_taps=target_taps,
        block_embeddings=block_embeddings,
        plan=plan,
    )
    output.float().square().mean().backward()

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert target_taps.grad is not None and torch.isfinite(target_taps.grad).all()
    assert block_embeddings.grad is not None
    assert torch.isfinite(block_embeddings.grad).all()
