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

import copy
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo_rl.models.automodel.shared_prefix import (
    SHARED_PREFIX_ATTENTION,
    build_shared_prefix_layout,
    infer_shared_prefix_response_bounds,
    register_shared_prefix_attention,
    response_logprobs_from_logits,
    scatter_response_logprobs,
    shared_prefix_flash_attention_forward,
)


def _interleaved_rollouts():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 21, 22],
            [40, 41, 51, 52, 53],
            [10, 11, 12, 31, 0],
            [40, 41, 61, 62, 0],
        ]
    )
    input_lengths = torch.tensor([5, 5, 4, 4])
    prompt_lengths = torch.tensor([3, 2, 3, 2])
    group_ids = torch.tensor([7, 9, 7, 9])
    return input_ids, input_lengths, prompt_lengths, group_ids


def test_infer_response_bounds_finds_prompt_lengths():
    _, input_lengths, prompt_lengths, _ = _interleaved_rollouts()
    token_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
        ]
    )

    actual, effective_input_lengths = infer_shared_prefix_response_bounds(
        token_mask, input_lengths
    )

    assert torch.equal(actual, prompt_lengths)
    assert torch.equal(effective_input_lengths, input_lengths)


def test_infer_response_bounds_crops_terminal_environment_observation():
    token_mask = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    input_lengths = torch.tensor([6, 4])

    prompt_lengths, effective_input_lengths = infer_shared_prefix_response_bounds(
        token_mask, input_lengths
    )

    assert prompt_lengths.tolist() == [2, 2]
    assert effective_input_lengths.tolist() == [4, 3]


def test_register_shared_prefix_attention_registers_attention_and_mask():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    register_shared_prefix_attention()

    assert (
        AttentionInterface()[SHARED_PREFIX_ATTENTION]
        is shared_prefix_flash_attention_forward
    )
    assert SHARED_PREFIX_ATTENTION in AttentionMaskInterface()


@pytest.mark.parametrize(
    "token_mask,match",
    [
        (torch.tensor([[0, 1, 0, 1]]), "single contiguous response span"),
        (torch.tensor([[0, 0, 0, 0]]), "at least one response token"),
        (torch.tensor([[1, 1, 1, 1]]), "at least one prompt token"),
    ],
)
def test_infer_response_bounds_rejects_unsupported_masks(token_mask, match):
    with pytest.raises(ValueError, match=match):
        infer_shared_prefix_response_bounds(token_mask, torch.tensor([4]))


def test_build_shared_prefix_layout_exact_mapping():
    layout = build_shared_prefix_layout(*_interleaved_rollouts())

    assert layout.compact_input_ids.tolist() == [
        [10, 11, 12, 21, 22, 31, 40, 41, 51, 52, 53, 61, 62]
    ]
    assert layout.compact_position_ids.tolist() == [
        [0, 1, 2, 3, 4, 3, 0, 1, 2, 3, 4, 2, 3]
    ]
    assert layout.prompt_token_indices.tolist() == [0, 1, 2, 6, 7]
    assert layout.response_token_indices.tolist() == [3, 4, 5, 8, 9, 10, 11, 12]
    assert layout.prompt_cu_seqlens.tolist() == [0, 3, 5]
    assert layout.response_cu_seqlens.tolist() == [0, 2, 3, 6, 8]
    assert layout.response_kv_cu_seqlens.tolist() == [0, 5, 9, 14, 18]
    assert layout.predictor_indices.tolist() == [2, 3, 2, 7, 8, 9, 7, 11]
    assert layout.response_target_ids.tolist() == [21, 22, 31, 51, 52, 53, 61, 62]
    assert layout.loss_logprob_scatter_indices.tolist() == [2, 3, 10, 5, 6, 7, 13, 14]
    assert layout.compact_tokens == 13


def test_build_shared_prefix_layout_rejects_false_groups():
    input_ids, input_lengths, prompt_lengths, group_ids = _interleaved_rollouts()
    input_ids[2, 1] = 99

    with pytest.raises(ValueError, match="identical prompts"):
        build_shared_prefix_layout(
            input_ids,
            input_lengths,
            prompt_lengths,
            group_ids,
        )


def test_scatter_response_logprobs_preserves_dense_loss_alignment():
    layout = build_shared_prefix_layout(*_interleaved_rollouts())
    response_logprobs = torch.arange(1, 9, dtype=torch.float32, requires_grad=True)

    dense = scatter_response_logprobs(response_logprobs, layout)

    torch.testing.assert_close(
        dense,
        torch.tensor(
            [
                [0, 0, 1, 2],
                [0, 4, 5, 6],
                [0, 0, 3, 0],
                [0, 7, 8, 0],
            ],
            dtype=torch.float32,
        ),
    )
    dense.sum().backward()
    torch.testing.assert_close(response_logprobs.grad, torch.ones(8))


def test_scatter_response_logprobs_preserves_inactive_base_values():
    layout = build_shared_prefix_layout(*_interleaved_rollouts())
    response_logprobs = torch.arange(1, 9, dtype=torch.float32)
    base = torch.full((4, 4), -7.0)

    dense = scatter_response_logprobs(response_logprobs, layout, base=base)

    assert torch.equal(dense[0, :2], torch.tensor([-7.0, -7.0]))
    assert dense[2, 3] == -7.0
    assert dense[3, 3] == -7.0


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_chunked_response_logprobs_match_dense_values_and_gradients(chunk_size):
    layout = build_shared_prefix_layout(*_interleaved_rollouts())
    torch.manual_seed(17)
    actual_logits = torch.randn(
        1,
        layout.response_tokens,
        128,
        dtype=torch.float32,
        requires_grad=True,
    )
    expected_logits = actual_logits.detach().clone().requires_grad_()
    weights = torch.linspace(-1.0, 1.0, layout.response_tokens)

    actual = response_logprobs_from_logits(
        actual_logits,
        layout,
        chunk_size=chunk_size,
    )
    expected = (
        torch.log_softmax(expected_logits.squeeze(0), dim=-1)
        .gather(-1, layout.response_target_ids.unsqueeze(-1))
        .squeeze(-1)
    )

    actual_loss = (actual * weights).sum()
    expected_loss = (expected * weights).sum()
    actual_grad = torch.autograd.grad(actual_loss, actual_logits)[0]
    expected_grad = torch.autograd.grad(expected_loss, expected_logits)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_chunked_response_logprobs_do_not_save_fp32_vocab_activations():
    layout = build_shared_prefix_layout(*_interleaved_rollouts())
    logits = torch.randn(
        1,
        layout.response_tokens,
        128,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        response_logprobs_from_logits(logits, layout, chunk_size=3).sum().backward()

    assert any(
        tensor.dtype == torch.bfloat16
        and tensor.shape == (layout.response_tokens, logits.shape[-1])
        for tensor in saved_tensors
    )
    assert any(
        tensor.dtype == torch.long and tensor.shape == (layout.response_tokens,)
        for tensor in saved_tensors
    )
    assert not any(
        tensor.dtype == torch.float32
        and tensor.ndim == 2
        and tensor.shape[-1] == logits.shape[-1]
        for tensor in saved_tensors
    )
    assert logits.grad is not None


def _reference_varlen_attention(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    del max_seqlen_q, max_seqlen_k
    assert dropout_p == 0.0
    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_k.dtype == torch.int32
    outputs = []
    for sequence in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = cu_seqlens_q[sequence : sequence + 2].tolist()
        k_start, k_end = cu_seqlens_k[sequence : sequence + 2].tolist()
        q = query[q_start:q_end].transpose(0, 1).float()
        k = key[k_start:k_end].transpose(0, 1).float()
        v = value[k_start:k_end].transpose(0, 1).float()
        scale = (
            softmax_scale if softmax_scale is not None else 1 / math.sqrt(q.shape[-1])
        )
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if causal:
            q_positions = torch.arange(q.shape[-2]).unsqueeze(-1)
            k_positions = torch.arange(k.shape[-2]).unsqueeze(0)
            causal_mask = k_positions <= q_positions + k.shape[-2] - q.shape[-2]
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probabilities, v).transpose(0, 1))
    return torch.cat(outputs).to(query.dtype)


def test_custom_backend_preserves_dense_packed_attention_and_gradients():
    """The train-only backend must leave prev/ref packed forwards unchanged."""

    torch.manual_seed(31)
    query = torch.randn(1, 2, 7, 8, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    expected_query = query.detach().clone().requires_grad_()
    expected_key = key.detach().clone().requires_grad_()
    expected_value = value.detach().clone().requires_grad_()
    # Exercise the dtype emitted by NeMo-RL's packed logprob path. The custom
    # backend must normalize it to FA2's int32 ABI.
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int64)
    packed_kwargs = SimpleNamespace(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens.clone(),
        max_seqlen_q=4,
        max_seqlen_k=4,
    )

    module = torch.nn.Module()
    module.is_causal = True
    with patch(
        "nemo_rl.models.automodel.shared_prefix._flash_attention_functions",
        return_value=(None, _reference_varlen_attention),
    ):
        actual, _ = shared_prefix_flash_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask=None,
            flash_attn_kwargs=packed_kwargs,
        )

    expected = _reference_varlen_attention(
        expected_query.transpose(1, 2).reshape(-1, 2, 8),
        expected_key.transpose(1, 2).reshape(-1, 2, 8),
        expected_value.transpose(1, 2).reshape(-1, 2, 8),
        cu_seqlens.to(dtype=torch.int32),
        cu_seqlens.to(dtype=torch.int32),
        4,
        4,
        causal=True,
    ).unsqueeze(0)
    weights = torch.linspace(0.25, 1.25, actual.numel()).reshape_as(actual)
    actual_loss = (actual.float() * weights).sum()
    expected_loss = (expected.float() * weights).sum()
    actual_grads = torch.autograd.grad(actual_loss, (query, key, value))
    expected_grads = torch.autograd.grad(
        expected_loss,
        (expected_query, expected_key, expected_value),
    )

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_split_attention_matches_logical_dense_attention_and_gradients():
    layout = build_shared_prefix_layout(*_interleaved_rollouts())
    torch.manual_seed(1234)
    compact_q = torch.randn(1, 2, layout.compact_tokens, 8, dtype=torch.bfloat16)
    compact_k = torch.randn_like(compact_q)
    compact_v = torch.randn_like(compact_q)
    compact_q.requires_grad_()
    compact_k.requires_grad_()
    compact_v.requires_grad_()

    dense_gather = torch.tensor(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 0, 1, 2, 5, 6, 7, 11, 12]
    )
    dense_cu = torch.tensor([0, 5, 10, 14, 18], dtype=torch.int32)
    dense_predictors = torch.tensor([2, 3, 12, 6, 7, 8, 15, 16])
    weights = torch.linspace(0.25, 2.0, 8).view(-1, 1, 1)

    def logical_dense_loss(q, k, v):
        dense_output = _reference_varlen_attention(
            q.transpose(1, 2).squeeze(0)[dense_gather],
            k.transpose(1, 2).squeeze(0)[dense_gather],
            v.transpose(1, 2).squeeze(0)[dense_gather],
            dense_cu,
            dense_cu,
            5,
            5,
            causal=True,
        )
        return (dense_output[dense_predictors].float() * weights).sum()

    dense_loss = logical_dense_loss(compact_q, compact_k, compact_v)
    dense_grads = torch.autograd.grad(
        dense_loss,
        (compact_q, compact_k, compact_v),
    )

    with patch(
        "nemo_rl.models.automodel.shared_prefix._flash_attention_functions",
        return_value=(None, _reference_varlen_attention),
    ):
        compact_output, _ = shared_prefix_flash_attention_forward(
            torch.nn.Identity(),
            compact_q,
            compact_k,
            compact_v,
            attention_mask=None,
            shared_prefix_layout=layout,
        )
    compact_loss = (
        compact_output.squeeze(0)[layout.predictor_indices].float() * weights
    ).sum()
    compact_grads = torch.autograd.grad(
        compact_loss,
        (compact_q, compact_k, compact_v),
    )

    torch.testing.assert_close(compact_loss, dense_loss, atol=2e-2, rtol=2e-3)
    for compact_grad, dense_grad in zip(compact_grads, dense_grads):
        torch.testing.assert_close(compact_grad, dense_grad, atol=2e-2, rtol=2e-2)


@pytest.mark.automodel
@pytest.mark.parametrize("activation_checkpointing", [False, True])
@pytest.mark.parametrize("model_family", ["qwen3", "llama"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_tiny_causal_lm_fa2_logits_and_gradients_match_dense(
    activation_checkpointing,
    model_family,
):
    pytest.importorskip("flash_attn")

    if model_family == "qwen3":
        from transformers import Qwen3Config as ModelConfig
        from transformers import Qwen3ForCausalLM as ModelForCausalLM
    else:
        from transformers import LlamaConfig as ModelConfig
        from transformers import LlamaForCausalLM as ModelForCausalLM

    register_shared_prefix_attention()
    dense_config = ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )
    dense_config._attn_implementation = "flash_attention_2"
    shared_config = copy.deepcopy(dense_config)
    shared_config._attn_implementation = SHARED_PREFIX_ATTENTION

    torch.manual_seed(42)
    dense_model = ModelForCausalLM(dense_config).cuda().train()
    shared_model = ModelForCausalLM(shared_config).cuda().train()
    shared_model.load_state_dict(dense_model.state_dict())
    if activation_checkpointing:
        dense_model.gradient_checkpointing_enable()
        shared_model.gradient_checkpointing_enable()

    input_ids = torch.tensor(
        [
            [5, 6, 7, 20, 21, 22],
            [8, 9, 10, 30, 31, 32],
            [5, 6, 7, 23, 24, 25],
            [8, 9, 10, 33, 34, 35],
        ],
        device="cuda",
    )
    lengths = torch.full((4,), 6, device="cuda")
    prompt_lengths = torch.full((4,), 3, device="cuda")
    group_ids = torch.tensor([0, 1, 0, 1], device="cuda")
    layout = build_shared_prefix_layout(
        input_ids,
        lengths,
        prompt_lengths,
        group_ids,
    )
    assert layout.predictor_indices.tolist() == [
        2,
        3,
        4,
        2,
        6,
        7,
        11,
        12,
        13,
        11,
        15,
        16,
    ]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dense_logits = dense_model(input_ids=input_ids, use_cache=False).logits
        shared_logits = shared_model(
            input_ids=layout.compact_input_ids,
            position_ids=layout.compact_position_ids,
            use_cache=False,
            shared_prefix_layout=layout,
            logits_to_keep=layout.predictor_indices,
        ).logits
        dense_response_logits = dense_logits[:, :-1].reshape(-1, 128)[
            layout.loss_logprob_scatter_indices
        ]

    torch.testing.assert_close(
        shared_logits.squeeze(0).float(),
        dense_response_logits.float(),
        atol=5e-2,
        rtol=2e-2,
    )

    targets = layout.response_target_ids.unsqueeze(-1)
    weights = torch.linspace(0.5, 1.5, layout.response_tokens, device="cuda")
    dense_logprobs = torch.log_softmax(dense_response_logits.float(), dim=-1)
    dense_loss = (dense_logprobs.gather(-1, targets).squeeze(-1) * weights).sum()
    shared_logprobs = torch.log_softmax(shared_logits.squeeze(0).float(), dim=-1)
    shared_loss = (shared_logprobs.gather(-1, targets).squeeze(-1) * weights).sum()
    dense_loss.backward()
    shared_loss.backward()

    torch.testing.assert_close(shared_loss, dense_loss, atol=5e-2, rtol=2e-3)
    for (dense_name, dense_parameter), (shared_name, shared_parameter) in zip(
        dense_model.named_parameters(),
        shared_model.named_parameters(),
    ):
        assert dense_name == shared_name
        assert dense_parameter.grad is not None, dense_name
        assert shared_parameter.grad is not None, shared_name
        torch.testing.assert_close(
            shared_parameter.grad,
            dense_parameter.grad,
            atol=7e-2,
            rtol=7e-2,
            msg=lambda message, name=dense_name: f"{name}: {message}",
        )

    dense_optimizer = torch.optim.AdamW(dense_model.parameters(), lr=1e-4)
    shared_optimizer = torch.optim.AdamW(shared_model.parameters(), lr=1e-4)
    dense_optimizer.step()
    shared_optimizer.step()
    for (dense_name, dense_parameter), (shared_name, shared_parameter) in zip(
        dense_model.named_parameters(),
        shared_model.named_parameters(),
    ):
        assert dense_name == shared_name
        torch.testing.assert_close(
            shared_parameter,
            dense_parameter,
            atol=2e-4,
            rtol=2e-4,
            msg=lambda message, name=dense_name: f"{name}: {message}",
        )
        dense_state = dense_optimizer.state[dense_parameter]
        shared_state = shared_optimizer.state[shared_parameter]
        for state_name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                shared_state[state_name],
                dense_state[state_name],
                atol=7e-3,
                rtol=8e-2,
                msg=lambda message, name=dense_name, state=state_name: (
                    f"{name}.{state}: {message}"
                ),
            )
