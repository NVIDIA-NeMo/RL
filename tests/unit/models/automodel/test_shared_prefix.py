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
from types import MethodType, SimpleNamespace
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
    assert layout.logical_token_weights.tolist() == [
        2,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
    ]
    assert layout.logical_token_weights.dtype == torch.long
    assert layout.logical_token_weights.sum() == 18
    assert layout.prompt_token_indices.tolist() == [0, 1, 2, 6, 7]
    assert layout.response_token_indices.tolist() == [3, 4, 5, 8, 9, 10, 11, 12]
    assert layout.prompt_cu_seqlens.tolist() == [0, 3, 5]
    assert layout.response_cu_seqlens.tolist() == [0, 2, 3, 6, 8]
    assert layout.response_kv_cu_seqlens.tolist() == [0, 5, 9, 14, 18]
    assert layout.predictor_indices.tolist() == [2, 3, 2, 7, 8, 9, 7, 11]
    assert layout.response_target_ids.tolist() == [21, 22, 31, 51, 52, 53, 61, 62]
    assert layout.loss_logprob_scatter_indices.tolist() == [2, 3, 10, 5, 6, 7, 13, 14]
    assert layout.compact_tokens == 13


def test_shared_prefix_moe_router_matches_logical_dense_tokens():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.megatron.moe_utils import (
        MoEAuxLossAutoScaler,
    )

    from nemo_rl.models.automodel.shared_prefix_moe import (
        _logical_router_forward,
        shared_prefix_moe_context,
    )

    layout = build_shared_prefix_layout(*_interleaved_rollouts())
    compact_x = (
        torch.linspace(
            -1.0,
            1.0,
            layout.compact_tokens * 3,
            dtype=torch.float32,
        )
        .reshape(layout.compact_tokens, 3)
        .requires_grad_()
    )
    dense_gather = torch.tensor(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 0, 1, 2, 5, 6, 7, 11, 12]
    )
    dense_x = compact_x.detach()[dense_gather].requires_grad_()

    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.03125,
        score_func="softmax",
        route_scale=1.0,
        dim=3,
        inter_dim=8,
        moe_inter_dim=8,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.float32,
    )
    dense_gate = Gate(config, gate_precision=torch.float32).train()
    with torch.no_grad():
        dense_gate.weight.copy_(
            torch.tensor(
                [
                    [0.50, -0.20, 0.10],
                    [-0.40, 0.30, 0.20],
                    [0.20, 0.60, -0.50],
                    [-0.10, -0.40, 0.70],
                ]
            )
        )
    compact_gate = copy.deepcopy(dense_gate)
    dense_gate._track_load_balance = True
    compact_gate._track_load_balance = True
    compact_gate._nemo_shared_prefix_original_forward = compact_gate.forward
    compact_gate.forward = MethodType(_logical_router_forward, compact_gate)

    dense_weights, dense_indices, dense_aux_loss = dense_gate(
        dense_x,
        torch.ones(dense_x.shape[0], dtype=torch.bool),
        None,
    )
    with shared_prefix_moe_context(compact_gate, layout, valid=True) as execution:
        compact_weights, compact_indices, compact_aux_loss = compact_gate(
            compact_x,
            torch.ones(compact_x.shape[0], dtype=torch.bool),
            None,
        )
        execution.commit()

    assert torch.equal(compact_indices[dense_gather], dense_indices)
    torch.testing.assert_close(compact_weights[dense_gather], dense_weights)
    torch.testing.assert_close(compact_aux_loss, dense_aux_loss)
    assert torch.equal(compact_gate._last_expert_load, dense_gate._last_expert_load)
    # A non-unit scale catches aux-loss adapters that silently drop the main
    # loss normalization used by AutoModel's forward/backward driver.
    with patch.object(
        MoEAuxLossAutoScaler,
        "main_loss_backward_scale",
        torch.tensor(0.375),
    ):
        dense_weights.sum().backward()
        compact_weights[dense_gather].sum().backward()
    torch.testing.assert_close(compact_gate.weight.grad, dense_gate.weight.grad)
    expected_compact_input_grad = torch.zeros_like(compact_x)
    expected_compact_input_grad.index_add_(0, dense_gather, dense_x.grad)
    torch.testing.assert_close(compact_x.grad, expected_compact_input_grad)


def test_shared_prefix_moe_context_is_module_scoped_and_restored():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate

    from nemo_rl.models.automodel import shared_prefix_moe

    config = MoEConfig(
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=2,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.float32,
    )
    gate = Gate(config, gate_precision=torch.float32)
    model = torch.nn.Sequential(gate)
    outer_layout = SimpleNamespace(logical_token_weights=torch.tensor([2, 1]))
    inner_layout = SimpleNamespace(logical_token_weights=torch.tensor([3, 4]))
    attribute = shared_prefix_moe._ROUTER_EXECUTION_ATTR

    assert not hasattr(gate, attribute)
    with shared_prefix_moe.shared_prefix_moe_context(
        model, outer_layout, valid=True
    ) as outer_execution:
        assert getattr(gate, attribute) is outer_execution
        with shared_prefix_moe.shared_prefix_moe_context(
            model, inner_layout, valid=False
        ) as inner_execution:
            assert getattr(gate, attribute) is inner_execution
        assert getattr(gate, attribute) is outer_execution
    assert not hasattr(gate, attribute)


@pytest.mark.parametrize(
    "bias_update_factor",
    [
        1.0e-3,
        -1.0e-3,
        True,
        False,
        None,
        float("nan"),
        float("inf"),
        -float("inf"),
        "x",
    ],
)
def test_shared_prefix_moe_router_rejects_dynamic_bias_without_layout(
    bias_update_factor,
):
    pytest.importorskip("nemo_automodel")
    from nemo_rl.models.automodel.shared_prefix_moe import _logical_router_forward

    gate = SimpleNamespace(
        bias_update_factor=bias_update_factor,
        _nemo_shared_prefix_original_forward=lambda *_: pytest.fail(
            "dynamic bias reached the original Gate forward"
        ),
    )

    with pytest.raises(ValueError, match="does not support dynamic bias"):
        _logical_router_forward(
            gate,
            torch.zeros(1, 2),
            torch.ones(1, dtype=torch.bool),
            None,
        )


def test_logical_aux_autoscaler_uses_no_saved_tensors_and_preserves_gradient():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.moe.megatron.moe_utils import (
        MoEAuxLossAutoScaler,
    )

    from nemo_rl.models.automodel.shared_prefix_moe import (
        _LogicalAuxLossAutoScaler,
    )

    output = torch.tensor([1.0, -2.0], requires_grad=True)
    aux_loss = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
    saved_tensors = []

    def pack(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        attached = _LogicalAuxLossAutoScaler.apply(output, aux_loss)

    assert saved_tensors == []
    with patch.object(
        MoEAuxLossAutoScaler,
        "main_loss_backward_scale",
        torch.tensor(0.375),
    ):
        attached.sum().backward()

    torch.testing.assert_close(output.grad, torch.ones_like(output))
    torch.testing.assert_close(aux_loss.grad, torch.tensor(0.375, dtype=torch.float64))


def test_router_execution_replay_does_not_double_commit_statistics():
    from nemo_rl.models.automodel import shared_prefix_moe

    gate = torch.nn.Module()
    execution = shared_prefix_moe._RouterExecution(
        logical_token_weights=torch.ones(2, dtype=torch.long),
        valid=True,
    )
    first_load = torch.tensor([3.0, 1.0], requires_grad=True)
    first_aux = torch.tensor(0.25, requires_grad=True)

    with patch.object(
        shared_prefix_moe,
        "_detach_optional",
        wraps=shared_prefix_moe._detach_optional,
    ) as detach_optional:
        execution.record(gate, first_load, first_aux)
        execution.commit()

        committed_load = gate._nemo_shared_prefix_logical_expert_load.clone()
        committed_aux = gate._nemo_shared_prefix_aux_loss_sum.clone()
        assert gate._nemo_shared_prefix_aux_loss_count == 1

        # Model activation checkpointing replays Gate.forward after commit.
        # Snapshot construction and the pending write stay isomorphic to the
        # original forward, while commit remains one-shot.
        replay_load = torch.tensor([1.0, 3.0], requires_grad=True)
        replay_aux = torch.tensor(0.75, requires_grad=True)
        execution.record(gate, replay_load, replay_aux)
        assert detach_optional.call_count == 2
        assert gate in execution.pending
        assert execution.pending[gate][0].grad_fn is None
        assert execution.pending[gate][1].grad_fn is None

        execution.commit()

    assert execution.pending == {}
    torch.testing.assert_close(
        gate._nemo_shared_prefix_logical_expert_load, committed_load
    )
    torch.testing.assert_close(gate._nemo_shared_prefix_aux_loss_sum, committed_aux)
    assert gate._nemo_shared_prefix_aux_loss_count == 1


@pytest.mark.parametrize(
    ("n_experts", "world_size", "expected_ids"),
    [
        (7, 2, ((0, 1, 2, 3), (4, 5, 6))),
        (7, 4, ((0, 1), (2, 3), (4, 5), (6,))),
        (3, 4, ((0,), (1,), (2,), ())),
    ],
)
def test_torch_chunk_expert_ownership(
    n_experts: int,
    world_size: int,
    expected_ids: tuple[tuple[int, ...], ...],
) -> None:
    from nemo_rl.models.automodel.shared_prefix_moe import (
        _torch_chunk_shard_size_and_offset,
    )

    actual_ids = []
    for rank in range(world_size):
        local_count, start = _torch_chunk_shard_size_and_offset(
            n_experts, world_size, rank
        )
        actual_ids.append(tuple(range(start, start + local_count)))

    assert tuple(actual_ids) == expected_ids


def test_qwen3_moe_ep_router_gradient_backport_is_idempotent():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.moe.experts import GroupedExperts

    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_ep_router_gradients,
    )

    original_forward = GroupedExperts.forward
    try:
        enable_qwen3_moe_ep_router_gradients()
        patched_forward = GroupedExperts.forward
        assert patched_forward is not original_forward
        assert patched_forward._nemo_ep_router_gradients is True
        enable_qwen3_moe_ep_router_gradients()
        assert GroupedExperts.forward is patched_forward
    finally:
        GroupedExperts.forward = original_forward


def test_qwen3_moe_ep_router_gradient_backport_fails_closed():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.moe.experts import GroupedExperts

    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_ep_router_gradients,
    )

    original_forward = GroupedExperts.forward
    with (
        patch(
            "nemo_rl.models.automodel.shared_prefix_moe.inspect.getsource",
            return_value="def forward(self):\n    return None\n",
        ),
        pytest.raises(RuntimeError, match="detached EP routing-weight gather"),
    ):
        enable_qwen3_moe_ep_router_gradients()

    assert GroupedExperts.forward is original_forward


def test_qwen3_moe_checkpoint_wrapper_compatibility_preserves_wrapper_call():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_moe.model import Block
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import MLP, MoE
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_checkpoint_wrapper_compatibility,
    )

    config = MoEConfig(
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=4,
        inter_dim=8,
        moe_inter_dim=6,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.float32,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    original = Block._mlp
    try:
        enable_qwen3_moe_checkpoint_wrapper_compatibility()
        patched = Block._mlp
        enable_qwen3_moe_checkpoint_wrapper_compatibility()
        assert Block._mlp is patched

        block = object.__new__(Block)
        torch.nn.Module.__init__(block)
        dense = MLP(4, 8, "torch", dtype=torch.float32)
        with torch.no_grad():
            dense.init_weights(torch.device("cpu"))
        block.mlp = checkpoint_wrapper(dense)
        dense_input = torch.randn(2, 3, 4)
        torch.testing.assert_close(
            block._mlp(dense_input, None),
            dense(dense_input),
        )

        moe = MoE(config, backend)
        with torch.no_grad():
            moe.init_weights(torch.device("cpu"))
        block.mlp = checkpoint_wrapper(moe)
        moe_input = torch.randn(2, 3, 4)
        padding_mask = torch.zeros(2, 3, dtype=torch.bool)
        torch.testing.assert_close(
            block._mlp(moe_input, padding_mask),
            moe(moe_input, padding_mask),
        )
    finally:
        Block._mlp = original


def test_qwen3_moe_checkpoint_wrapper_compatibility_fails_closed():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.models.qwen3_moe.model import Block

    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_checkpoint_wrapper_compatibility,
    )

    original = Block._mlp
    with (
        patch(
            "nemo_rl.models.automodel.shared_prefix_moe.inspect.getsource",
            return_value="def _mlp(self, x, padding_mask):\n    return x\n",
        ),
        pytest.raises(RuntimeError, match="checkpoint-wrapper source"),
    ):
        enable_qwen3_moe_checkpoint_wrapper_compatibility()

    assert Block._mlp is original


def _qwen3_moe_dtype_backport_methods():
    from nemo_automodel.components.models.qwen3_moe.layers import (
        Qwen3MoeAttention,
    )
    from nemo_automodel.components.models.qwen3_moe.model import (
        Block,
        Qwen3MoeForCausalLM,
        Qwen3MoeModel,
    )

    return (
        (Qwen3MoeAttention, "__init__"),
        (Block, "__init__"),
        (Qwen3MoeModel, "__init__"),
        (Qwen3MoeForCausalLM, "__init__"),
        (Qwen3MoeForCausalLM, "forward"),
        (Qwen3MoeForCausalLM, "initialize_weights"),
    )


def test_qwen3_moe_configured_dtype_backport_constructs_fp32_on_cpu():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_moe.model import (
        Qwen3MoeForCausalLM,
    )
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )

    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_configured_dtype,
    )

    targets = _qwen3_moe_dtype_backport_methods()
    originals = tuple(getattr(owner, name) for owner, name in targets)
    try:
        enable_qwen3_moe_configured_dtype()
        config = Qwen3MoeConfig(
            vocab_size=64,
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            num_hidden_layers=2,
            intermediate_size=32,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            mlp_only_layers=[0],
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            rope_theta=5000.0,
            router_aux_loss_coef=0.01,
            use_sliding_window=False,
        )
        config.torch_dtype = torch.float32
        backend = BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

        # The pinned model only stores this device in its RoPE helper during
        # construction; avoid CUDA initialization so this remains a CPU test.
        with patch.object(torch.cuda, "current_device", return_value=0):
            model = Qwen3MoeForCausalLM(config, backend=backend)
        assert model.model.moe_config.dtype == torch.float32
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())

        model.initialize_weights(buffer_device=torch.device("cpu"))
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    finally:
        for (owner, name), original in zip(targets, originals):
            setattr(owner, name, original)


def test_qwen3_moe_configured_dtype_backport_is_idempotent():
    pytest.importorskip("nemo_automodel")
    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_configured_dtype,
    )

    targets = _qwen3_moe_dtype_backport_methods()
    originals = tuple(getattr(owner, name) for owner, name in targets)
    try:
        enable_qwen3_moe_configured_dtype()
        patched = tuple(getattr(owner, name) for owner, name in targets)
        assert patched != originals
        assert all(
            "__class__" not in method.__code__.co_freevars for method in patched[:4]
        )
        assert all(
            method._nemo_qwen3_moe_configured_dtype is True for method in patched
        )

        enable_qwen3_moe_configured_dtype()
        assert tuple(getattr(owner, name) for owner, name in targets) == patched
    finally:
        for (owner, name), original in zip(targets, originals):
            setattr(owner, name, original)


def test_qwen3_moe_configured_dtype_backport_casts_lm_head_input():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.models.qwen3_moe.model import (
        Qwen3MoeForCausalLM,
    )

    from nemo_rl.models.automodel.shared_prefix_moe import (
        enable_qwen3_moe_configured_dtype,
    )

    class _HiddenState(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden

        def forward(self, *args, **kwargs):
            return self.hidden

    targets = _qwen3_moe_dtype_backport_methods()
    originals = tuple(getattr(owner, name) for owner, name in targets)
    try:
        enable_qwen3_moe_configured_dtype()
        hidden = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
        model = object.__new__(Qwen3MoeForCausalLM)
        torch.nn.Module.__init__(model)
        model.model = _HiddenState(hidden)
        model.lm_head = torch.nn.Linear(
            4,
            5,
            bias=False,
            dtype=torch.bfloat16,
        )

        logits = model(torch.zeros(2, 3, dtype=torch.long))
        expected = torch.nn.functional.linear(
            hidden.to(torch.bfloat16),
            model.lm_head.weight,
        )
        torch.testing.assert_close(logits, expected)
        assert logits.dtype == torch.bfloat16

        logits.float().sum().backward()
        assert hidden.grad is not None
        assert model.lm_head.weight.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert torch.isfinite(model.lm_head.weight.grad).all()
    finally:
        for (owner, name), original in zip(targets, originals):
            setattr(owner, name, original)


def test_qwen3_moe_configured_dtype_backport_fails_without_partial_mutation():
    pytest.importorskip("nemo_automodel")
    from nemo_rl.models.automodel import shared_prefix_moe

    targets = _qwen3_moe_dtype_backport_methods()
    originals = tuple(getattr(owner, name) for owner, name in targets)
    real_getsource = shared_prefix_moe.inspect.getsource

    def mismatched_third_method(method):
        if method is originals[2]:
            return "def __init__(self):\n    pass\n"
        return real_getsource(method)

    try:
        with (
            patch.object(
                shared_prefix_moe.inspect,
                "getsource",
                side_effect=mismatched_third_method,
            ),
            pytest.raises(RuntimeError, match="no longer matches"),
        ):
            shared_prefix_moe.enable_qwen3_moe_configured_dtype()

        assert tuple(getattr(owner, name) for owner, name in targets) == originals
    finally:
        for (owner, name), original in zip(targets, originals):
            setattr(owner, name, original)


def test_shared_prefix_moe_bf16_fp32_gate_aux_matches_expanded_tokens():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.megatron.moe_utils import (
        MoEAuxLossAutoScaler,
    )

    from nemo_rl.models.automodel.shared_prefix_moe import (
        _logical_router_forward,
        shared_prefix_moe_context,
    )

    # The logical total (515) is not exactly representable in BF16. This catches
    # implementations that accidentally cast load/count normalization to the
    # router-probability dtype instead of preserving the dense FP32 semantics.
    multiplicity = torch.tensor([129, 128, 127, 66, 65], dtype=torch.long)
    compact_x = torch.tensor(
        [
            [0.25, -0.50, 0.75],
            [-0.80, 0.20, 0.45],
            [0.60, 0.15, -0.35],
            [-0.10, 0.90, 0.30],
            [0.55, -0.25, -0.65],
        ],
        dtype=torch.bfloat16,
    )
    dense_x = compact_x.repeat_interleave(multiplicity, dim=0)
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.125,
        score_func="softmax",
        route_scale=1.0,
        dim=3,
        inter_dim=8,
        moe_inter_dim=8,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.bfloat16,
    )
    dense_gate = Gate(config, gate_precision=torch.float32).train()
    with torch.no_grad():
        dense_gate.weight.copy_(
            torch.tensor(
                [
                    [0.50, -0.20, 0.10],
                    [-0.40, 0.30, 0.20],
                    [0.20, 0.60, -0.50],
                    [-0.10, -0.40, 0.70],
                ],
                dtype=torch.bfloat16,
            )
        )
    compact_gate = copy.deepcopy(dense_gate)
    compact_gate._nemo_shared_prefix_original_forward = compact_gate.forward
    compact_gate.forward = MethodType(_logical_router_forward, compact_gate)
    token_mask = torch.ones(compact_x.shape[0], dtype=torch.bool)
    dense_mask = torch.ones(dense_x.shape[0], dtype=torch.bool)
    layout = SimpleNamespace(logical_token_weights=multiplicity)

    dense_weights, dense_indices, dense_aux = dense_gate(dense_x, dense_mask, None)
    with shared_prefix_moe_context(compact_gate, layout, valid=True) as execution:
        compact_weights, compact_indices, compact_aux = compact_gate(
            compact_x, token_mask, None
        )
        execution.commit()

    assert torch.equal(
        compact_indices.repeat_interleave(multiplicity, dim=0), dense_indices
    )
    torch.testing.assert_close(
        compact_weights.repeat_interleave(multiplicity, dim=0), dense_weights
    )
    torch.testing.assert_close(compact_aux, dense_aux, rtol=3e-3, atol=3e-4)
    with patch.object(
        MoEAuxLossAutoScaler,
        "main_loss_backward_scale",
        torch.tensor(1.0),
    ):
        (dense_weights.sum() * 0.0).backward()
        (compact_weights.sum() * 0.0).backward()
    torch.testing.assert_close(
        compact_gate.weight.grad,
        dense_gate.weight.grad,
        rtol=2e-2,
        atol=2e-3,
    )


def test_shared_prefix_moe_single_expert_statistics_are_finite():
    pytest.importorskip("nemo_automodel")
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.layers import Gate

    from nemo_rl.models.automodel.shared_prefix_moe import (
        collect_shared_prefix_moe_statistics,
    )

    config = MoEConfig(
        n_routed_experts=1,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=2,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=False,
        softmax_before_topk=True,
        dtype=torch.float32,
    )
    gate = Gate(config, gate_precision=torch.float32)
    gate._nemo_shared_prefix_logical_expert_load = torch.tensor([7])
    stats_model = torch.nn.Module()
    stats_model.add_module("gate", gate)
    stats_model._nemo_shared_prefix_custom_qwen3_moe = True

    with patch("torch.distributed.all_reduce"):
        stats = collect_shared_prefix_moe_statistics(stats_model, None)

    assert math.isfinite(stats["logical_load_cv_mean"])
    assert stats["logical_load_cv_mean"] == 0.0
    assert stats["logical_expert_utilization_min"] == 1.0
    assert stats["logical_expert_utilization_max"] == 1.0
    assert stats["logical_dead_expert_fraction_mean"] == 0.0
    assert stats["logical_expert_diversity_mean"] == 1.0
    assert stats["logical_token_layer_events"] == 7.0
    assert all(math.isfinite(value) for value in stats.values())


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
@pytest.mark.parametrize("model_family", ["qwen3", "llama", "qwen3_moe"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_tiny_causal_lm_fa2_logits_and_gradients_match_dense(
    activation_checkpointing,
    model_family,
):
    pytest.importorskip("flash_attn")

    if model_family == "qwen3":
        from transformers import Qwen3Config as ModelConfig
        from transformers import Qwen3ForCausalLM as ModelForCausalLM
    elif model_family == "llama":
        from transformers import LlamaConfig as ModelConfig
        from transformers import LlamaForCausalLM as ModelForCausalLM
    else:
        from transformers import Qwen3MoeConfig as ModelConfig
        from transformers import Qwen3MoeForCausalLM as ModelForCausalLM

    register_shared_prefix_attention()
    config_kwargs = dict(
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
    if model_family == "qwen3_moe":
        config_kwargs.update(
            moe_intermediate_size=64,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            num_experts=2,
            num_experts_per_tok=2,
            norm_topk_prob=False,
            use_sliding_window=False,
            output_router_logits=False,
        )
    dense_config = ModelConfig(**config_kwargs)
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
    moe_gradients: dict[str, torch.Tensor] = {}
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
        if ".mlp.gate.weight" in dense_name or ".mlp.experts." in dense_name:
            moe_gradients[dense_name] = shared_parameter.grad

    if model_family == "qwen3_moe":
        assert any(".mlp.gate.weight" in name for name in moe_gradients)
        assert any(".mlp.experts.gate_up_proj" in name for name in moe_gradients)
        assert any(".mlp.experts.down_proj" in name for name in moe_gradients)
        assert all(
            gradient.abs().sum().item() > 0 for gradient in moe_gradients.values()
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
