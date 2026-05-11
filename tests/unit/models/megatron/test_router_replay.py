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

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.mcore
def test_configure_vllm_for_router_replay_disables_prefix_cache():
    from nemo_rl.models.megatron.router_replay import (
        configure_vllm_for_router_replay,
        validate_router_replay_config,
    )

    config = {
        "router_replay": {"enabled": True},
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {"enable_prefix_caching": True},
            "vllm_kwargs": {},
        },
        "megatron_cfg": {"enabled": True},
    }

    with pytest.warns(RuntimeWarning, match="disables vLLM prefix caching"):
        configure_vllm_for_router_replay(config)

    assert config["generation"]["vllm_kwargs"]["enable_return_routed_experts"] is True
    assert config["generation"]["vllm_cfg"]["enable_prefix_caching"] is False
    validate_router_replay_config(config)


@pytest.mark.mcore
def test_validate_router_replay_config_rejects_prefix_cache_default():
    from nemo_rl.models.megatron.router_replay import validate_router_replay_config

    config = {
        "router_replay": {"enabled": True},
        "generation": {"backend": "vllm", "vllm_cfg": {}, "vllm_kwargs": {}},
        "megatron_cfg": {"enabled": True},
    }

    with pytest.raises(ValueError, match="prefix caching"):
        validate_router_replay_config(config)


@pytest.mark.mcore
def test_normalize_routed_experts_dense_batch_uses_seq_major_order():
    from nemo_rl.models.megatron.router_replay import (
        _normalize_routed_experts_for_mcore,
    )

    routed_experts = torch.arange(2 * 3 * 2 * 1, dtype=torch.int32).reshape(2, 3, 2, 1)

    normalized = _normalize_routed_experts_for_mcore(routed_experts)

    assert torch.equal(
        normalized,
        routed_experts.transpose(0, 1).reshape(6, 2, 1),
    )


@pytest.mark.mcore
def test_build_router_replay_tensors_maps_global_moe_layer_order():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron.router_replay import build_router_replay_tensors

    RouterReplay.clear_global_router_replay_instances()

    class DummyRouter(torch.nn.Module):
        def __init__(self, replay, layer_number):
            super().__init__()
            self.router_replay = replay
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_layers=4, moe_layer_freq=[1, 0, 1, 0])
            self.router_1 = DummyRouter(RouterReplay(), layer_number=1)
            self.router_3 = DummyRouter(RouterReplay(), layer_number=3)

    try:
        model = DummyModel()
        routed_experts = torch.tensor(
            [
                [[10, 11], [30, 31]],
                [[12, 13], [32, 33]],
            ],
            dtype=torch.int32,
        )

        replay_tensors = build_router_replay_tensors(model, routed_experts)

        assert len(replay_tensors) == 2
        assert torch.equal(replay_tensors[0], routed_experts[:, 0, :].to(torch.long))
        assert torch.equal(replay_tensors[1], routed_experts[:, 1, :].to(torch.long))
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_build_router_replay_tensors_maps_full_layer_payload_to_moe_layers():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron.router_replay import build_router_replay_tensors

    RouterReplay.clear_global_router_replay_instances()

    class DummyRouter(torch.nn.Module):
        def __init__(self, replay, layer_number):
            super().__init__()
            self.router_replay = replay
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_layers=4, moe_layer_freq=[0, 1, 1, 0])
            self.router_2 = DummyRouter(RouterReplay(), layer_number=2)
            self.router_3 = DummyRouter(RouterReplay(), layer_number=3)

    try:
        model = DummyModel()
        routed_experts = torch.tensor(
            [
                [[10, 11], [20, 21], [30, 31], [40, 41]],
                [[12, 13], [22, 23], [32, 33], [42, 43]],
            ],
            dtype=torch.int32,
        )

        replay_tensors = build_router_replay_tensors(model, routed_experts)

        assert len(replay_tensors) == 2
        assert torch.equal(replay_tensors[0], routed_experts[:, 1, :].to(torch.long))
        assert torch.equal(replay_tensors[1], routed_experts[:, 2, :].to(torch.long))
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_build_router_replay_tensors_rejects_partial_padded_layer_payload():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron.router_replay import build_router_replay_tensors

    RouterReplay.clear_global_router_replay_instances()

    class DummyRouter(torch.nn.Module):
        def __init__(self, replay, layer_number):
            super().__init__()
            self.router_replay = replay
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_layers=8, moe_layer_freq=[0, 1, 0, 1, 0, 1, 0, 0]
            )
            self.router_2 = DummyRouter(RouterReplay(), layer_number=2)
            self.router_4 = DummyRouter(RouterReplay(), layer_number=4)
            self.router_6 = DummyRouter(RouterReplay(), layer_number=6)

    try:
        model = DummyModel()
        routed_experts = torch.zeros(2, 6, 2, dtype=torch.int32)

        with pytest.raises(
            ValueError,
            match=(
                "Expected exactly 3 layers for compressed MoE-layer layout or "
                "8 layers for vLLM full-transformer-layer layout"
            ),
        ):
            build_router_replay_tensors(model, routed_experts)
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_router_replay_actions_are_scoped_to_model_instances():
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    from nemo_rl.models.megatron.router_replay import (
        clear_router_replay,
        set_router_replay_backward,
        set_router_replay_forward,
    )

    RouterReplay.clear_global_router_replay_instances()

    class DummyRouter(torch.nn.Module):
        def __init__(self, replay, layer_number):
            super().__init__()
            self.router_replay = replay
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_layers=4, moe_layer_freq=[1, 0, 1, 0])
            self.router_1 = DummyRouter(RouterReplay(), layer_number=1)
            self.router_3 = DummyRouter(RouterReplay(), layer_number=3)

    try:
        active_model = DummyModel()
        inactive_model = DummyModel()
        routed_experts = torch.tensor(
            [
                [[10, 11], [30, 31]],
                [[12, 13], [32, 33]],
            ],
            dtype=torch.int32,
        )

        set_router_replay_forward(active_model, routed_experts)

        active_replays = [
            active_model.router_1.router_replay,
            active_model.router_3.router_replay,
        ]
        inactive_replays = [
            inactive_model.router_1.router_replay,
            inactive_model.router_3.router_replay,
        ]
        assert all(
            replay.router_replay_action == RouterReplayAction.REPLAY_FORWARD
            for replay in active_replays
        )
        assert all(replay.target_topk_idx is not None for replay in active_replays)
        assert all(replay.router_replay_action is None for replay in inactive_replays)
        assert all(replay.target_topk_idx is None for replay in inactive_replays)

        set_router_replay_backward(active_model)
        assert all(
            replay.router_replay_action == RouterReplayAction.REPLAY_BACKWARD
            for replay in active_replays
        )
        assert all(replay.router_replay_action is None for replay in inactive_replays)

        clear_router_replay(active_model)
        assert all(replay.router_replay_action is None for replay in active_replays)
        assert all(replay.target_topk_idx is None for replay in active_replays)
        assert all(replay.router_replay_action is None for replay in inactive_replays)
        assert all(replay.target_topk_idx is None for replay in inactive_replays)
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_router_replay_noops_on_pp_rank_without_local_moe_layers():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron.router_replay import (
        build_router_replay_assignments,
        set_router_replay_forward,
    )

    RouterReplay.clear_global_router_replay_instances()

    class DenseLayer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_layers=4, moe_layer_freq=[0, 1, 0, 1])
            self.layer_1 = DenseLayer(layer_number=1)
            self.layer_3 = DenseLayer(layer_number=3)

    try:
        model = DummyModel()
        routed_experts = torch.tensor(
            [
                [[10, 11], [40, 41]],
                [[12, 13], [42, 43]],
            ],
            dtype=torch.int32,
        )

        assert build_router_replay_assignments(model, routed_experts) == []
        set_router_replay_forward(model, routed_experts)
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_router_replay_requires_instances_on_pp_rank_with_local_moe_layers():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron.router_replay import build_router_replay_assignments

    RouterReplay.clear_global_router_replay_instances()

    class MoeLayerWithoutReplay(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_layers=4, moe_layer_freq=[0, 1, 0, 1])
            self.layer_2 = MoeLayerWithoutReplay(layer_number=2)

    try:
        model = DummyModel()
        routed_experts = torch.tensor(
            [
                [[10, 11], [40, 41]],
                [[12, 13], [42, 43]],
            ],
            dtype=torch.int32,
        )

        with pytest.raises(ValueError, match="local MoE layers \\[2\\]"):
            build_router_replay_assignments(model, routed_experts)
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_router_replay_worker_guard_requires_routed_experts_when_expected():
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _should_use_router_replay,
    )

    data = BatchedDataDict({"input_ids": torch.ones(2, 4, dtype=torch.long)})

    assert not _should_use_router_replay(
        enabled=False,
        data=data,
        stage="prev-logprob",
        require=True,
    )
    assert not _should_use_router_replay(
        enabled=True,
        data=data,
        stage="reference-logprob",
        require=False,
    )
    with pytest.raises(RuntimeError, match="requires routed_experts for prev-logprob"):
        _should_use_router_replay(
            enabled=True,
            data=data,
            stage="prev-logprob",
            require=True,
        )

    data["routed_experts"] = torch.zeros(2, 4, 3, 2, dtype=torch.int32)
    assert _should_use_router_replay(
        enabled=True,
        data=data,
        stage="prev-logprob",
        require=True,
    )


@pytest.mark.mcore
def test_router_replay_backward_recompute_matches_baseline_grad():
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )
    from torch.utils.checkpoint import checkpoint

    RouterReplay.clear_global_router_replay_instances()

    def route_loss(logits, router_replay=None):
        probs, _ = topk_routing_with_score_function(
            logits=logits,
            topk=2,
            use_pre_softmax=True,
            router_replay=router_replay,
            score_function="softmax",
            dense_output=True,
        )
        weights = torch.tensor(
            [
                [0.3, -0.4],
                [1.2, 0.5],
                [-0.7, 0.9],
                [0.1, 1.7],
                [-1.1, 0.2],
                [0.8, -0.6],
            ],
            dtype=probs.dtype,
            device=probs.device,
        )
        return (probs * weights).sum()

    try:
        logits_data = torch.tensor(
            [
                [2.0, -1.0, 0.5, 1.2, -0.3],
                [-0.4, 0.9, 1.7, -1.1, 0.2],
                [0.3, 1.1, -0.8, 2.4, -0.5],
                [1.5, -0.2, 0.7, -1.4, 0.4],
                [-1.2, 0.6, 1.3, 0.8, -0.7],
                [0.9, -0.6, 0.1, 1.8, -1.5],
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            _, target_indices = topk_routing_with_score_function(
                logits=logits_data,
                topk=2,
                use_pre_softmax=True,
                score_function="softmax",
                dense_output=True,
            )

        baseline_logits = logits_data.clone().requires_grad_()
        baseline_loss = route_loss(baseline_logits)
        baseline_loss.backward()

        replay = RouterReplay()
        replay._nrl_layer_number = 1
        replay.set_target_indices(target_indices.clone())
        replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        calls = []

        def checkpointed_route_loss(logits):
            calls.append(replay.router_replay_action)
            return route_loss(logits, router_replay=replay)

        replay_logits = logits_data.clone().requires_grad_()
        replay_loss = checkpoint(
            checkpointed_route_loss,
            replay_logits,
            use_reentrant=True,
        )
        assert calls == [RouterReplayAction.REPLAY_FORWARD]

        replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        replay_loss.backward()

        assert calls == [
            RouterReplayAction.REPLAY_FORWARD,
            RouterReplayAction.REPLAY_BACKWARD,
        ]
        assert replay.replay_backward_list == []
        torch.testing.assert_close(replay_loss, baseline_loss)
        torch.testing.assert_close(replay_logits.grad, baseline_logits.grad)
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mcore_moe_replay_backward_recompute_matches_parameter_grads(tmp_path):
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    RouterReplay.clear_global_router_replay_instances()

    created_process_group = False

    def make_moe_layer():
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=False,
            moe_token_dispatcher_type="allgather",
            moe_router_load_balancing_type="none",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            recompute_granularity="selective",
            recompute_modules=["moe"],
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            params_dtype=torch.float32,
            moe_enable_routing_replay=True,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=4, moe_grouped_gemm=False
        )
        return MoELayer(config, submodules.mlp.submodules).cuda()

    def run_layer(layer, hidden_states, loss_weights, action, target_indices=None):
        layer.zero_grad(set_to_none=True)
        hidden_states = hidden_states.detach().clone().requires_grad_(True)
        router_replay = layer.router.router_replay
        if target_indices is not None:
            router_replay.set_target_indices(target_indices.clone())
        router_replay.set_router_replay_action(action)

        output, _ = layer(hidden_states)
        recorded_indices = None
        if action == RouterReplayAction.RECORD:
            recorded_indices = router_replay.get_recorded_indices().detach().clone()
        loss = (output.float() * loss_weights).sum()
        if action == RouterReplayAction.REPLAY_FORWARD:
            router_replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        loss.backward()

        parameter_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in layer.named_parameters()
        }
        return (
            loss.detach(),
            hidden_states.grad.detach().clone(),
            parameter_grads,
            recorded_indices,
            router_replay,
        )

    try:
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=0,
                world_size=1,
                init_method=f"file://{tmp_path / 'mcore_pg_init'}",
            )
            created_process_group = True

        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)

        torch.manual_seed(1234)
        baseline_layer = make_moe_layer()
        replay_layer = make_moe_layer()
        replay_layer.load_state_dict(baseline_layer.state_dict())

        hidden_states = torch.randn(
            6, 2, 16, device=torch.cuda.current_device(), dtype=torch.float32
        )
        loss_weights = torch.randn_like(hidden_states)

        (
            baseline_loss,
            baseline_input_grad,
            baseline_parameter_grads,
            target_indices,
            _,
        ) = run_layer(
            baseline_layer,
            hidden_states,
            loss_weights,
            RouterReplayAction.RECORD,
        )
        (
            replay_loss,
            replay_input_grad,
            replay_parameter_grads,
            _,
            replay_router,
        ) = run_layer(
            replay_layer,
            hidden_states,
            loss_weights,
            RouterReplayAction.REPLAY_FORWARD,
            target_indices=target_indices,
        )

        assert replay_router.replay_backward_list == []
        torch.testing.assert_close(replay_loss, baseline_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            replay_input_grad, baseline_input_grad, rtol=1e-5, atol=1e-5
        )
        assert replay_parameter_grads.keys() == baseline_parameter_grads.keys()
        for name, baseline_grad in baseline_parameter_grads.items():
            replay_grad = replay_parameter_grads[name]
            if baseline_grad is None or replay_grad is None:
                assert baseline_grad is None and replay_grad is None, name
            else:
                torch.testing.assert_close(
                    replay_grad,
                    baseline_grad,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=name,
                )
    finally:
        parallel_state.destroy_model_parallel()
        RouterReplay.clear_global_router_replay_instances()
        if created_process_group and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.mcore
def test_build_router_replay_tensors_splits_tokens_for_sequence_parallel(monkeypatch):
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron import router_replay

    RouterReplay.clear_global_router_replay_instances()

    class DummyRouter(torch.nn.Module):
        def __init__(self, replay, layer_number):
            super().__init__()
            self.router_replay = replay
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_layers=4,
                moe_layer_freq=[1, 0, 1, 0],
                sequence_parallel=True,
            )
            self.router_1 = DummyRouter(RouterReplay(), layer_number=1)
            self.router_3 = DummyRouter(RouterReplay(), layer_number=3)

    monkeypatch.setattr(
        router_replay, "_get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(router_replay, "_get_tensor_model_parallel_rank", lambda: 1)

    try:
        model = DummyModel()
        routed_experts = torch.tensor(
            [
                [[10, 11], [30, 31]],
                [[12, 13], [32, 33]],
                [[14, 15], [34, 35]],
                [[16, 17], [36, 37]],
            ],
            dtype=torch.int32,
        )

        replay_tensors = router_replay.build_router_replay_tensors(
            model, routed_experts
        )

        assert len(replay_tensors) == 2
        assert torch.equal(replay_tensors[0], routed_experts[2:, 0, :].to(torch.long))
        assert torch.equal(replay_tensors[1], routed_experts[2:, 1, :].to(torch.long))
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_build_router_replay_tensors_skips_duplicate_check_by_default(monkeypatch):
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron.router_replay import build_router_replay_tensors

    monkeypatch.delenv("NRL_ROUTER_REPLAY_VALIDATE", raising=False)
    RouterReplay.clear_global_router_replay_instances()

    class DummyRouter(torch.nn.Module):
        def __init__(self, replay, layer_number):
            super().__init__()
            self.router_replay = replay
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_layers=1,
                moe_layer_freq=[1],
                num_moe_experts=8,
            )
            self.router_1 = DummyRouter(RouterReplay(), layer_number=1)

    try:
        model = DummyModel()
        routed_experts = torch.tensor([[[0, 0]]], dtype=torch.int32)

        replay_tensors = build_router_replay_tensors(model, routed_experts)

        assert torch.equal(replay_tensors[0], routed_experts[:, 0, :].to(torch.long))
    finally:
        RouterReplay.clear_global_router_replay_instances()


@pytest.mark.mcore
def test_build_router_replay_tensors_rejects_duplicate_topk_rows_when_enabled(
    monkeypatch,
):
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from nemo_rl.models.megatron.router_replay import build_router_replay_tensors

    monkeypatch.setenv("NRL_ROUTER_REPLAY_VALIDATE", "1")
    RouterReplay.clear_global_router_replay_instances()

    class DummyRouter(torch.nn.Module):
        def __init__(self, replay, layer_number):
            super().__init__()
            self.router_replay = replay
            self.layer_number = layer_number

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_layers=1,
                moe_layer_freq=[1],
                num_moe_experts=8,
            )
            self.router_1 = DummyRouter(RouterReplay(), layer_number=1)

    try:
        model = DummyModel()
        routed_experts = torch.tensor([[[0, 0]]], dtype=torch.int32)

        with pytest.raises(ValueError, match="duplicate expert ids"):
            build_router_replay_tensors(model, routed_experts)
    finally:
        RouterReplay.clear_global_router_replay_instances()
