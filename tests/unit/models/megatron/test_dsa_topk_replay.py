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


def _model_config(**overrides):
    values = {
        "num_layers": 8,
        "dsa_indexer_topk": 2,
        "dsa_indexer_topk_freq": 4,
        "dsa_indexer_skip_topk_offset": 3,
        "dsa_indexer_loss_coeff": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dsa_topk_replay_config_uses_discriminated_models():
    from pydantic import ValidationError

    from nemo_rl.models.policy import (
        DSATopKReplayConfig,
        DSATopKReplayConfigDisabled,
        coerce_dsa_topk_replay_config,
    )

    enabled = coerce_dsa_topk_replay_config({"enabled": True, "layer_ids": [6, 0, 2]})
    disabled = coerce_dsa_topk_replay_config({"enabled": False})

    assert isinstance(enabled, DSATopKReplayConfig)
    assert enabled.layer_ids == [6, 0, 2]
    assert isinstance(disabled, DSATopKReplayConfigDisabled)
    with pytest.raises(ValidationError, match="layer_ids"):
        coerce_dsa_topk_replay_config({"enabled": True})


def _fake_dsa(layer_number, config, *, skip_topk=False, tp_size=1, tp_rank=0):
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAttention,
    )

    class FakeGroup:
        def size(self):
            return tp_size

        def rank(self):
            return tp_rank

    module = DSAttention.__new__(DSAttention)
    torch.nn.Module.__init__(module)
    module.layer_number = layer_number
    module.skip_topk = skip_topk
    module.config = config
    module.pg_collection = SimpleNamespace(tp=FakeGroup())
    return module


class _Chunk(torch.nn.Module):
    def __init__(self, config, layer_numbers):
        super().__init__()
        self.config = config
        for layer_number in layer_numbers:
            self.add_module(f"dsa_{layer_number}", _fake_dsa(layer_number, config))


@pytest.mark.mcore
def test_configure_vllm_for_dsa_topk_replay_records_layer_contract():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        configure_vllm_for_dsa_topk_replay,
        validate_dsa_topk_replay_config,
    )

    config = {
        "dsa_topk_replay": {"enabled": True, "layer_ids": [6, 0, 2]},
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {
                "async_engine": False,
                "pipeline_parallel_size": 1,
                "enforce_eager": True,
                "enable_prefix_caching": True,
            },
            "vllm_kwargs": {},
        },
        "megatron_cfg": {
            "enabled": True,
            "model_overrides": {"dsa_indexer_loss_coeff": 0.0},
        },
    }

    configure_vllm_for_dsa_topk_replay(config)
    validate_dsa_topk_replay_config(config)

    generation = config["generation"]
    assert generation["_dsa_topk_replay_enabled"] is True
    assert generation["_dsa_topk_replay_layer_ids"] == [0, 2, 6]
    assert generation["vllm_kwargs"]["enable_return_routed_experts"] is True
    assert generation["vllm_cfg"]["enable_prefix_caching"] is True


@pytest.mark.mcore
@pytest.mark.parametrize(
    ("config_update", "error_match"),
    [
        (
            {"router_replay": {"enabled": True}},
            "cannot be enabled together",
        ),
        (
            {"megatron_cfg": {"enabled": False}},
            "requires the Megatron policy backend",
        ),
        (
            {
                "megatron_cfg": {
                    "enabled": True,
                    "cuda_graph_impl": "local",
                    "model_overrides": {},
                }
            },
            "cuda_graph_impl=none",
        ),
        (
            {
                "megatron_cfg": {
                    "enabled": True,
                    "model_overrides": {"dsa_indexer_loss_coeff": 0.1},
                }
            },
            "requires dsa_indexer_loss_coeff=0",
        ),
        (
            {"megatron_cfg": {"enabled": True, "model_overrides": {}}},
            "requires dsa_indexer_loss_coeff=0 to be explicitly set",
        ),
        (
            {
                "generation": {
                    "backend": "vllm",
                    "vllm_cfg": {
                        "async_engine": True,
                        "pipeline_parallel_size": 1,
                        "enforce_eager": True,
                    },
                    "vllm_kwargs": {},
                }
            },
            "async_engine=false",
        ),
        (
            {
                "generation": {
                    "backend": "vllm",
                    "vllm_cfg": {
                        "async_engine": False,
                        "pipeline_parallel_size": 1,
                        "enforce_eager": True,
                    },
                    "vllm_kwargs": {
                        "speculative_config": {"num_speculative_tokens": 1}
                    },
                }
            },
            "speculative decoding to be disabled",
        ),
    ],
)
def test_validate_dsa_topk_replay_rejects_unsupported_config(
    config_update, error_match
):
    from nemo_rl.models.megatron.dsa_topk_replay import (
        validate_dsa_topk_replay_config,
    )

    config = {
        "dsa_topk_replay": {"enabled": True, "layer_ids": None},
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {
                "async_engine": False,
                "pipeline_parallel_size": 1,
                "enforce_eager": True,
            },
            "vllm_kwargs": {},
        },
        "megatron_cfg": {
            "enabled": True,
            "model_overrides": {"dsa_indexer_loss_coeff": 0.0},
        },
    }
    config.update(config_update)

    with pytest.raises(ValueError, match=error_match):
        validate_dsa_topk_replay_config(config)


@pytest.mark.mcore
def test_dsa_topk_replay_worker_guard_requires_capture_unless_skipped():
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron.dsa_topk_replay import (
        should_use_dsa_topk_replay,
    )

    data = BatchedDataDict({"input_ids": torch.ones(2, 4, dtype=torch.long)})

    assert not should_use_dsa_topk_replay(
        enabled=False,
        data=data,
        stage="prev-logprob",
        require=True,
    )
    assert not should_use_dsa_topk_replay(
        enabled=True,
        data=data,
        stage="reference-logprob",
        require=False,
    )
    with pytest.raises(
        RuntimeError, match="requires dsa_topk_indices for prev-logprob"
    ):
        should_use_dsa_topk_replay(
            enabled=True,
            data=data,
            stage="prev-logprob",
            require=True,
        )

    data["dsa_topk_indices"] = torch.zeros(2, 4, 3, 2, dtype=torch.int16)
    assert should_use_dsa_topk_replay(
        enabled=True,
        data=data,
        stage="prev-logprob",
        require=True,
    )


@pytest.mark.mcore
def test_build_assignments_maps_zero_based_subset_across_pp_chunks():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        build_dsa_topk_replay_assignments,
    )

    config = _model_config()
    # Compute layers for offset=3/freq=4 are 1, 2, 3, and 7.  Supply chunks in
    # reverse pipeline order to ensure mapping uses global layer_number.
    chunks = [_Chunk(config, [7]), _Chunk(config, [1, 2, 3])]
    payload = torch.arange(5 * 3 * 2, dtype=torch.int16).reshape(5, 3, 2)

    assignments = build_dsa_topk_replay_assignments(
        chunks, payload, layer_ids=[6, 0, 2]
    )

    by_layer = {module.layer_number: replay for module, replay in assignments}
    assert sorted(by_layer) == [1, 3, 7]
    assert torch.equal(by_layer[1], payload[None, :, 0, :])
    assert torch.equal(by_layer[3], payload[None, :, 1, :])
    assert torch.equal(by_layer[7], payload[None, :, 2, :])


@pytest.mark.mcore
def test_build_assignments_excludes_skip_layers_and_mtp_subtree():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        build_dsa_topk_replay_assignments,
    )

    config = _model_config(
        num_layers=3,
        dsa_indexer_topk_freq=2,
        dsa_indexer_skip_topk_offset=1,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.compute = _fake_dsa(1, config)
            self.skip = _fake_dsa(2, config, skip_topk=True)
            self.mtp = torch.nn.Module()
            self.mtp.is_mtp_layer = True
            self.mtp.compute = _fake_dsa(3, config)

    model = Model()
    payload = torch.arange(4 * 2 * 2, dtype=torch.int32).reshape(4, 2, 2)

    assignments = build_dsa_topk_replay_assignments(model, payload)

    assert [(module.layer_number, tensor.shape) for module, tensor in assignments] == [
        (1, torch.Size([1, 4, 2]))
    ]


@pytest.mark.mcore
def test_build_assignments_rejects_skip_layer_in_explicit_subset():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        build_dsa_topk_replay_assignments,
    )

    config = _model_config()
    model = _Chunk(config, [1])
    payload = torch.zeros(4, 1, 2, dtype=torch.int16)

    with pytest.raises(ValueError, match="top-k-computing DSA layers"):
        build_dsa_topk_replay_assignments(model, payload, layer_ids=[3])


@pytest.mark.mcore
def test_build_assignments_rejects_missing_local_dsa_but_allows_remote_subset():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        build_dsa_topk_replay_assignments,
    )

    config = _model_config(num_layers=2, dsa_indexer_topk_freq=1)

    class LocalLayer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number

    class Model(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.config = config
            self.layer = LocalLayer(layer_number)

    payload = torch.zeros(4, 1, 2, dtype=torch.int16)

    with pytest.raises(ValueError, match="selected local DSA layers"):
        build_dsa_topk_replay_assignments(Model(1), payload, layer_ids=[0])

    # A PP stage containing only layer 2 has no work when the subset selected
    # layer 1; an empty assignment list is valid in that case.
    assert build_dsa_topk_replay_assignments(Model(2), payload, layer_ids=[0]) == []


@pytest.mark.mcore
def test_indices_for_selector_takes_contiguous_sequence_parallel_slice():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        DSATopKReplayState,
        _ActiveReplay,
        _indices_for_selector,
    )

    config = _model_config(dsa_indexer_topk_freq=1)
    module = _fake_dsa(1, config, tp_size=2, tp_rank=1)
    target = torch.arange(1 * 8 * 2, dtype=torch.int16).reshape(1, 8, 2)
    state = DSATopKReplayState(layer_number=1)
    active = _ActiveReplay(module=module, state=state, target=target)
    q = torch.empty(4, 1, 1, 1)

    actual = _indices_for_selector(active, q, index_topk=2)

    assert actual.dtype == torch.int32
    assert torch.equal(actual, target[:, 4:, :].to(torch.int32))


@pytest.mark.mcore
def test_replay_state_uses_fifo_for_backward_recompute():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        DSATopKReplayAction,
        DSATopKReplayState,
        _ActiveReplay,
        _complete_active_replay,
        _target_for_state,
    )

    state = DSATopKReplayState(
        layer_number=1,
        action=DSATopKReplayAction.REPLAY_FORWARD,
        target_topk_indices=torch.tensor([[[0, 1]]]),
    )
    module = SimpleNamespace()
    first = torch.tensor([[[1, 0]]], dtype=torch.int32)
    second = torch.tensor([[[2, 0]]], dtype=torch.int32)

    _complete_active_replay(
        _ActiveReplay(module=module, state=state, target=first, effective_indices=first)
    )
    _complete_active_replay(
        _ActiveReplay(
            module=module, state=state, target=second, effective_indices=second
        )
    )
    state.action = DSATopKReplayAction.REPLAY_BACKWARD

    assert torch.equal(_target_for_state(state), first)
    _complete_active_replay(
        _ActiveReplay(module=module, state=state, target=first, effective_indices=first)
    )
    assert torch.equal(_target_for_state(state), second)


@pytest.mark.mcore
def test_switching_replay_subset_clears_deselected_module_fifo():
    from nemo_rl.models.megatron.dsa_topk_replay import (
        DSATopKReplayAction,
        _state_for_module,
        set_dsa_topk_replay_forward,
    )

    config = _model_config(num_layers=2, dsa_indexer_topk_freq=1)
    model = _Chunk(config, [1, 2])
    payload = torch.zeros(4, 1, 2, dtype=torch.int16)

    set_dsa_topk_replay_forward(model, payload, layer_ids=[0])
    first_module = model.dsa_1
    first_state = _state_for_module(first_module)
    first_state.replay_backward_list.append(torch.tensor([[[0, 1]]], dtype=torch.int32))

    set_dsa_topk_replay_forward(model, payload, layer_ids=[1])

    assert first_state.action is None
    assert first_state.target_topk_indices is None
    assert first_state.replay_backward_list == []
    assert _state_for_module(model.dsa_2).action == DSATopKReplayAction.REPLAY_FORWARD


@pytest.mark.mcore
def test_patched_selectors_canonicalize_fused_replay_and_bypass_full_fused_path():
    from megatron.core.transformer.experimental_attention_variant import (
        dsa,
        dsa_kernels,
    )

    from nemo_rl.models.megatron.dsa_topk_replay import (
        _ACTIVE_REPLAY,
        DSATopKReplayState,
        _ActiveReplay,
        _install_dsa_topk_replay_patch,
    )

    _install_dsa_topk_replay_patch()
    config = _model_config(dsa_indexer_topk=3, dsa_indexer_topk_freq=1)
    module = _fake_dsa(1, config)
    state = DSATopKReplayState(layer_number=1)
    target = torch.tensor([[[2, -1, 0], [2, 0, 1]]], dtype=torch.int16)
    q = torch.ones(2, 1, 1, 1)
    k = torch.ones(3, 1, 1)
    weights = torch.ones(2, 1, 1)

    fused_active = _ActiveReplay(module=module, state=state, target=target)
    token = _ACTIVE_REPLAY.set(fused_active)
    try:
        fused_indices, topk_length = dsa_kernels.run_fused_qk_topk(
            None, q, k, weights, 3, None, None, 1
        )
        assert dsa_kernels.run_fused_dsa_attention() is None
    finally:
        _ACTIVE_REPLAY.reset(token)

    expected_fused = torch.tensor([[[0, 2, -1], [0, 1, 2]]], dtype=torch.int32)
    assert torch.equal(fused_indices, expected_fused)
    assert torch.equal(topk_length, torch.tensor([[2, 3]], dtype=torch.int32))
    assert torch.equal(fused_active.effective_indices, fused_indices)

    naive_active = _ActiveReplay(module=module, state=state, target=target)
    token = _ACTIVE_REPLAY.set(naive_active)
    try:
        scores, naive_indices = dsa.fused_qk_topk_naive(q, k, weights, 3)
    finally:
        _ACTIVE_REPLAY.reset(token)

    assert scores.numel() == 0
    assert torch.equal(naive_indices, target.to(torch.int32))


@pytest.mark.mcore
def test_naive_selector_uses_native_topk_for_all_missing_row():
    from megatron.core.transformer.experimental_attention_variant import dsa

    from nemo_rl.models.megatron.dsa_topk_replay import (
        _ACTIVE_REPLAY,
        DSATopKReplayState,
        _ActiveReplay,
        _install_dsa_topk_replay_patch,
    )

    _install_dsa_topk_replay_patch()
    config = _model_config(dsa_indexer_topk_freq=1)
    module = _fake_dsa(1, config)
    state = DSATopKReplayState(layer_number=1)
    target = torch.tensor([[[-1, -1], [1, 0]]], dtype=torch.int16)
    active = _ActiveReplay(module=module, state=state, target=target)
    q = torch.tensor([[[[1.0]]], [[[2.0]]]])
    k = torch.tensor([[[1.0]], [[2.0]]])
    weights = torch.ones(2, 1, 1)

    token = _ACTIVE_REPLAY.set(active)
    try:
        _, replayed = dsa.fused_qk_topk_naive(q, k, weights, 2)
    finally:
        _ACTIVE_REPLAY.reset(token)

    assert torch.all(replayed[0, 0] >= 0)
    assert torch.equal(replayed[0, 1], torch.tensor([1, 0], dtype=torch.int32))
    assert torch.equal(active.effective_indices, replayed)


@pytest.mark.mcore
def test_naive_selector_pads_short_native_fallback_to_replay_width():
    from megatron.core.transformer.experimental_attention_variant import dsa

    from nemo_rl.models.megatron.dsa_topk_replay import (
        _ACTIVE_REPLAY,
        DSATopKReplayState,
        _ActiveReplay,
        _install_dsa_topk_replay_patch,
    )

    _install_dsa_topk_replay_patch()
    config = _model_config(dsa_indexer_topk=4, dsa_indexer_topk_freq=1)
    module = _fake_dsa(1, config)
    state = DSATopKReplayState(layer_number=1)
    target = torch.tensor([[[-1, -1, -1, -1], [1, 0, -1, -1]]], dtype=torch.int16)
    active = _ActiveReplay(module=module, state=state, target=target)
    q = torch.tensor([[[[1.0]]], [[[2.0]]]])
    k = torch.tensor([[[1.0]], [[2.0]]])
    weights = torch.ones(2, 1, 1)

    token = _ACTIVE_REPLAY.set(active)
    try:
        _, replayed = dsa.fused_qk_topk_naive(q, k, weights, 4)
    finally:
        _ACTIVE_REPLAY.reset(token)

    assert replayed.shape == target.shape
    assert torch.all(replayed[0, 0, :2] >= 0)
    assert torch.equal(replayed[0, 0, 2:], torch.tensor([-1, -1]))
    assert torch.equal(replayed[0, 1], target[0, 1].to(torch.int32))
    assert torch.equal(active.effective_indices, replayed)


@pytest.mark.mcore
def test_value_validation_accepts_int16_suffix_and_rejects_duplicates(monkeypatch):
    from nemo_rl.models.megatron.dsa_topk_replay import _validate_replay_tensor

    monkeypatch.setenv("NRL_DSA_TOPK_REPLAY_VALIDATE", "1")
    _validate_replay_tensor(
        torch.tensor([[[0, 1, -1]]], dtype=torch.int16),
        layer_number=1,
        payload_idx=0,
    )

    with pytest.raises(ValueError, match="duplicate valid key ids"):
        _validate_replay_tensor(
            torch.tensor([[[0, 0, -1]]], dtype=torch.int16),
            layer_number=1,
            payload_idx=0,
        )
