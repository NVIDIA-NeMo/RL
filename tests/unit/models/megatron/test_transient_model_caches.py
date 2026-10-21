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

"""CPU regressions for GPU tensors retained outside registered module state."""

import gc
import weakref
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron.common import clear_transient_model_caches

pytestmark = pytest.mark.mcore


def make_model() -> tuple[torch.nn.Module, list[weakref.ReferenceType]]:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model[0].weight.grad = torch.ones_like(model[0].weight)
    # Match MCore: the dispatcher is deliberately NOT a registered nn.Module.
    dispatcher = SimpleNamespace(
        probs=model[0].weight.softmax(dim=-1),
        routing_map=torch.ones(2, 2, dtype=torch.bool),
        reversed_local_input_permutation_mapping=torch.arange(4),
        local_probs=torch.ones(2, 2),
        local_map=torch.arange(4),
        metadata={"num_experts": 2},
    )
    model[0].token_dispatcher = dispatcher
    refs = [
        weakref.ref(value)
        for value in vars(dispatcher).values()
        if isinstance(value, torch.Tensor)
    ]
    return model, refs


def test_clear_releases_plain_dispatcher_tensors_without_changing_training_state():
    model, refs = make_model()
    weights = model[0].weight.detach().clone()
    grad = model[0].weight.grad
    clear_transient_model_caches(model)
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert model[0].token_dispatcher.metadata == {"num_experts": 2}
    torch.testing.assert_close(model[0].weight, weights)
    assert model[0].weight.grad is grad
    torch.testing.assert_close(grad, torch.ones_like(grad))
    # Repeated phase transitions and modules with no dispatcher are safe.
    clear_transient_model_caches(model)
    clear_transient_model_caches(torch.nn.Linear(2, 2))


def test_clear_releases_rotary_lru_tensors(monkeypatch):
    from nemo_rl.models.megatron import common

    @lru_cache(maxsize=32)
    def forward(length):
        return torch.ones(length)

    monkeypatch.setattr(common, "RotaryEmbedding", SimpleNamespace(forward=forward))
    value = forward(32)
    ref = weakref.ref(value)
    del value
    assert ref() is not None
    clear_transient_model_caches(torch.nn.Identity())
    assert ref() is None
    assert forward.cache_info().currsize == 0


@pytest.mark.parametrize(
    "phase", ["policy_inference", "value_inference", "value_training"]
)
def test_completed_ppo_phases_release_routing_before_yielding_gpu(monkeypatch, phase):
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )
    from nemo_rl.models.value.workers.megatron_value_worker import (
        MegatronValueWorkerImpl,
    )

    cls = (
        MegatronPolicyWorkerImpl
        if phase == "policy_inference"
        else MegatronValueWorkerImpl
    )
    worker = object.__new__(cls)
    worker.model, refs = make_model()
    worker.optimizer = None
    worker.optimizer_cpu_offload = True
    weights = worker.model[0].weight.detach().clone()
    grad = worker.model[0].weight.grad
    calls = []

    def move_model(model, device, **kwargs):
        # The old .to/offload implementation cannot discover dispatcher tensors.
        assert all(ref() is None for ref in refs)
        assert device == "cpu"
        assert model[0].weight.grad is grad
        torch.testing.assert_close(model[0].weight, weights)
        calls.append(kwargs)
        return model

    worker.move_model = move_model
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    if phase == "value_training":
        worker.finish_training()
        assert not worker.model.training
    else:
        worker.finish_inference()
    assert calls == [{"move_params": True, "move_grads": phase == "value_training"}]
