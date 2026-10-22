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

"""PPO must release completed routing graphs when yielding a shared GPU."""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


def test_ppo_offload_releases_routing_before_model_storage(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    parameter = model[0].weight
    parameter.grad = torch.ones_like(parameter)
    model[0].token_dispatcher = SimpleNamespace(
        probs=parameter.softmax(-1),
        routing_map=torch.ones(2, 2, dtype=torch.bool),
        reversed_local_input_permutation_mapping=torch.arange(4),
        local_probs=torch.ones(2, 2),
        local_map=torch.arange(4),
    )
    refs = [weakref.ref(t) for t in vars(model[0].token_dispatcher).values()]
    weights, gradient = parameter.detach().clone(), parameter.grad
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = model
    worker.cfg = {"megatron_cfg": {"clear_memory_caches_before_refit": False}}
    worker.finalize_async_save = Mock()
    worker._uses_mxfp8_overlap_shared_param_buffer = lambda: False
    worker.offload_before_refit = Mock()
    events = []

    def move_model(module, device, **kwargs):
        # GPU offload alone cannot find non-Module dispatcher attributes.
        assert all(ref() is None for ref in refs)
        assert device == "cpu"
        assert not module.training
        assert module[0].weight.grad is gradient
        torch.testing.assert_close(module[0].weight, weights, rtol=0, atol=0)
        events.append(kwargs)
        return module

    worker.move_model = move_model

    class AllocatorWakeup:
        def cuda(self):
            return self

    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: AllocatorWakeup())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args, **kwargs: 0)
    from nemo_rl.models.policy import lm_policy

    monkeypatch.setattr(lm_policy.ray, "get", lambda result: result)
    policy = object.__new__(Policy)
    policy.cfg = {"megatron_cfg": {"enabled": True}}
    policy.worker_group = SimpleNamespace(
        run_all_workers_single_data=lambda method, **kwargs: [
            getattr(worker, method)(**kwargs)
        ]
    )
    policy.offload_to_cpu()
    assert events == [{"move_params": True, "move_grads": True}]
    worker.finalize_async_save.assert_called_once()
    worker.offload_before_refit.assert_called_once()
    assert model[0].weight.grad is gradient
    torch.testing.assert_close(gradient, torch.ones_like(gradient), rtol=0, atol=0)


def test_non_megatron_ppo_offload_keeps_existing_worker_api(monkeypatch):
    from nemo_rl.models.policy import lm_policy

    monkeypatch.setattr(lm_policy.ray, "get", lambda result: result)
    policy = object.__new__(Policy)
    policy.cfg = {"megatron_cfg": {"enabled": False}}
    policy.worker_group = Mock()
    policy.worker_group.run_all_workers_single_data.return_value = []
    policy.offload_to_cpu()
    policy.worker_group.run_all_workers_single_data.assert_called_once_with(
        "offload_after_refit"
    )
