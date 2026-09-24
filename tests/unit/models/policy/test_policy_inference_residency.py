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

"""CPU-only contract tests for post-logprob policy parameter residency."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _megatron_worker_impl():
    module = pytest.importorskip("nemo_rl.models.policy.workers.megatron_policy_worker")
    return module.MegatronPolicyWorkerImpl


def _dtensor_worker_impl():
    module = pytest.importorskip(
        "nemo_rl.models.policy.workers.dtensor_policy_worker_v2"
    )
    return module.DTensorPolicyWorkerV2Impl


def test_megatron_residency_checks_buffer_storage_not_only_parameter_device():
    class FakeDistributedDataParallel:
        def __init__(self):
            parameter = MagicMock()
            parameter.is_cuda = True
            self._parameters = [parameter]
            param_data = MagicMock()
            param_data.is_cuda = True
            param_data.untyped_storage.return_value.nbytes.return_value = 0
            self.buffers = [SimpleNamespace(param_data=param_data)]
            self.expert_parallel_buffers = []

        def parameters(self):
            return self._parameters

    worker = object.__new__(_megatron_worker_impl())
    worker.model = FakeDistributedDataParallel()

    with patch(
        "nemo_rl.models.policy.workers.megatron_policy_worker.DistributedDataParallel",
        FakeDistributedDataParallel,
    ):
        result = worker._policy_param_residency()

    assert result == {"params_resident_on_cuda": False, "checked_units": 2}


def test_megatron_residency_falls_back_when_ddp_has_no_param_buffer():
    class FakeDistributedDataParallel:
        def __init__(self):
            parameter = MagicMock()
            parameter.is_cuda = True
            parameter.untyped_storage.return_value.nbytes.return_value = 1024
            self._parameters = [parameter]
            self.buffers = [SimpleNamespace(param_data=None)]
            self.expert_parallel_buffers = []

        def parameters(self):
            return self._parameters

    worker = object.__new__(_megatron_worker_impl())
    worker.model = FakeDistributedDataParallel()

    with patch(
        "nemo_rl.models.policy.workers.megatron_policy_worker.DistributedDataParallel",
        FakeDistributedDataParallel,
    ):
        result = worker._policy_param_residency()

    assert result == {"params_resident_on_cuda": True, "checked_units": 1}


@pytest.mark.parametrize("keep_params_for_training", [False, True])
def test_megatron_finish_inference_optionally_retains_params(
    keep_params_for_training,
):
    worker = object.__new__(_megatron_worker_impl())
    worker.model = MagicMock()
    parameter = MagicMock()
    parameter.is_cuda = keep_params_for_training
    parameter.untyped_storage.return_value.nbytes.return_value = (
        1024 if keep_params_for_training else 0
    )
    worker.model.parameters.return_value = [parameter]
    worker.move_model = MagicMock(return_value=worker.model)

    with (
        patch(
            "nemo_rl.models.policy.workers.megatron_policy_worker.gc.collect"
        ) as collect,
        patch(
            "nemo_rl.models.policy.workers.megatron_policy_worker.torch.cuda.empty_cache"
        ) as empty_cache,
    ):
        result = worker.finish_inference(
            keep_params_for_training=keep_params_for_training
        )

    if keep_params_for_training:
        worker.move_model.assert_not_called()
    else:
        worker.move_model.assert_called_once_with(
            worker.model,
            "cpu",
            move_params=True,
            move_grads=False,
        )
    worker.model.eval.assert_called_once_with()
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    assert result["params_resident_on_cuda"] is keep_params_for_training
    assert result["checked_units"] == 1


@pytest.mark.parametrize("keep_params_for_training", [False, True])
def test_dtensor_finish_inference_optionally_retains_params(
    keep_params_for_training,
):
    worker = object.__new__(_dtensor_worker_impl())
    worker.model = MagicMock()
    worker.cpu_offload = False
    parameter = MagicMock()
    local_parameter = MagicMock()
    local_parameter.is_cuda = keep_params_for_training
    local_parameter.untyped_storage.return_value.nbytes.return_value = (
        1024 if keep_params_for_training else 0
    )
    parameter.to_local.return_value = local_parameter
    worker.model.parameters.return_value = [parameter]
    worker.move_to_cpu = MagicMock(return_value=worker.model)

    with (
        patch(
            "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.gc.collect"
        ) as collect,
        patch(
            "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.torch.cuda.empty_cache"
        ) as empty_cache,
    ):
        result = worker.finish_inference(
            keep_params_for_training=keep_params_for_training
        )

    if keep_params_for_training:
        worker.move_to_cpu.assert_not_called()
    else:
        worker.move_to_cpu.assert_called_once_with(worker.model)
    worker.model.eval.assert_called_once_with()
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    assert result["params_resident_on_cuda"] is keep_params_for_training
    assert result["checked_units"] == 1
