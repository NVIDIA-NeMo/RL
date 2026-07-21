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

"""Tests for DTensor checkpoint-engine weight transfer."""

import pytest
import torch


def test_dtensor_checkpoint_engine_weight_iterator():
    from nemo_rl.models.policy.workers.dtensor_policy_worker import (
        DTensorPolicyWorkerImpl,
    )

    worker = object.__new__(DTensorPolicyWorkerImpl)
    worker.model = torch.nn.Linear(2, 1)
    worker.dtype = torch.float32

    weights = list(DTensorPolicyWorkerImpl._checkpoint_engine_weight_iterator(worker))

    assert [name for name, _tensor in weights] == ["weight", "bias"]
    for _name, tensor in weights:
        assert tensor.dtype == torch.float32
        assert tensor.is_contiguous()


def test_dtensor_checkpoint_engine_rejects_kv_scales():
    from nemo_rl.models.policy.workers.dtensor_policy_worker import (
        DTensorPolicyWorkerImpl,
    )

    worker = object.__new__(DTensorPolicyWorkerImpl)
    worker.model = torch.nn.Linear(2, 1)
    worker.dtype = torch.float32

    with pytest.raises(NotImplementedError, match="FP8 kvcache"):
        DTensorPolicyWorkerImpl._checkpoint_engine_weight_iterator(
            worker, kv_scales={"scale": 1.0}
        )


def test_dtensor_checkpoint_engine_cpu_offload_hooks():
    from nemo_rl.models.policy.workers.dtensor_policy_worker import (
        DTensorPolicyWorkerImpl,
    )

    worker = object.__new__(DTensorPolicyWorkerImpl)
    worker.model = "cpu_model"
    worker.cpu_offload = True
    calls = []

    def move_to_cuda(model):
        calls.append(("cuda", model))
        return "cuda_model"

    def move_to_cpu(model):
        calls.append(("cpu", model))
        return "cpu_model"

    worker.move_to_cuda = move_to_cuda
    worker.move_to_cpu = move_to_cpu

    with pytest.warns(UserWarning, match="cpu_offload adds an onload/offload cycle"):
        DTensorPolicyWorkerImpl._prepare_checkpoint_engine_weight_send(worker)
    assert worker.model == "cuda_model"
    DTensorPolicyWorkerImpl._finalize_checkpoint_engine_weight_send(worker)

    assert worker.model == "cpu_model"
    assert calls == [("cuda", "cpu_model"), ("cpu", "cuda_model")]
