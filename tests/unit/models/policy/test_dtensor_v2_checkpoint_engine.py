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

"""Tests for Automodel DTensor checkpoint-engine weight transfer."""

import pytest
import torch
import torch.nn as nn

try:
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
        DTensorPolicyWorkerV2Impl,
    )

    NEMO_AUTOMODEL_AVAILABLE = True
except ImportError:
    NEMO_AUTOMODEL_AVAILABLE = False


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_dtensor_v2_checkpoint_engine_weight_iterator():
    worker = object.__new__(DTensorPolicyWorkerV2Impl)
    worker.model = nn.Linear(2, 1)
    worker.dtype = torch.float32

    weights = list(DTensorPolicyWorkerV2Impl._checkpoint_engine_weight_iterator(worker))

    assert [name for name, _tensor in weights] == ["weight", "bias"]
    for _name, tensor in weights:
        assert tensor.dtype == torch.float32
        assert tensor.is_contiguous()


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_dtensor_v2_checkpoint_engine_rejects_kv_scales():
    worker = object.__new__(DTensorPolicyWorkerV2Impl)
    worker.model = nn.Linear(2, 1)
    worker.dtype = torch.float32

    with pytest.raises(NotImplementedError, match="FP8 kvcache"):
        DTensorPolicyWorkerV2Impl._checkpoint_engine_weight_iterator(
            worker, kv_scales={"scale": 1.0}
        )


@pytest.mark.automodel
@pytest.mark.skipif(not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available")
def test_dtensor_v2_checkpoint_engine_cpu_offload_hooks():
    worker = object.__new__(DTensorPolicyWorkerV2Impl)
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
        DTensorPolicyWorkerV2Impl._prepare_checkpoint_engine_weight_send(worker)
    assert worker.model == "cuda_model"
    DTensorPolicyWorkerV2Impl._finalize_checkpoint_engine_weight_send(worker)

    assert worker.model == "cpu_model"
    assert calls == [("cuda", "cpu_model"), ("cpu", "cuda_model")]
