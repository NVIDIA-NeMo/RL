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
import pytest
import torch
from torch import nn

from nemo_rl.models.policy.workers.base_policy_worker import (
    AbstractPolicyWorker,
    first_parameter_device,
)


class FakePolicyWorker(AbstractPolicyWorker):
    """Minimal stand-in exposing only what the onload guard reads.

    The real workers are Ray actors that need a GPU to construct, so the guard
    is exercised against the one attribute it touches.
    """

    def __init__(self, model):
        self.model = model


def _cpu_model() -> nn.Module:
    return nn.Linear(2, 2, device="cpu")


def _offloaded_worker() -> FakePolicyWorker:
    return FakePolicyWorker(_cpu_model())


def test_first_parameter_device_single_module():
    assert first_parameter_device(_cpu_model()) == torch.device("cpu")


def test_first_parameter_device_module_list():
    # The Megatron worker holds a list of virtual pipeline chunks.
    chunks = [_cpu_model(), _cpu_model()]
    assert first_parameter_device(chunks) == torch.device("cpu")


def test_first_parameter_device_no_parameters():
    assert first_parameter_device(nn.Identity()) is None


def test_raises_when_model_is_offloaded():
    worker = _offloaded_worker()

    with pytest.raises(RuntimeError) as excinfo:
        worker._assert_model_onloaded("get_logprobs", "prepare_for_lp_inference")

    message = str(excinfo.value)
    assert "FakePolicyWorker.get_logprobs()" in message
    assert "prepare_for_lp_inference()" in message
    assert "illegal memory access" in message


def test_raises_for_module_list():
    worker = FakePolicyWorker([_cpu_model(), _cpu_model()])

    with pytest.raises(RuntimeError, match="prepare_for_training"):
        worker._assert_model_onloaded("train", "prepare_for_training")


def test_allows_offloaded_params_when_cpu_offload_is_enabled():
    # FSDP cpu_offload keeps parameters on CPU during compute by design.
    worker = _offloaded_worker()

    worker._assert_model_onloaded(
        "train",
        "prepare_for_training",
        params_may_be_offloaded=True,
    )


def test_allows_onloaded_model():
    # 'meta' stands in for a non-CPU device so the check runs without a GPU.
    worker = FakePolicyWorker(nn.Linear(2, 2, device="meta"))

    worker._assert_model_onloaded("train", "prepare_for_training")


def test_allows_model_without_parameters():
    worker = FakePolicyWorker(nn.Identity())

    worker._assert_model_onloaded("train", "prepare_for_training")
