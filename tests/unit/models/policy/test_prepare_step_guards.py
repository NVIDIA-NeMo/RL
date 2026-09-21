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
import ast
from pathlib import Path

import pytest
import torch
from torch import nn

import nemo_rl.models.policy.workers as policy_workers
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


def test_fresh_worker_is_not_parked():
    assert FakePolicyWorker(_cpu_model())._training_state_parked is False


def test_training_state_check_passes_when_not_parked():
    worker = FakePolicyWorker(nn.Linear(2, 2, device="meta"))

    worker._assert_training_state_restored("train")


def test_raises_when_training_state_was_parked():
    """prepare_for_lp_inference leaves params on GPU but parks grads/optimizer.

    The device check cannot see that, so it is the flag that has to.
    """
    worker = FakePolicyWorker(nn.Linear(2, 2, device="meta"))
    # What prepare_for_lp_inference(keep_train_buffers=False) records.
    worker._training_state_parked = True

    # The device check still passes: parameters are not on CPU.
    worker._assert_model_onloaded("train", "prepare_for_training")

    with pytest.raises(RuntimeError) as excinfo:
        worker._assert_training_state_restored("train")

    message = str(excinfo.value)
    assert "FakePolicyWorker.train()" in message
    assert "prepare_for_lp_inference()" in message
    assert "prepare_for_training()" in message


def test_restoring_training_state_clears_the_error():
    worker = FakePolicyWorker(nn.Linear(2, 2, device="meta"))
    worker._training_state_parked = True
    # What prepare_for_training() records once it has onloaded everything.
    worker._training_state_parked = False

    worker._assert_training_state_restored("train")


@pytest.mark.parametrize(
    "worker_module",
    [
        "dtensor_policy_worker",
        "dtensor_policy_worker_v2",
        "megatron_policy_worker",
    ],
)
def test_every_worker_that_parks_also_restores(worker_module):
    """A worker that sets the parked flag must also clear it.

    Forgetting the clear is the dangerous asymmetry: the flag would stay set for
    the life of the worker and every later train step would raise. Checked
    structurally because the real workers are Ray actors that need a GPU.
    """
    workers_dir = Path(policy_workers.__file__).parent
    source = (workers_dir / f"{worker_module}.py").read_text()
    tree = ast.parse(source)

    setters = {"True": set(), "False": set()}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Assign)
                and len(sub.targets) == 1
                and isinstance(sub.targets[0], ast.Attribute)
                and sub.targets[0].attr == "_training_state_parked"
            ):
                setters[str(sub.value.value)].add(node.name)

    assert setters["True"] == {"prepare_for_lp_inference"}, (
        f"{worker_module}: the parked flag should only be set where "
        f"prepare_for_lp_inference parks training state, got {setters['True']}"
    )
    assert setters["False"] == {"prepare_for_training"}, (
        f"{worker_module}: prepare_for_training must clear the parked flag, "
        f"got {setters['False']}"
    )
