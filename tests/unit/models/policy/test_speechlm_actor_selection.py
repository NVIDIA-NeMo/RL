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

from nemo_rl.distributed.actor_environments import ACTOR_ENVIRONMENTS
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.models.policy.utils import get_automodel_policy_worker_cls


def test_native_speechlm_uses_prepared_driver_environment_only():
    actor = get_automodel_policy_worker_cls({"native_model": {}})
    assert (
        actor
        == "nemo_rl.models.policy.workers.speechlm_policy_worker.SpeechLMPolicyWorker"
    )
    assert ACTOR_ENVIRONMENTS[actor] is None
    assert get_actor_python_env(actor) == PY_EXECUTABLES.SYSTEM
    generic = get_automodel_policy_worker_cls({})
    assert ACTOR_ENVIRONMENTS[generic] == ["automodel"]
    assert ACTOR_ENVIRONMENTS[
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    ] == ["vllm", "nemo_gym"]


def test_native_speechlm_rejects_quantization_before_actor_launch():
    with pytest.raises(ValueError, match="quantization"):
        get_automodel_policy_worker_cls({"native_model": {}, "quant_cfg": {}})


def test_generic_automodel_keeps_quantized_worker_selection():
    actor = get_automodel_policy_worker_cls({"quant_cfg": {}})
    assert (
        actor
        == "nemo_rl.modelopt.models.policy.workers.automodel_quant_policy_worker.AutomodelQuantPolicyWorker"
    )
