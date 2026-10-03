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

import warnings

import pytest

from nemo_rl.models.generation.vllm.utils import resolve_generation_worker_cls
from nemo_rl.models.policy.utils import resolve_policy_worker_cls


@pytest.fixture(
    params=[
        (
            resolve_policy_worker_cls,
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
            "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker.MegatronQuantPolicyWorker",
        ),
        (
            resolve_policy_worker_cls,
            "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2",
            "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker_v2.DTensorQuantPolicyWorkerV2",
        ),
        (
            resolve_generation_worker_cls,
            "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker",
            "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker",
        ),
        (
            resolve_generation_worker_cls,
            "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker",
            "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker",
        ),
    ],
    ids=["megatron", "dtensor-v2", "vllm-sync", "vllm-async"],
)
def worker_resolver(request):
    return request.param


@pytest.mark.parametrize("include_null", [False, True])
def test_quantization_fills_config_once(worker_resolver, include_null: bool) -> None:
    resolve, default_cls, quantized_cls = worker_resolver
    config = {"quant_cfg": "NVFP4"}
    if include_null:
        config["worker_extension_cls_fqn"] = None
    with pytest.warns(UserWarning, match="setting it automatically") as recorded:
        assert resolve(default_cls, config) == quantized_cls
    assert quantized_cls in str(recorded[0].message)
    assert config["worker_extension_cls_fqn"] == quantized_cls
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve(default_cls, config) == quantized_cls


def test_matching_quant_worker_is_accepted_without_warning(worker_resolver) -> None:
    resolve, default_cls, quantized_cls = worker_resolver
    config = {"quant_cfg": "NVFP4", "worker_extension_cls_fqn": quantized_cls}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve(default_cls, config) == quantized_cls


@pytest.mark.parametrize("extension", ["default", "custom", "other_quant_backend"])
def test_quantization_rejects_incompatible_worker(
    worker_resolver, extension: str
) -> None:
    resolve, default_cls, quantized_cls = worker_resolver
    if extension == "default":
        configured_cls = default_cls
    elif extension == "custom":
        configured_cls = "tests.workers.CustomWorker"
    else:
        async_worker = "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker"
        configured_cls = (
            "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker.MegatronQuantPolicyWorker"
            if quantized_cls == async_worker
            else async_worker
        )
    config = {"quant_cfg": "NVFP4", "worker_extension_cls_fqn": configured_cls}
    with pytest.raises(ValueError, match="quant_cfg requires") as exc:
        resolve(default_cls, config)
    assert quantized_cls in str(exc.value)
    assert configured_cls in str(exc.value)
    assert config["worker_extension_cls_fqn"] == configured_cls


@pytest.mark.parametrize("extension", [None, "tests.workers.CustomWorker"])
def test_unquantized_worker_selection(worker_resolver, extension: str | None) -> None:
    resolve, default_cls, _ = worker_resolver
    config = {"quant_cfg": None, "worker_extension_cls_fqn": extension}
    original = config.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve(default_cls, config) == (extension or default_cls)
    assert config == original
