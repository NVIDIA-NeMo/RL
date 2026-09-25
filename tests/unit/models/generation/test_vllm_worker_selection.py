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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def generation_config(request):
    return {
        "backend": "vllm",
        "model_name": "test-model",
        "dtype": "bfloat16",
        "max_new_tokens": 5,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 1024,
            "async_engine": request.param,
            "skip_tokenizer_init": False,
            "kv_cache_dtype": "auto",
        },
        "vllm_kwargs": {},
        "quant_cfg": "NVFP4",
    }


@pytest.mark.parametrize("explicit", [False, True])
def test_vllm_quant_worker_selection(generation_config, explicit: bool) -> None:
    async_engine = generation_config["vllm_cfg"]["async_engine"]
    expected = (
        "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker"
        if async_engine
        else "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker"
    )
    if explicit:
        generation_config["worker_extension_cls_fqn"] = expected
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    cluster.num_gpus_per_node = 1
    with (
        patch(
            "nemo_rl.models.generation.vllm.vllm_generation.RayWorkerBuilder"
        ) as builder,
        patch("nemo_rl.models.generation.vllm.vllm_generation.RayWorkerGroup") as group,
        patch(
            "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
            return_value=[None],
        ),
        nullcontext()
        if explicit
        else pytest.warns(UserWarning, match="setting it automatically"),
    ):
        group.return_value.dp_size = 1
        VllmGeneration(cluster, generation_config, defer_model_load=True)
    assert builder.call_args.args[0] == expected
    assert builder.call_args.args[1]["worker_extension_cls_fqn"] == expected
    assert builder.call_args.kwargs == (
        {"defer_model_load": True} if async_engine else {}
    )
    assert generation_config["worker_extension_cls_fqn"] == expected


def test_vllm_rejects_quant_worker_for_wrong_engine_before_placement(
    generation_config,
) -> None:
    async_engine = generation_config["vllm_cfg"]["async_engine"]
    generation_config["worker_extension_cls_fqn"] = (
        "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker"
        if async_engine
        else "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker"
    )
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    cluster.num_gpus_per_node = 1
    with (
        patch(
            "nemo_rl.models.generation.vllm.vllm_generation.RayWorkerBuilder"
        ) as builder,
        pytest.raises(ValueError, match="quant_cfg requires"),
    ):
        VllmGeneration(cluster, generation_config, defer_model_load=True)
    builder.assert_not_called()
    cluster._init_placement_groups.assert_not_called()


@pytest.mark.parametrize("custom", [False, True])
def test_vllm_unquantized_worker_selection(generation_config, custom: bool) -> None:
    generation_config["quant_cfg"] = None
    expected = (
        "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
        if generation_config["vllm_cfg"]["async_engine"]
        else "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
    )
    if custom:
        expected = "tests.extensions.CustomGenerationWorker"
        generation_config["worker_extension_cls_fqn"] = expected
    cluster = MagicMock()
    cluster.world_size.return_value = 1
    cluster.num_gpus_per_node = 1
    with (
        patch.dict(
            "nemo_rl.distributed.ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY",
            {expected: "python"},
        ),
        patch(
            "nemo_rl.models.generation.vllm.vllm_generation.RayWorkerBuilder"
        ) as builder,
        patch("nemo_rl.models.generation.vllm.vllm_generation.RayWorkerGroup") as group,
        patch(
            "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
            return_value=[None],
        ),
    ):
        group.return_value.dp_size = 1
        VllmGeneration(cluster, generation_config, defer_model_load=True)
    assert builder.call_args.args[0] == expected
