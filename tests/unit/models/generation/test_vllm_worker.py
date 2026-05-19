# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, cast

import pytest

from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker


def _worker_with_vllm_cfg(vllm_cfg: dict[str, Any]) -> BaseVllmGenerationWorker:
    config = {
        "model_name": "dummy-model",
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
            "max_model_len": 128,
            "skip_tokenizer_init": True,
            "async_engine": False,
            "precision": "bfloat16",
            "kv_cache_dtype": "auto",
            **vllm_cfg,
        },
    }
    return BaseVllmGenerationWorker(cast(VllmConfig, config), bundle_indices=None)


def test_vllm_sleep_level_defaults_to_level_1():
    worker = _worker_with_vllm_cfg({})

    assert worker.sleep_level == 1


@pytest.mark.parametrize("sleep_level", [0, 1, 2])
def test_vllm_sleep_level_accepts_supported_levels(sleep_level):
    worker = _worker_with_vllm_cfg({"sleep_level": sleep_level})

    assert worker.sleep_level == sleep_level


@pytest.mark.parametrize("sleep_level", [-1, 3, "2", True])
def test_vllm_sleep_level_rejects_unsupported_levels(sleep_level):
    with pytest.raises(ValueError, match=r"vllm_cfg\.sleep_level must be 0, 1, or 2"):
        _worker_with_vllm_cfg({"sleep_level": sleep_level})
