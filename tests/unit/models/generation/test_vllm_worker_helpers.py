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

"""Tests for vLLM worker helper functions."""

from unittest.mock import patch

import pytest

from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker
from nemo_rl.models.generation.vllm.worker_utils import (
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
)


def test_worker_merges_configured_env_vars_with_subclass_env_vars() -> None:
    config: VllmConfig = {
        "backend": "vllm",
        "model_name": "test-model",
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
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
            "env_vars": {
                "PYTHONPATH": "/configured/python/path",
                "NTRACE_CUPTI_EARLY_BOOTSTRAP": "1",
            },
        },
    }
    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)

    with patch(
        "nemo_rl.models.generation.vllm.vllm_worker._apply_vllm_patches"
    ) as apply_vllm_patches:
        worker._init_config(
            config,
            bundle_indices=None,
            fraction_of_gpus=1.0,
            seed=None,
            # ModelOpt workers supply these and previously replaced config names.
            extra_env_vars=["VLLM_QUANT_CFG", "PYTHONPATH"],
        )

    assert worker._extra_env_vars == [
        "VLLM_QUANT_CFG",
        "PYTHONPATH",
        "NTRACE_CUPTI_EARLY_BOOTSTRAP",
    ]
    apply_vllm_patches.assert_called_once_with(
        worker.py_executable,
        extra_env_vars=worker._extra_env_vars,
    )


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )
