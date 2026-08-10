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

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.generation.profiling import (
    validate_rollout_profiler_topology,
)
from nemo_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    VllmGenerationWorkerImpl,
)


def _worker_with_profiler():
    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker._rollout_profiler = MagicMock()
    worker._create_engine = MagicMock()
    return worker


def test_engine_creation_drives_profiler_initialization_window():
    worker = _worker_with_profiler()
    token = object()
    worker._rollout_profiler.begin_engine_initialization.return_value = token
    llm_kwargs = {"model": "test-model"}

    worker._create_engine_with_profiler(llm_kwargs)

    worker._rollout_profiler.begin_engine_initialization.assert_called_once_with()
    worker._create_engine.assert_called_once_with(llm_kwargs)
    worker._rollout_profiler.end_engine_initialization.assert_called_once_with(token)


def test_engine_failure_closes_profiler_initialization_window():
    worker = _worker_with_profiler()
    token = object()
    engine_error = RuntimeError("engine failed")
    worker._rollout_profiler.begin_engine_initialization.return_value = token
    worker._create_engine.side_effect = engine_error

    with pytest.raises(RuntimeError, match="engine failed") as exc_info:
        worker._create_engine_with_profiler({})

    assert exc_info.value is engine_error
    worker._rollout_profiler.end_engine_initialization.assert_called_once_with(token)


def test_worker_drives_rollout_profiler_lifecycle():
    worker = _worker_with_profiler()

    worker.begin_rollout_profile(step_id="step2/attempt3")
    worker.finish_rollout_profile()
    worker.abort_rollout_profile(reason="rollout_error")

    worker._rollout_profiler.begin_rollout.assert_called_once_with(
        step_id="step2/attempt3"
    )
    worker._rollout_profiler.finish_rollout.assert_called_once_with()
    worker._rollout_profiler.abort_rollout.assert_called_once_with(
        reason="rollout_error"
    )


@pytest.mark.parametrize(
    (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "async_engine",
    ),
    [(2, 1, 1, False), (1, 2, 1, False), (1, 1, 2, False), (1, 1, 1, True)],
)
def test_rollout_profiler_rejects_unsupported_topology_before_worker_start(
    tensor_parallel_size,
    pipeline_parallel_size,
    expert_parallel_size,
    async_engine,
):
    with pytest.raises(ValueError, match="Synchronous rollout profiling"):
        validate_rollout_profiler_topology(
            class_path="profiler.Plugin",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            async_engine=async_engine,
        )


def test_rollout_profiler_topology_validation_is_inert_when_disabled():
    validate_rollout_profiler_topology(
        class_path="",
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        expert_parallel_size=4,
        async_engine=True,
    )


def test_worker_shutdown_closes_rollout_profiler():
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker._rollout_profiler = MagicMock()
    worker._sparse_refit_receiver = None
    worker.llm = None
    worker.tokenizer = object()

    with (
        patch("nemo_rl.models.generation.vllm.vllm_worker.gc.collect"),
        patch("nemo_rl.models.generation.vllm.vllm_worker.torch.cuda.empty_cache"),
    ):
        assert worker.shutdown() is True

    worker._rollout_profiler.close.assert_called_once_with()
    assert worker.tokenizer is None
