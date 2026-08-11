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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from nemo_rl.models.generation.profiling import (
    validate_rollout_profiler_topology,
)
from nemo_rl.models.generation.vllm.vllm_backend import (
    RolloutProfilingVllmWorker,
    configure_rollout_profiler_worker,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    VllmGenerationWorkerImpl,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
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


def test_internal_worker_owns_profiler_lifecycle_with_dense_rank():
    events = []
    config = SimpleNamespace(
        additional_config={
            "nemo_rl_rollout_profiler": {
                "class_path": "profiler.Plugin",
                "rank_prefix": 16,
            }
        },
        events=events,
    )
    profiler = MagicMock()
    profiler.begin_engine_initialization.side_effect = lambda: events.append(
        "profile_begin"
    )

    def record_worker_init(_worker, **kwargs):
        events.append(("worker_init", kwargs["rank"]))

    with (
        patch(
            "nemo_rl.models.generation.vllm.vllm_backend.load_rollout_profiler",
            return_value=profiler,
        ) as load_profiler,
        patch(
            "nemo_rl.models.generation.vllm.vllm_backend.VllmWorker.__init__",
            autospec=True,
            side_effect=record_worker_init,
        ),
        patch(
            "nemo_rl.models.generation.vllm.vllm_backend.VllmWorker.shutdown",
            autospec=True,
        ) as shutdown_worker,
    ):
        worker = RolloutProfilingVllmWorker(
            vllm_config=config,
            local_rank=2,
            rank=2,
            distributed_init_method="tcp://test",
        )

        load_profiler.assert_called_once_with(rank=18)
        assert events == ["profile_begin", ("worker_init", 2)]

        worker.end_rollout_profiler_engine_initialization()
        worker.begin_rollout_profile(step_id="step2/attempt3")
        worker.finish_rollout_profile()
        worker.abort_rollout_profile(reason="rollout_error")
        worker.shutdown()

        shutdown_worker.assert_called_once_with(worker)

    profiler.end_engine_initialization.assert_called_once_with(None)
    profiler.begin_rollout.assert_called_once_with(step_id="step2/attempt3")
    profiler.finish_rollout.assert_called_once_with()
    profiler.abort_rollout.assert_called_once_with(reason="rollout_error")
    profiler.close.assert_called_once_with()


def test_configure_internal_profiler_worker_preserves_additional_config():
    vllm_kwargs = {"additional_config": {"existing": "value"}}

    configure_rollout_profiler_worker(
        vllm_kwargs, class_path="profiler.Plugin", rank_prefix=16
    )

    assert vllm_kwargs["worker_cls"].endswith(".RolloutProfilingVllmWorker")
    assert vllm_kwargs["additional_config"] == {
        "existing": "value",
        "nemo_rl_rollout_profiler": {
            "class_path": "profiler.Plugin",
            "rank_prefix": 16,
        },
    }


def test_configure_internal_profiler_worker_composes_with_nixl():
    vllm_kwargs = {
        "worker_cls": "nemo_rl.models.generation.vllm.vllm_backend.NixlVllmWorker"
    }

    configure_rollout_profiler_worker(
        vllm_kwargs, class_path="profiler.Plugin", rank_prefix=0
    )

    assert vllm_kwargs["worker_cls"].endswith(".RolloutProfilingNixlVllmWorker")


def test_configure_internal_profiler_worker_rejects_other_worker_class():
    with pytest.raises(ValueError, match="cannot be composed"):
        configure_rollout_profiler_worker(
            {"worker_cls": "custom.Worker"},
            class_path="profiler.Plugin",
            rank_prefix=0,
        )


def test_outer_worker_fans_out_profiler_lifecycle_to_internal_workers():
    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker._rollout_profiler = None
    worker._use_internal_rollout_profiler = True
    worker.llm = MagicMock()

    worker.begin_rollout_profile(step_id=4)
    worker.finish_rollout_profile()
    worker.abort_rollout_profile(reason="rollout_error")

    assert worker.llm.collective_rpc.call_args_list == [
        call("begin_rollout_profile", args=tuple(), kwargs={"step_id": 4}),
        call("finish_rollout_profile", args=tuple(), kwargs={}),
        call(
            "abort_rollout_profile",
            args=tuple(),
            kwargs={"reason": "rollout_error"},
        ),
    ]


@pytest.mark.asyncio
async def test_async_outer_worker_awaits_internal_profiler_lifecycle():
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._use_internal_rollout_profiler = True
    worker.llm = SimpleNamespace(collective_rpc=AsyncMock())

    await worker.begin_rollout_profile_async(step_id=4)
    await worker.finish_rollout_profile_async()
    await worker.abort_rollout_profile_async(reason="rollout_error")

    assert worker.llm.collective_rpc.await_args_list == [
        call("begin_rollout_profile", args=tuple(), kwargs={"step_id": 4}),
        call("finish_rollout_profile", args=tuple()),
        call(
            "abort_rollout_profile",
            args=tuple(),
            kwargs={"reason": "rollout_error"},
        ),
    ]


@pytest.mark.parametrize(
    ("async_engine", "expected_method"),
    [(False, "begin_rollout_profile"), (True, "begin_rollout_profile_async")],
)
def test_generation_selects_engine_specific_profiler_rpc(async_engine, expected_method):
    generation = VllmGeneration.__new__(VllmGeneration)
    generation.rollout_profiler_enabled = True
    generation.cfg = {"vllm_cfg": {"async_engine": async_engine}}
    generation.worker_group = MagicMock()
    generation.shutdown = MagicMock()
    generation.worker_group.run_all_workers_single_data.return_value = [object()]

    with patch("nemo_rl.models.generation.vllm.vllm_generation.ray.get") as ray_get:
        generation.begin_rollout_profile(step_id=4)

    generation.worker_group.run_all_workers_single_data.assert_called_once_with(
        expected_method,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        step_id=4,
    )
    ray_get.assert_called_once_with(
        generation.worker_group.run_all_workers_single_data.return_value
    )


@pytest.mark.parametrize(
    (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
    ),
    [(8, 2, 1), (8, 1, 2)],
)
def test_rollout_profiler_rejects_unsupported_topology_before_worker_start(
    tensor_parallel_size,
    pipeline_parallel_size,
    expert_parallel_size,
):
    with pytest.raises(ValueError, match="Rollout profiling"):
        validate_rollout_profiler_topology(
            class_path="profiler.Plugin",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
        )


def test_rollout_profiler_accepts_tp8_topology():
    validate_rollout_profiler_topology(
        class_path="profiler.Plugin",
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
    )


def test_rollout_profiler_topology_validation_is_inert_when_disabled():
    validate_rollout_profiler_topology(
        class_path="",
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        expert_parallel_size=4,
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


def test_worker_shutdown_closes_internal_rollout_profilers():
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker._rollout_profiler = None
    worker._use_internal_rollout_profiler = True
    worker._sparse_refit_receiver = None
    llm = MagicMock()
    worker.llm = llm
    worker.tokenizer = object()

    with (
        patch("nemo_rl.models.generation.vllm.vllm_worker.gc.collect"),
        patch("nemo_rl.models.generation.vllm.vllm_worker.torch.cuda.empty_cache"),
    ):
        assert worker.shutdown() is True

    assert llm.collective_rpc.call_args_list == [
        call("close_rollout_profiler", args=tuple()),
        call("cleanup", args=tuple()),
    ]
    assert worker.llm is None
