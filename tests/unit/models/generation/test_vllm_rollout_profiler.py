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

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    VllmGenerationWorkerImpl,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.models.profiling import ROLLOUT_PROFILER_METHODS

BACKEND = "nemo_rl.models.generation.vllm.vllm_backend"


def _internal_worker(use_nixl, events, profiler, *, worker_init_error=None):
    """Construct an internal vLLM GPU worker without starting vLLM."""
    # The internal worker imports vLLM; collect this module in the driver first.
    from nemo_rl.models.generation.vllm.vllm_backend import (
        RolloutProfilingNixlVllmWorker,
        RolloutProfilingVllmWorker,
    )

    config = SimpleNamespace(
        additional_config={
            "nemo_rl_rollout_profiler": {
                "class_path": "profiler.Plugin",
                "rank_prefix": 16,
            }
        },
    )
    profiler.begin_engine_initialization.side_effect = (
        lambda: events.append("profile_begin") or "init-token"
    )

    def record_worker_init(_worker, **kwargs):
        events.append(("worker_init", kwargs["rank"]))
        if worker_init_error is not None:
            raise worker_init_error

    with (
        patch(
            f"{BACKEND}.preinit_nixl_from_vllm_config",
            side_effect=lambda _config: events.append("nixl_preinit"),
        ),
        patch(f"{BACKEND}.load_profiler", return_value=profiler) as load_profiler,
        patch(f"{BACKEND}._current_device_uuid", return_value="GPU-test"),
        patch(
            f"{BACKEND}.VllmWorker.__init__",
            autospec=True,
            side_effect=record_worker_init,
        ),
    ):
        worker_class = (
            RolloutProfilingNixlVllmWorker if use_nixl else RolloutProfilingVllmWorker
        )
        worker = worker_class(
            vllm_config=config, local_rank=2, rank=2, distributed_init_method="tcp://"
        )
    load_profiler.assert_called_once_with(
        "profiler.Plugin", rank=18, required_methods=ROLLOUT_PROFILER_METHODS
    )
    return worker


@pytest.mark.vllm
@pytest.mark.parametrize("use_nixl", [False, True])
def test_internal_worker_captures_from_engine_startup_with_dense_rank(use_nixl):
    events, profiler = [], MagicMock()
    worker = _internal_worker(use_nixl, events, profiler)

    # Capture starts in the GPU process before vLLM builds the model and graphs.
    assert events == (["nixl_preinit"] if use_nixl else []) + [
        "profile_begin",
        ("worker_init", 2),
    ]
    with pytest.raises(RuntimeError, match="initialization is still open"):
        worker.full_step_profile("configure_capture", run_id="run")

    worker.end_rollout_profiler_engine_initialization()
    worker.full_step_profile("configure_capture", run_id="run")
    worker.full_step_profile("begin_step", step_id="generation0")
    profiler.begin_phase.return_value = "phase-token"
    worker.full_step_profile("begin_phase", name="generation_epoch")
    worker.full_step_profile("end_phase")
    with patch(f"{BACKEND}.VllmWorker.shutdown", autospec=True) as shutdown_worker:
        worker.shutdown()

    profiler.end_engine_initialization.assert_called_once_with("init-token")
    profiler.configure_capture.assert_called_once_with(
        device_uuid="GPU-test", run_id="run"
    )
    profiler.end_phase.assert_called_once_with("phase-token")
    profiler.close.assert_called_once_with()
    shutdown_worker.assert_called_once_with(worker)
    with pytest.raises(RuntimeError, match="not active"):
        worker.full_step_profile("finish_step")


@pytest.mark.vllm
def test_internal_worker_initialization_failure_closes_profiler():
    events, profiler = [], MagicMock()
    error = RuntimeError("graph capture failed")

    with pytest.raises(RuntimeError, match="graph capture failed"):
        _internal_worker(False, events, profiler, worker_init_error=error)

    assert events == ["profile_begin", ("worker_init", 2)]
    profiler.end_engine_initialization.assert_called_once_with("init-token")
    profiler.close.assert_called_once_with()


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("worker_cls", "expected"),
    [
        (None, "RolloutProfilingVllmWorker"),
        (
            "nemo_rl.models.generation.vllm.vllm_backend.NixlVllmWorker",
            "RolloutProfilingNixlVllmWorker",
        ),
        ("custom.Worker", None),
    ],
)
def test_profiling_worker_composes_only_with_known_worker_classes(worker_cls, expected):
    from nemo_rl.models.generation.vllm.vllm_backend import (
        configure_rollout_profiler_worker,
    )

    vllm_kwargs = {"additional_config": {"existing": "value"}}
    if worker_cls is not None:
        vllm_kwargs["worker_cls"] = worker_cls
    if expected is None:
        with pytest.raises(ValueError, match="cannot be composed"):
            configure_rollout_profiler_worker(
                vllm_kwargs, class_path="profiler.Plugin", rank_prefix=0
            )
        return

    configure_rollout_profiler_worker(
        vllm_kwargs, class_path="profiler.Plugin", rank_prefix=16
    )

    assert vllm_kwargs["worker_cls"] == f"{BACKEND}.{expected}"
    assert vllm_kwargs["additional_config"] == {
        "existing": "value",
        "nemo_rl_rollout_profiler": {
            "class_path": "profiler.Plugin",
            "rank_prefix": 16,
        },
    }


def test_model_owner_forwards_capture_control_to_internal_workers():
    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker.llm = MagicMock()

    with pytest.raises(RuntimeError, match="unavailable"):
        worker.full_step_profile("begin_step", step_id=4)
    worker._rollout_profiler_class = "profiler.Plugin"
    worker.full_step_profile("begin_step", step_id=4)

    worker.llm.collective_rpc.assert_called_once_with(
        "full_step_profile",
        args=tuple(),
        kwargs={"command": "begin_step", "step_id": 4},
    )


@pytest.mark.asyncio
async def test_async_model_owner_awaits_internal_capture_control():
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._rollout_profiler_class = "profiler.Plugin"
    worker.llm = SimpleNamespace(collective_rpc=AsyncMock())

    await worker.full_step_profile_async("end_phase")

    worker.llm.collective_rpc.assert_awaited_once_with(
        "full_step_profile", args=tuple(), kwargs={"command": "end_phase"}
    )


@pytest.mark.parametrize(
    ("enabled", "async_engine", "expected_method"),
    [
        (True, False, "full_step_profile"),
        (True, True, "full_step_profile_async"),
        (False, False, None),
    ],
)
def test_generation_routes_capture_control_to_model_owners(
    enabled, async_engine, expected_method
):
    generation = VllmGeneration.__new__(VllmGeneration)
    generation._rollout_profiler_enabled = enabled
    generation.cfg = {"vllm_cfg": {"async_engine": async_engine}}
    generation.worker_group = MagicMock()
    generation.shutdown = MagicMock()

    with patch("nemo_rl.models.generation.vllm.vllm_generation.ray.get") as ray_get:
        if expected_method is None:
            with pytest.raises(RuntimeError, match="NRL_ROLLOUT_PROFILER_CLASS"):
                generation.full_step_profile("finish_step")
            return
        generation.full_step_profile("finish_step")

    generation.worker_group.run_all_workers_single_data.assert_called_once_with(
        expected_method,
        run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        command="finish_step",
    )
    ray_get.assert_called_once()


@pytest.mark.parametrize("close_fails", [False, True])
def test_worker_shutdown_closes_profilers_before_cleanup(close_fails, capsys):
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker._rollout_profiler_class = "profiler.Plugin"
    worker._sparse_refit_receiver = None
    llm = MagicMock()
    if close_fails:
        llm.collective_rpc.side_effect = [RuntimeError("close failed"), None]
    worker.llm = llm
    worker.tokenizer = object()

    with (
        patch("nemo_rl.models.generation.vllm.vllm_worker.gc.collect"),
        patch("nemo_rl.models.generation.vllm.vllm_worker.torch.cuda.empty_cache"),
        patch("nemo_rl.models.generation.vllm.vllm_worker.shutdown_telemetry"),
    ):
        assert worker.shutdown() is not close_fails

    assert llm.collective_rpc.call_args_list == [
        call("close_rollout_profiler", args=tuple()),
        call("cleanup", args=tuple()),
    ]
    assert (worker.llm, worker.tokenizer) == (None, None)
    assert ("close failed" in capsys.readouterr().out) is close_fails


@pytest.mark.asyncio
async def test_async_worker_shutdown_cleans_up_after_profiler_close_failure(capsys):
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._rollout_profiler_class = "profiler.Plugin"
    worker._sparse_refit_receiver = None
    llm = MagicMock()
    llm.collective_rpc = AsyncMock(side_effect=[RuntimeError("close failed"), None])
    worker.llm = llm
    worker.tokenizer = object()
    worker.server_thread = None

    with (
        patch("nemo_rl.models.generation.vllm.vllm_worker_async.gc.collect"),
        patch(
            "nemo_rl.models.generation.vllm.vllm_worker_async.torch.cuda.empty_cache"
        ),
        patch("nemo_rl.models.generation.vllm.vllm_worker_async.shutdown_telemetry"),
    ):
        assert await worker.shutdown() is False

    assert llm.collective_rpc.await_args_list == [
        call("close_rollout_profiler", args=tuple()),
        call("cleanup", args=tuple()),
    ]
    llm.shutdown.assert_called_once_with()
    assert (worker.llm, worker.tokenizer) == (None, None)
    assert "Error during vLLM shutdown: close failed" in capsys.readouterr().out
