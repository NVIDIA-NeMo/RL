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

"""vLLM general plugin: deferred-capture nsight config for the v2 inner TP workers.

vLLM loads plugins registered in the ``vllm.general_plugins`` entry-point group
*early in every process it starts* -- including the spawned EngineCore subprocess
where the v2 ``RayExecutorV2`` builds the runtime_env for the inner ``RayWorkerProc``
workers. NeMo-RL cannot reach that subprocess with an in-process monkey-patch
(``spawn`` = fresh interpreter, pristine vLLM import), so without this the inner
workers launch with vLLM's stock always-on nsight config (``worker_process_%p``,
no capture-range) that never finalizes under the teardown SIGKILL.

This plugin runs inside that subprocess and installs a deferred-capture nsight
config (``build_inner_nsight_config``), so tracing is gated by the
``cudaProfilerStart/Stop`` NeMo-RL fires over ``NRL_NSYS_PROFILE_STEP_RANGE``, and
``capture-range-end=stop`` makes nsys write each inner worker's report while the
worker is still running rather than at teardown. It is a no-op unless
``NRL_NSYS_WORKER_PATTERNS`` is set. The v1 executor is patched in-process in
``vllm_worker.py``; both executors share ``build_executor_nsight_config``.
"""

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _engine_owns_its_gpus(world_size: int) -> bool:
    """Whether a vLLM engine has its GPUs to itself (not colocated with training).

    NeMo-RL requests ``fraction_of_gpus / world_size`` GPUs per inner worker
    (``VLLM_RAY_PER_WORKER_GPUS``, set in ``vllm_worker._load_model``), so the engine
    holds whole GPUs unless it shares them with colocated training workers.
    """
    per_worker_gpus = float(os.environ.get("VLLM_RAY_PER_WORKER_GPUS", "1"))
    return per_worker_gpus * world_size >= 1 - 1e-6


def _samples_worker_gpu_metrics(executor_cls: type, parallel_config: Any) -> bool:
    """Whether the inner workers of this engine sample their own GPU's metrics.

    True when ``{"gpu-metrics-devices": "worker"}`` is requested and the engine is not
    colocated with the training workers (they already sample the shared GPUs). vLLM
    gives all inner workers of an engine one runtime_env, so this also wraps the
    executor module's ``ray`` to resolve the ``"worker"`` marker per actor creation.
    """
    from nemo_rl.utils.nsys import worker_gpu_metrics_flag

    if worker_gpu_metrics_flag() is None or not _engine_owns_its_gpus(
        parallel_config.world_size
    ):
        return False
    module = sys.modules[executor_cls.__module__]
    if not isinstance(module.ray, _WorkerGpuMetricsRay):
        module.ray = _WorkerGpuMetricsRay(module.ray)
    return True


def build_executor_nsight_config(
    executor_cls: type,
    parallel_config: Any,
    capture_range_end: str,
    wait: str | None = None,
) -> dict[str, Any]:
    """Inner-worker nsight config for a vLLM Ray executor (v1 patch and v2 plugin).

    With ``{"gpu-metrics-devices": "worker"}`` in NRL_NSYS_EXTRA_OPTIONS, each inner
    worker samples the GPU it runs on (see ``_samples_worker_gpu_metrics``).
    """
    from nemo_rl.utils.nsys import build_inner_nsight_config

    return build_inner_nsight_config(
        capture_range_end=capture_range_end,
        wait=wait,
        gpu_metrics_per_worker=_samples_worker_gpu_metrics(
            executor_cls, parallel_config
        ),
    )


def _resolve_worker_gpu_metrics(options: dict[str, Any]) -> dict[str, Any]:
    """Return actor options whose ``"worker"`` GPU-metrics marker names the worker's GPU."""
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from nemo_rl.utils.nsys import (
        GPU_METRICS_WORKER,
        set_worker_gpu_metrics_device,
        worker_gpu_metrics_flag,
    )

    flag = worker_gpu_metrics_flag()
    nsight_config = (options.get("runtime_env") or {}).get("nsight")
    if (
        flag is None
        or not isinstance(nsight_config, dict)
        or nsight_config.get(flag) != GPU_METRICS_WORKER
    ):
        return options
    # Copy: the runtime_env is shared by all inner workers of the engine.
    nsight_config = dict(nsight_config)
    strategy = options.get("scheduling_strategy")
    if (
        isinstance(strategy, PlacementGroupSchedulingStrategy)
        and strategy.placement_group_bundle_index >= 0
    ):
        set_worker_gpu_metrics_device(
            nsight_config,
            strategy.placement_group,
            strategy.placement_group_bundle_index,
        )
    else:
        # Not pinned to a bundle, so its GPU is unknown: do not sample.
        del nsight_config[flag]
    return {
        **options,
        "runtime_env": {**options["runtime_env"], "nsight": nsight_config},
    }


class _WorkerGpuMetricsRay:
    """``ray`` as used by a vLLM Ray executor module, resolving per-worker GPU metrics.

    Every actor creation (v1: ``ray.remote(**options)(cls)``, v2:
    ``ray.remote(cls).options(**options)``) gets ``_resolve_worker_gpu_metrics``
    applied to its options. Everything else is passed to ``ray`` unchanged.
    """

    def __init__(self, ray_module: Any):
        self._ray = ray_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ray, name)

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            return self._ray.remote(*args, **_resolve_worker_gpu_metrics(kwargs))
        return _ActorClassWithWorkerGpuMetrics(self._ray.remote(*args))


class _ActorClassWithWorkerGpuMetrics:
    def __init__(self, actor_class: Any):
        self._actor_class = actor_class

    def __getattr__(self, name: str) -> Any:
        return getattr(self._actor_class, name)

    def options(self, **kwargs: Any) -> Any:
        return self._actor_class.options(**_resolve_worker_gpu_metrics(kwargs))


def register_nsight_inner_worker_profiling() -> None:
    """Entry point for vLLM's ``vllm.general_plugins`` group; run by ``load_general_plugins()``."""
    # Only act when NeMo-RL nsys profiling is requested (same gate as the workers).
    # This runs in every vLLM process, so check the raw env var instead of importing
    # nemo_rl.utils.nsys (torch, rich) when profiling is off.
    if not os.environ.get("NRL_NSYS_WORKER_PATTERNS"):
        return

    # No fallback: if a vLLM upgrade moves RayExecutorV2, this raises when profiling
    # is requested instead of leaving the inner workers silently unprofiled.
    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    if getattr(RayExecutorV2, "_nrl_nsight_patched", False):
        return

    _orig_build_runtime_env = RayExecutorV2._build_runtime_env

    def _patched_build_runtime_env(self):
        runtime_env = _orig_build_runtime_env(self)
        if self.parallel_config.ray_workers_use_nsight:
            # Built here so a bad NRL_NSYS_EXTRA_OPTIONS raises at executor startup.
            # capture-range-end=stop ends collection at the end of the profile range
            # and nsys writes the report while the worker keeps running.
            # wait=primary: nsys waits only for the profiled worker process, not for
            # the child processes it spawns; without it, some inner reports were
            # never written.
            runtime_env["nsight"] = build_executor_nsight_config(
                RayExecutorV2,
                self.parallel_config,
                capture_range_end="stop",
                wait="primary",
            )
        elif isinstance(runtime_env.get("_nsight"), dict) and (
            _samples_worker_gpu_metrics(RayExecutorV2, self.parallel_config)
        ):
            # Only the outer async vLLM worker is profiled (pattern matching
            # vllm_async_generation_worker alone): the inner workers inherit its nsight
            # config through vLLM's ray_runtime_env. Give them their own GPU's metrics.
            from nemo_rl.utils.nsys import GPU_METRICS_WORKER, worker_gpu_metrics_flag

            runtime_env["nsight"] = {
                **runtime_env.pop("_nsight"),
                str(worker_gpu_metrics_flag()): GPU_METRICS_WORKER,
            }
        return runtime_env

    RayExecutorV2._build_runtime_env = _patched_build_runtime_env
    RayExecutorV2._nrl_nsight_patched = True
    logger.info("Installed NeMo-RL deferred-capture nsight config on RayExecutorV2")
