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
``vllm_worker.py``.
"""

import logging
import os

logger = logging.getLogger(__name__)


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
            # Imported here so a bad NRL_NSYS_EXTRA_OPTIONS raises at executor startup.
            from nemo_rl.utils.nsys import build_inner_nsight_config

            # capture-range-end=stop ends collection at the end of the profile range
            # and nsys writes the report while the worker keeps running.
            # wait=primary: nsys waits only for the profiled worker process, not for
            # the child processes it spawns; without it, some inner reports were
            # never written.
            runtime_env["nsight"] = build_inner_nsight_config(
                capture_range_end="stop", wait="primary"
            )
        return runtime_env

    RayExecutorV2._build_runtime_env = _patched_build_runtime_env
    RayExecutorV2._nrl_nsight_patched = True
    logger.info("Installed NeMo-RL deferred-capture nsight config on RayExecutorV2")
