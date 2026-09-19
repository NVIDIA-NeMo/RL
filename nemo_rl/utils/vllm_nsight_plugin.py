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

This plugin runs inside that subprocess and installs the same deferred-capture
nsight config NeMo-RL uses for the outer/policy workers, so tracing is gated by the
``cudaProfilerStart/Stop`` NeMo-RL fires over ``NRL_NSYS_PROFILE_STEP_RANGE`` and each
inner worker's trace is finalized non-destructively (see
``VllmInternalWorkerExtension.stop_gpu_profiling``). It is a no-op unless
``NRL_NSYS_WORKER_PATTERNS`` is set, and no-ops on vLLM builds without the v2 executor
(the v1 executor is patched in-process in ``vllm_worker.py``).
"""

import os


def register_nsight_inner_worker_profiling() -> None:
    """vLLM general-plugin entry point; called by ``load_general_plugins()``."""
    # Only act when NeMo-RL nsys profiling is requested (same gate as the workers).
    if not os.environ.get("NRL_NSYS_WORKER_PATTERNS"):
        return

    try:
        from vllm.v1.executor.ray_executor_v2 import RayExecutorV2
    except ImportError:
        # Older vLLM without the v2 executor -- nothing to patch here; the v1 path
        # is handled in-process by NeMo-RL's own monkey-patch.
        return

    if getattr(RayExecutorV2, "_nrl_nsight_patched", False):
        return

    _orig_build_runtime_env = RayExecutorV2._build_runtime_env

    def _patched_build_runtime_env(self):
        runtime_env = _orig_build_runtime_env(self)
        if self.parallel_config.ray_workers_use_nsight:
            import json

            rng = os.environ.get("NRL_NSYS_PROFILE_STEP_RANGE", "all")
            nsight_config = {
                "t": "cuda,cudnn,cublas,nvtx",
                "o": f"'vllm_tp_worker_{rng}_%p'",
                "stop-on-exit": "true",
                "s": "none",
                "capture-range": "cudaProfilerApi",
                "wait": "primary",
                "capture-range-end": "stop",
            }
            # NRL_NSYS_EXTRA_OPTIONS lets the user override the trace target, e.g.
            # t=cuda-sw,nvtx on platforms where the HW cuda target crashes at
            # cudaProfilerStart.
            extra = os.environ.get("NRL_NSYS_EXTRA_OPTIONS")
            if extra:
                nsight_config.update(json.loads(extra))
            runtime_env["nsight"] = nsight_config
        return runtime_env

    RayExecutorV2._build_runtime_env = _patched_build_runtime_env
    RayExecutorV2._nrl_nsight_patched = True
