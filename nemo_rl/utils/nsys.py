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
import atexit
import json
import os
from typing import Any, Protocol

import rich
import torch

NRL_NSYS_WORKER_PATTERNS = os.environ.get("NRL_NSYS_WORKER_PATTERNS", "")
NRL_NSYS_PROFILE_STEP_RANGE = os.environ.get("NRL_NSYS_PROFILE_STEP_RANGE", "")


def _parse_extra_options(raw: str) -> dict[str, Any]:
    """Parse NRL_NSYS_EXTRA_OPTIONS as a JSON object of nsys flag -> value.

    Empty string parses to an empty dict. Invalid JSON or non-object payloads raise
    so misconfiguration surfaces immediately rather than silently being ignored.
    A ``null`` value removes that flag from the built-in defaults
    (see ``apply_nsys_extra_options``).
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse NRL_NSYS_EXTRA_OPTIONS as JSON: {e}. "
            "Expected a JSON object mapping nsys flag names (without leading --) to "
            'values, e.g. \'{"gpu-metrics-devices": "worker", "cuda-memory-usage": "true"}\'. '
            "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"NRL_NSYS_EXTRA_OPTIONS must be a JSON object, got {type(parsed).__name__}. "
            "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
        )
    return parsed


NRL_NSYS_EXTRA_OPTIONS: dict[str, Any] = _parse_extra_options(
    os.environ.get("NRL_NSYS_EXTRA_OPTIONS", "")
)


# NRL_NSYS_EXTRA_OPTIONS value for the nsys GPU-metrics flag meaning "the GPU this
# worker runs on", e.g. {"gpu-metrics-devices": "worker"}. Only one nsys process can
# sample a GPU's metrics, and profiled workers see every GPU of their node, so one
# value for all workers collides. NeMo-RL resolves "worker" per profiled process.
GPU_METRICS_WORKER = "worker"
# nsys spells the flag --gpu-metrics-devices (--gpu-metrics-device in older releases).
_GPU_METRICS_FLAGS = ("gpu-metrics-devices", "gpu-metrics-device")


def worker_gpu_metrics_flag() -> str | None:
    """Return the GPU-metrics flag set to ``"worker"`` in NRL_NSYS_EXTRA_OPTIONS, if any."""
    for flag in _GPU_METRICS_FLAGS:
        if NRL_NSYS_EXTRA_OPTIONS.get(flag) == GPU_METRICS_WORKER:
            return flag
    return None


def set_worker_gpu_metrics_device(
    nsight_config: dict[str, Any], placement_group: Any, bundle_index: int
) -> None:
    """Point the ``"worker"`` GPU-metrics flag at the GPU of a placement-group bundle.

    That is the GPU Ray assigned to the bundle the worker is scheduled into, i.e. the
    GPU the worker runs on. No-op unless ``"worker"`` was requested.
    """
    flag = worker_gpu_metrics_flag()
    if flag is None:
        return
    from nemo_rl.distributed.virtual_cluster import get_bundle_gpu_id

    nsight_config[flag] = str(get_bundle_gpu_id(placement_group, bundle_index))


def apply_nsys_extra_options(
    nsight_config: dict[str, Any], extra_options: dict[str, Any]
) -> dict[str, Any]:
    """Merge user nsys flags into ``nsight_config`` in place and return it.

    User keys override the built-in defaults; a ``None`` value (JSON ``null``) removes
    the key instead, e.g. ``{"cuda-graph-trace": null}`` drops that default flag.
    A ``"worker"`` GPU-metrics value is not copied; it is set per worker when the
    worker is created (``set_worker_gpu_metrics_device``).
    """
    for key, value in extra_options.items():
        if value is None or (key in _GPU_METRICS_FLAGS and value == GPU_METRICS_WORKER):
            nsight_config.pop(key, None)
        else:
            nsight_config[key] = value
    return nsight_config


def build_inner_nsight_config(
    capture_range_end: str,
    wait: str | None = None,
    gpu_metrics_per_worker: bool = False,
) -> dict[str, Any]:
    """Deferred-capture nsight config for vLLM's internal TP workers.

    Shared by the v1 executor patch (``vllm_worker.py``) and the v2 executor plugin
    (``vllm_nsight_plugin.py``); each passes its executor-specific
    ``capture_range_end`` and ``wait``. Tracing starts/stops on the
    cudaProfilerStart/Stop that ``start/stop_gpu_profiling`` send to every inner worker.
    With ``gpu_metrics_per_worker``, a ``"worker"`` GPU-metrics request stays in the
    config as a marker that ``vllm_nsight_plugin`` resolves for each inner worker.
    """
    nsight_config: dict[str, Any] = {
        "t": "cuda,cudnn,cublas,nvtx",
        "o": f"'vllm_tp_worker_{NRL_NSYS_PROFILE_STEP_RANGE}_%p'",
        "stop-on-exit": "true",
        "s": "none",
        "capture-range": "cudaProfilerApi",
        "capture-range-end": capture_range_end,
        "cuda-graph-trace": "node",
    }
    if wait is not None:
        nsight_config["wait"] = wait
    apply_nsys_extra_options(nsight_config, NRL_NSYS_EXTRA_OPTIONS)
    flag = worker_gpu_metrics_flag()
    if gpu_metrics_per_worker and flag is not None:
        nsight_config[flag] = GPU_METRICS_WORKER
    return nsight_config


class ProfilablePolicy(Protocol):
    def start_gpu_profiling(self) -> None: ...

    def stop_gpu_profiling(self) -> None: ...


def maybe_gpu_profile_step(policy: ProfilablePolicy, step: int):
    assert not (bool(NRL_NSYS_WORKER_PATTERNS) ^ bool(NRL_NSYS_PROFILE_STEP_RANGE)), (
        "Either both NRL_NSYS_WORKER_PATTERNS and NRL_NSYS_PROFILE_STEP_RANGE must be set, or neither. See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
    )

    if not NRL_NSYS_WORKER_PATTERNS:
        return

    NSYS_START_STEP, NSYS_STOP_STEP = NRL_NSYS_PROFILE_STEP_RANGE.split(":")
    try:
        NSYS_START_STEP = int(NSYS_START_STEP)
        NSYS_STOP_STEP = int(NSYS_STOP_STEP)
    except ValueError as e:
        raise ValueError(
            f"Error parsing NRL_NSYS_PROFILE_STEP_RANGE: {str(e)}. "
            "Please ensure the format is 'start:stop' where both values are integers. "
            "See https://github.com/NVIDIA/NeMo-RL/tree/main/docs/nsys-profiling.md for more details."
        ) from e

    assert NSYS_START_STEP < NSYS_STOP_STEP, (
        f"{NRL_NSYS_PROFILE_STEP_RANGE=} must be a non-empty range"
    )
    assert NSYS_START_STEP >= 1, (
        f"The start step in {NRL_NSYS_PROFILE_STEP_RANGE=} must be >= 1"
    )

    # Use slice syntax of left inclusive and right exclusive
    if NSYS_START_STEP <= step < NSYS_STOP_STEP:
        if not getattr(policy, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Starting GPU profiling for {policy} for step {step}[/bold red]"
            )
            policy.start_gpu_profiling()
            policy.__NRL_PROFILE_STARTED = True

            def stop_profiler_on_exit():
                rich.print(
                    f"[bold red]Stopping GPU profiling on exit for {policy} for step {step}[/bold red]"
                )
                policy.stop_gpu_profiling()

            atexit.register(stop_profiler_on_exit)
    else:
        if getattr(policy, "__NRL_PROFILE_STARTED", False):
            rich.print(
                f"[bold red]Stopping GPU profiling for {policy} for step {step}[/bold red]"
            )
            policy.stop_gpu_profiling()
            policy.__NRL_PROFILE_STARTED = False


def wrap_with_nvtx_name(name: str):
    """A decorator to wrap a function with an NVTX range with the given name."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            ret = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return ret

        return wrapper

    return decorator
