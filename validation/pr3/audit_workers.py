# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


"""Allocation-only probes for synchronized training and generation GPU execution."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_TRACES: dict[str, Any] = {}
_EXPORTED: set[str] = set()


def start_trace(event: str) -> None:
    if (
        os.environ.get("PR3_PROFILE_KERNELS") != "1"
        or event in _EXPORTED
        or event in _TRACES
    ):
        return
    # CUDA ownership must be established before loading Torch in worker hooks.
    import torch

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    )
    profiler.start()
    _TRACES[event] = profiler


def stop_trace(event: str) -> None:
    profiler = _TRACES.pop(event, None)
    if profiler is None:
        return
    profiler.stop()
    profiler.export_chrome_trace(
        str(Path(os.environ["PR3_AUDIT_DIR"]) / f"{event}-{os.getpid()}.trace.json")
    )
    _EXPORTED.add(event)


def emit(event: str, **values: Any) -> None:
    directory = Path(os.environ["PR3_AUDIT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"worker-{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid(), **values}) + "\n")


def timed_gpu_call(event: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    """Import CUDA dependencies only after the actor has assigned GPU visibility."""
    import torch

    if event == "training_gpu":
        start_trace(event)
    torch.cuda.synchronize()
    start_ns = time.monotonic_ns()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    with torch.profiler.record_function(f"pr3_{event}"):
        result = function(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
    emit(
        event,
        start_ns=start_ns,
        end_ns=time.monotonic_ns(),
        cuda_ms=begin.elapsed_time(end),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        logical_device=torch.cuda.current_device(),
    )
    if event == "training_gpu":
        stop_trace(event)
    return result


def setup() -> None:
    from nemo_rl.distributed.ray_worker_setup import ray_worker_process_setup

    ray_worker_process_setup()
    if "vllm_worker" not in sys.prefix:
        return
    from nemo_rl.models.generation.vllm import vllm_worker

    validate = vllm_worker._validate_worker_extension_cls

    def extension(kwargs: dict[str, Any], required: str) -> None:
        validate(kwargs, required)
        kwargs["worker_extension_cls"] = (
            "validation.pr3.audit_vllm.StreamingAuditExtension"
        )

    vllm_worker._validate_worker_extension_cls = extension
