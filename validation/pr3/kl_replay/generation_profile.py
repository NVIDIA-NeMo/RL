# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture actual decode kernels on TP rank0 between two ordinary refits."""

import os
from pathlib import Path
from typing import Any

import torch

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension


class TraceExtension(VllmInternalWorkerExtension):
    _pr3_kl_profiler = None
    _pr3_kl_trace_started = False
    _pr3_kl_refits = 0

    def _pr3_stop_trace(self) -> None:
        if self._pr3_kl_profiler is not None:
            self._pr3_kl_profiler.stop()
            directory = Path(os.environ["PR3_AUDIT_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            self._pr3_kl_profiler.export_chrome_trace(
                str(directory / f"generation_gpu-{os.getpid()}.trace.json")
            )
            self._pr3_kl_profiler = None

    def _pr3_start_trace(self) -> None:
        if self._pr3_kl_trace_started or torch.distributed.get_rank() != 0:
            return
        self._pr3_kl_refits += 1
        if self._pr3_kl_refits != int(os.environ.get("PR3_KL_PROFILE_ITERATION", "1")):
            return
        self._pr3_kl_trace_started = True
        execute = self.model_runner.execute_model

        def traced(*args: Any, **kwargs: Any) -> Any:
            # Synchronization keeps the kernel timestamps inside the labelled
            # execution window. This profile is evidence of overlap only.
            torch.cuda.synchronize()
            with torch.profiler.record_function("pr3_generation_gpu"):
                result = execute(*args, **kwargs)
                torch.cuda.synchronize()
            return result

        self.model_runner.execute_model = traced
        self._pr3_kl_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        self._pr3_kl_profiler.start()

    def update_weights_from_collective(self, *args: Any, **kwargs: Any) -> Any:
        self._pr3_stop_trace()
        result = super().update_weights_from_collective(*args, **kwargs)
        self._pr3_start_trace()
        return result

    def update_weights_via_ipc_zmq(self, *args: Any, **kwargs: Any) -> Any:
        self._pr3_stop_trace()
        result = super().update_weights_via_ipc_zmq(*args, **kwargs)
        self._pr3_start_trace()
        return result
