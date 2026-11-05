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


"""Audit-only vLLM execution timing, preserving the normal refit implementation."""

import os
import time
from typing import Any

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension
from validation.pr3.audit_workers import start_trace, stop_trace, timed_gpu_call


class StreamingAuditExtension(VllmInternalWorkerExtension):
    def _install_probe(self) -> None:
        if getattr(self, "_streaming_probe_installed", False):
            return
        self._streaming_probe_installed = True
        original = self.model_runner.execute_model

        def execute(*args: Any, **kwargs: Any) -> Any:
            # A private test gate stretches the tail so actual GPU overlap can
            # be observed even when this tiny model finishes rollout quickly.
            delay = float(os.environ.get("PR3_GENERATION_TICK_DELAY", "0"))
            if delay:
                time.sleep(delay)
            return timed_gpu_call("generation_gpu", original, *args, **kwargs)

        self.model_runner.execute_model = execute

    def update_weights_from_collective(self, *args: Any, **kwargs: Any) -> Any:
        stop_trace("generation_gpu")
        result = VllmInternalWorkerExtension.update_weights_from_collective(
            self, *args, **kwargs
        )
        self._install_probe()
        start_trace("generation_gpu")
        return result

    def update_weights_via_ipc_zmq(self, *args: Any, **kwargs: Any) -> Any:
        stop_trace("generation_gpu")
        result = VllmInternalWorkerExtension.update_weights_via_ipc_zmq(
            self, *args, **kwargs
        )
        self._install_probe()
        start_trace("generation_gpu")
        return result
