# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import ray

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.vllm.vllm_worker import (
    VllmGenerationWorkerImpl,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)
class VllmQuantGenerationWorker(VllmGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = ["VLLM_QUANT_CFG"]
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        llm_kwargs["worker_cls"] = (
            "nemo_rl.models.generation.vllm.quantization.vllm_quant_patch.FakeQuantWorker"
        )
        llm_kwargs["worker_extension_cls"] = (
            "nemo_rl.models.generation.vllm.quantization.vllm_quant_backend.VllmQuantInternalWorkerExtension"
        )
        if self.cfg["quant_cfg"]:
            print("setting VLLM_QUANT_CFG to: ", self.cfg["quant_cfg"])
            os.environ["VLLM_QUANT_CFG"] = self.cfg["quant_cfg"]

        super()._create_engine(llm_kwargs)

    def export_amax(self) -> dict[str, Any]:
        """Export amax buffers for testing/debugging."""
        if not hasattr(self, "llm"):
            return {}
        try:
            results = self.llm.collective_rpc("export_amax", args=tuple())
            return results[0] if results else {}
        except Exception:
            return {}


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)
class VllmQuantAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = ["VLLM_QUANT_CFG"]
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        llm_kwargs["worker_cls"] = (
            "nemo_rl.models.generation.vllm.quantization.vllm_quant_patch.FakeQuantWorker"
        )
        llm_kwargs["worker_extension_cls"] = (
            "nemo_rl.models.generation.vllm.quantization.vllm_quant_backend.VllmQuantInternalWorkerExtension"
        )
        if self.cfg["quant_cfg"]:
            os.environ["VLLM_QUANT_CFG"] = self.cfg["quant_cfg"]

        super()._create_engine(llm_kwargs)

    def export_amax(self) -> dict[str, Any]:
        """Export amax buffers for testing/debugging."""
        if not hasattr(self, "llm"):
            return {}
        try:
            results = self.llm.collective_rpc("export_amax", args=tuple())
            return results[0] if results else {}
        except Exception:
            return {}
