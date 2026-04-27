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
)  # pragma: no cover
class VllmQuantGenerationWorker(VllmGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = ["VLLM_QUANT_CFG"]
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        llm_kwargs["worker_cls"] = (
            "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
        )
        llm_kwargs["worker_extension_cls"] = (
            "nemo_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
        )
        if self.cfg["quant_cfg"]:
            print("setting VLLM_QUANT_CFG to: ", self.cfg["quant_cfg"])
            os.environ["VLLM_QUANT_CFG"] = self.cfg["quant_cfg"]

        super()._create_engine(llm_kwargs)

    def _collective_rpc_or_empty(self, method: str) -> dict[str, Any]:
        """Best-effort RPC call; returns {} on any failure.

        collective_rpc can propagate arbitrary exceptions from the internal
        worker (RuntimeError, AttributeError, etc.), so broad except is
        intentional here -- consistent with the base class pattern.
        """
        if not hasattr(self, "llm"):
            return {}
        try:
            results = self.llm.collective_rpc(method, args=tuple())
            return results[0] if results else {}
        except Exception:
            return {}

    def export_amax(self) -> dict[str, Any]:
        """Export amax buffers for testing/debugging."""
        return self._collective_rpc_or_empty("export_amax")

    def get_quantizer_stats(self) -> dict[str, Any]:
        """Return quantizer statistics. Mirrors MegatronQuantPolicyWorker.get_quantizer_stats()."""
        return self._collective_rpc_or_empty("get_quantizer_stats")

    def get_weight_snapshot(self, name: str) -> Any:
        """Return a CPU copy of a named parameter for before/after comparison."""
        if not hasattr(self, "llm"):
            return None
        results = self.llm.collective_rpc("get_weight_snapshot", args=(name,))
        return results[0] if results else None


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmQuantAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = ["VLLM_QUANT_CFG"]
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        llm_kwargs["worker_cls"] = (
            "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
        )
        llm_kwargs["worker_extension_cls"] = (
            "nemo_rl.modelopt.models.generation.vllm_quant_backend.VllmQuantInternalWorkerExtension"
        )
        if self.cfg["quant_cfg"]:
            os.environ["VLLM_QUANT_CFG"] = self.cfg["quant_cfg"]

        super()._create_engine(llm_kwargs)

    async def _collective_rpc_or_empty(self, method: str) -> dict[str, Any]:
        """Best-effort async RPC call; returns {} on any failure.

        See sync counterpart for rationale on broad except.
        """
        if not hasattr(self, "llm"):
            return {}
        try:
            results = await self.llm.collective_rpc(method, args=tuple())
            return results[0] if results else {}
        except Exception:
            return {}

    async def export_amax(self) -> dict[str, Any]:
        """Export amax buffers for testing/debugging."""
        return await self._collective_rpc_or_empty("export_amax")

    async def get_quantizer_stats(self) -> dict[str, Any]:
        """Return quantizer statistics. Mirrors MegatronQuantPolicyWorker.get_quantizer_stats()."""
        return await self._collective_rpc_or_empty("get_quantizer_stats")

    async def get_weight_snapshot(self, name: str) -> Any:
        """Return a CPU copy of a named parameter for before/after comparison."""
        if not hasattr(self, "llm"):
            return None
        results = await self.llm.collective_rpc("get_weight_snapshot", args=(name,))
        return results[0] if results else None
