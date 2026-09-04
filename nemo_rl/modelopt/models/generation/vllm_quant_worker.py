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
from nemo_rl.models.generation.vllm.config import (
    VLLM_SPARSE_REFIT_TRANSPORTS,
    VllmConfig,
)
from nemo_rl.models.generation.vllm.vllm_worker import (
    VllmGenerationWorkerImpl,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.weight_sync.checkpoint_engine_config import (
    checkpoint_engine_refit_config,
)

_EXTRA_ENV_VARS = (
    "VLLM_QUANT_CFG",
    "PYTHONPATH",
)


def _quant_cfg_for_worker_env(quant_cfg: str) -> str:
    expanded = os.path.expanduser(quant_cfg)
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    return quant_cfg


def _configure_quant_engine_kwargs(
    cfg: VllmConfig,
    llm_kwargs: dict[str, Any],
) -> None:
    real_quant = bool(cfg.get("real_quant"))
    checkpoint_engine_config = checkpoint_engine_refit_config(cfg)
    refit_transport = cfg.get("refit_transport")
    if real_quant and (
        checkpoint_engine_config is not None
        or refit_transport in VLLM_SPARSE_REFIT_TRANSPORTS
    ):
        raise ValueError(
            f"ModelOpt real quantization does not support refit_transport="
            f"{refit_transport!r} because it bypasses vLLM's native layerwise "
            "reload lifecycle"
        )

    extension_name = "VllmQuantInternalWorkerExtension"
    if checkpoint_engine_config is not None:
        extension_name += "WithCheckpointEngine"
    llm_kwargs["worker_extension_cls"] = (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend." + extension_name
    )
    if real_quant:
        os.environ.pop("VLLM_QUANT_CFG", None)
        quantization_config = llm_kwargs.get("hf_overrides", {}).get(
            "quantization_config"
        )
        if not quantization_config:
            raise ValueError(
                "Real quantization requires a policy-produced "
                "hf_overrides.quantization_config before vLLM initialization"
            )
    else:
        llm_kwargs["worker_cls"] = (
            "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
        )
        # Expert fakequant needs a decomposed MoE path; explicit user config still wins.
        llm_kwargs.setdefault("moe_backend", "triton")
        os.environ.pop("VLLM_QUANT_CFG", None)
        if cfg["quant_cfg"]:
            os.environ["VLLM_QUANT_CFG"] = _quant_cfg_for_worker_env(cfg["quant_cfg"])


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)  # pragma: no cover
class VllmQuantGenerationWorker(VllmGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = _EXTRA_ENV_VARS
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        _configure_quant_engine_kwargs(self.cfg, llm_kwargs)
        super()._create_engine(llm_kwargs)

    def get_quantizer_stats(self) -> dict[str, Any]:
        """Return quantizer statistics. Mirrors MegatronQuantPolicyWorker.get_quantizer_stats()."""
        results = self.llm.collective_rpc("get_quantizer_stats", args=tuple())
        return results[0]

    def get_weight_snapshot(self, name: str) -> Any:
        """Return a CPU copy of a named parameter for before/after comparison."""
        results = self.llm.collective_rpc("get_weight_snapshot", args=(name,))
        return results[0]


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmQuantAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    def __init__(self, *args, **kwargs):
        kwargs["extra_env_vars"] = _EXTRA_ENV_VARS
        super().__init__(*args, **kwargs)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        _configure_quant_engine_kwargs(self.cfg, llm_kwargs)
        super()._create_engine(llm_kwargs)

    async def get_quantizer_stats(self) -> dict[str, Any]:
        """Return quantizer statistics. Mirrors MegatronQuantPolicyWorker.get_quantizer_stats()."""
        results = await self.llm.collective_rpc("get_quantizer_stats", args=tuple())
        return results[0]

    async def get_weight_snapshot(self, name: str) -> Any:
        """Return a CPU copy of a named parameter for before/after comparison."""
        results = await self.llm.collective_rpc("get_weight_snapshot", args=(name,))
        return results[0]
