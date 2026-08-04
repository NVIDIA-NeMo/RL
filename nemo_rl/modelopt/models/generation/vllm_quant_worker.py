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
from nemo_rl.modelopt.utils import (
    MODELOPT_KV_CACHE_QUANT_SKIP_FIRST_N,
    get_kv_cache_quant_skip_first_n,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
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
    "VLLM_MODELOPT_REAL_QUANT",
    MODELOPT_KV_CACHE_QUANT_SKIP_FIRST_N,
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
    first_n = get_kv_cache_quant_skip_first_n()
    if first_n:
        if cfg.get("real_quant"):
            raise ValueError(
                "K/V boundary skipping supports simulated quantization only."
            )
        if not cfg.get("quant_cfg"):
            raise ValueError("K/V boundary skipping requires generation.quant_cfg.")
        if llm_kwargs.get("kv_cache_dtype") != "auto":
            raise ValueError("K/V boundary skipping requires kv_cache_dtype='auto'.")
        if llm_kwargs.get("enable_prefix_caching") is not False:
            raise ValueError(
                "K/V boundary skipping requires prefix caching to be disabled."
            )
        if llm_kwargs.get("speculative_config") is not None:
            raise ValueError(
                "K/V boundary skipping does not support speculative decoding."
            )
        if llm_kwargs.get("decode_context_parallel_size", 1) != 1:
            raise ValueError("K/V boundary skipping does not support vLLM DCP.")
        if llm_kwargs.get("enable_dbo", False):
            raise ValueError("K/V boundary skipping does not support vLLM DBO.")
        attention_backend = llm_kwargs.get("attention_backend")
        if str(attention_backend).split(".")[-1] != "FLASH_ATTN":
            raise ValueError(
                "K/V boundary skipping requires attention_backend='FLASH_ATTN'."
            )
        if llm_kwargs.get("enforce_eager") is not True:
            raise ValueError("K/V boundary skipping requires enforce_eager=true.")

    extension_name = "VllmQuantInternalWorkerExtension"
    if checkpoint_engine_refit_config(cfg) is not None:
        extension_name += "WithCheckpointEngine"
    llm_kwargs["worker_extension_cls"] = (
        "nemo_rl.modelopt.models.generation.vllm_quant_backend." + extension_name
    )
    real_quant = bool(cfg.get("real_quant"))
    if real_quant:
        from nemo_rl.modelopt.models.generation.vllm_modelopt import (
            quantization_method_for_mode,
            register_nemo_modelopt_nvfp4,
        )
        from nemo_rl.modelopt.utils import (
            build_vllm_modelopt_nvfp4_config,
            resolve_nvfp4_real_quant_mode,
        )

        quant_cfg = cfg.get("quant_cfg")
        if not quant_cfg:
            raise ValueError("NVFP4 real quantization requires a non-empty quant_cfg.")
        mode = resolve_nvfp4_real_quant_mode(quant_cfg)
        register_nemo_modelopt_nvfp4()
        os.environ.pop("VLLM_QUANT_CFG", None)
        os.environ["VLLM_MODELOPT_REAL_QUANT"] = "1"

        hf_overrides = llm_kwargs.setdefault("hf_overrides", {})
        hf_overrides["quantization_config"] = build_vllm_modelopt_nvfp4_config(
            mode=mode,
            ignore=cfg.get("real_quant_ignore"),
        )
        llm_kwargs["quantization"] = quantization_method_for_mode(mode)
    else:
        llm_kwargs["worker_cls"] = (
            "nemo_rl.modelopt.models.generation.vllm_quant_patch.FakeQuantWorker"
        )
        # Expert fakequant needs a decomposed MoE path; explicit user config still wins.
        llm_kwargs.setdefault("moe_backend", "triton")
        os.environ.pop("VLLM_MODELOPT_REAL_QUANT", None)
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
