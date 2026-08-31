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

from __future__ import annotations

import math
from typing import Any, cast

from nemo_rl.models.policy import PolicyConfig


def sampling_mask_replay_enabled(config: PolicyConfig) -> bool:
    return bool((config.get("sampling_mask_replay") or {}).get("enabled", False))


def configure_vllm_for_sampling_mask_replay(config: PolicyConfig) -> None:
    """Validate and apply the vLLM settings required for mask replay."""
    if not sampling_mask_replay_enabled(config):
        return

    generation = cast(dict[str, Any], config.get("generation") or {})
    megatron_cfg = cast(dict[str, Any], config.get("megatron_cfg") or {})

    if generation.get("backend") != "vllm":
        raise ValueError("sampling_mask_replay.enabled requires vLLM generation.")
    if not megatron_cfg.get("enabled", False):
        raise ValueError(
            "sampling_mask_replay.enabled requires the Megatron policy backend."
        )

    temperature = generation.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            "sampling_mask_replay.enabled requires generation.temperature > 0."
        )

    top_k = generation.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(
            "sampling_mask_replay.enabled requires generation.top_k to be a "
            "positive integer."
        )

    if (config.get("sequence_packing") or {}).get("enabled", False):
        raise ValueError(
            "sampling_mask_replay.enabled does not support sequence packing."
        )
    if megatron_cfg.get("use_fused_linear_logprobs", False):
        raise ValueError(
            "sampling_mask_replay.enabled does not support "
            "megatron_cfg.use_fused_linear_logprobs."
        )

    vllm_kwargs = generation.setdefault("vllm_kwargs", {})
    vllm_kwargs["return_sampling_mask"] = True
    vllm_cfg = generation.setdefault("vllm_cfg", {})
    vllm_cfg["logprobs_mode"] = "processed_logprobs"
    env_vars = vllm_cfg.get("env_vars") or {}
    env_vars["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    vllm_cfg["env_vars"] = env_vars
