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

"""FP8 engine-argument helpers with no vLLM import, so they run anywhere."""

from typing import Any

# Carries an ``FP8Config`` (as JSON of its fields) to engine worker processes that
# NeMo RL does not create itself, such as managed Dynamo's. NeMo RL's worker
# extension applies the FP8 patches from it when vLLM loads the extension, which
# happens before the model is built.
FP8_CONFIG_ENV_VAR = "NRL_VLLM_FP8_CONFIG"


def merge_fp8_kwargs(vllm_kwargs: dict[str, Any], fp8_kwargs: dict[str, Any]) -> None:
    """Merge fp8 init kwargs into ``vllm_kwargs`` in place, preserving user overrides.

    ``init_fp8`` returns a nested ``hf_overrides`` (holding ``quantization_config``),
    so a blanket ``vllm_kwargs.update(fp8_kwargs)`` would wholesale-replace any
    user-supplied ``hf_overrides``. We pop ``hf_overrides`` before the shallow
    update and merge it separately so that fp8's ``quantization_config`` is the
    base while user overrides (e.g. ``max_position_embeddings``) survive and take
    precedence. Required FP8 ignore entries are combined with user-provided ignore
    entries so generated safety exclusions cannot be removed. This regression was
    reintroduced once already; see #1413/#2904.
    """
    fp8_kwargs = dict(fp8_kwargs)
    fp8_hf_overrides = fp8_kwargs.pop("hf_overrides", {})
    vllm_kwargs.update(fp8_kwargs)
    existing_hf_overrides = vllm_kwargs.get("hf_overrides") or {}
    merged_hf_overrides = {**fp8_hf_overrides, **existing_hf_overrides}

    fp8_quantization_config = fp8_hf_overrides.get("quantization_config")
    existing_quantization_config = existing_hf_overrides.get("quantization_config")
    if existing_quantization_config is None:
        if fp8_quantization_config is not None:
            merged_hf_overrides["quantization_config"] = fp8_quantization_config
    elif not isinstance(existing_quantization_config, dict):
        raise ValueError("hf_overrides.quantization_config must be a mapping")
    elif isinstance(fp8_quantization_config, dict):
        merged_quantization_config = {
            **fp8_quantization_config,
            **existing_quantization_config,
        }
        for key in ("ignore", "ignored_layers"):
            generated_values = fp8_quantization_config.get(key, [])
            user_values = existing_quantization_config.get(key, [])
            if not isinstance(generated_values, list) or not isinstance(
                user_values, list
            ):
                raise ValueError(f"quantization_config.{key} must be a list")
            if generated_values or user_values:
                merged_quantization_config[key] = list(
                    dict.fromkeys([*generated_values, *user_values])
                )
        merged_hf_overrides["quantization_config"] = merged_quantization_config

    vllm_kwargs["hf_overrides"] = merged_hf_overrides
