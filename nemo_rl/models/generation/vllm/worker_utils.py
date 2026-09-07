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

from collections.abc import Mapping
from typing import Any

_REFIT_CACHE_LOADER_ROUTES_KEY = "nemo_rl_refit_cache_loader_routes"


def configure_refit_runtime(
    vllm_cfg: Mapping[str, Any], vllm_kwargs: dict[str, Any]
) -> None:
    """Forward NeMo-RL refit options through vLLM's worker config."""
    additional_config = dict(vllm_kwargs.get("additional_config") or {})
    additional_config[_REFIT_CACHE_LOADER_ROUTES_KEY] = vllm_cfg.get(
        "refit_cache_loader_routes", False
    )
    vllm_kwargs["additional_config"] = additional_config


def refit_cache_loader_routes_enabled(vllm_config: Any) -> bool:
    """Return the configured loader-route cache setting in a vLLM worker."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if _REFIT_CACHE_LOADER_ROUTES_KEY not in additional_config:
        return False
    value = additional_config[_REFIT_CACHE_LOADER_ROUTES_KEY]
    if not isinstance(value, bool):
        raise TypeError(f"{_REFIT_CACHE_LOADER_ROUTES_KEY} must be a boolean")
    return value


def resolve_distributed_executor_backend(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> str | None:
    if tensor_parallel_size * pipeline_parallel_size > 1:
        return "ray"
    if expert_parallel_size > tensor_parallel_size:
        # External DP actors already own one GPU each.
        return "uni"
    return None


def resolve_data_parallel_local_rank(
    rank: int, model_parallel_size: int, executor_backend: str | None
) -> int:
    # Ray remaps one GPU into each external-DP actor.
    if executor_backend == "uni":
        return 0
    return (rank % 8) // model_parallel_size
