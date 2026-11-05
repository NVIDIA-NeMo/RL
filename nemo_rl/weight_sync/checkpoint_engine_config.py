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

from typing import Any, cast

from nemo_rl.models.generation.interfaces import (
    CheckpointEngineConfig,
    GenerationConfig,
)
from nemo_rl.models.generation.vllm.config import (
    VLLM_SPARSE_REFIT_TRANSPORTS,
    VllmCheckpointEnginePluginConfig,
    VllmConfig,
    normalize_vllm_refit_config,
)


def checkpoint_engine_refit_config(
    generation_config: GenerationConfig,
) -> CheckpointEngineConfig | None:
    """Translate a checkpoint-engine refit scope into the internal schema."""
    config = cast(VllmConfig, generation_config)
    transport = config.get("refit_transport")
    refit_config = normalize_vllm_refit_config(config)
    if (
        refit_config is None
        or transport is None
        or transport in VLLM_SPARSE_REFIT_TRANSPORTS
    ):
        return None

    if transport == "nixl":
        scoped_config = refit_config.nixl
        backend = "nixl"
    else:
        plugin_config = cast(dict[str, Any], refit_config.model_extra or {}).get(
            transport
        )
        scoped_config = VllmCheckpointEnginePluginConfig.model_validate(plugin_config)
        backend = transport

    engine_kwargs: dict[str, dict[str, Any]] = {
        backend: scoped_config.model_dump(
            exclude={"update_weights_bucket_memory_ratio"}
        )
    }
    checkpoint_engine_config: CheckpointEngineConfig = {
        "backend": backend,
        "update_weights_bucket_memory_ratio": (
            scoped_config.update_weights_bucket_memory_ratio
        ),
        "engine_kwargs": engine_kwargs,
    }
    return checkpoint_engine_config
