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

"""Pure configuration validation for the supported research execution path."""

from typing import TYPE_CHECKING

from turn_level_credit.config import TurnCreditConfig

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig


def _uses_async_rollouts(master_config: "MasterConfig") -> bool:
    """Mirror core rollout dispatch without importing the trainer module."""
    generation_config = master_config.policy["generation"]
    if generation_config is None:
        return False
    backend = generation_config["backend"]
    if backend == "sglang":
        return bool(generation_config.get("use_async_rollouts"))
    if backend == "vllm":
        return bool(generation_config["vllm_cfg"]["async_engine"])
    if backend == "trtllm":
        return True
    if backend == "megatron":
        return bool(generation_config["mcore_generation_config"]["async_engine"])
    return False


def validate_supported_path(
    master_config: "MasterConfig",
    turn_credit_config: TurnCreditConfig,
) -> None:
    """Fail at startup for execution paths not validated by this project."""
    estimator_name = master_config.grpo.adv_estimator.name
    if estimator_name != "grpo":
        raise ValueError(
            "Turn-level credit currently supports only grpo.adv_estimator.name=grpo"
        )
    if master_config.grpo.async_grpo.enabled:
        raise ValueError("Turn-level credit research does not support async GRPO")
    if master_config.data_plane and master_config.data_plane["enabled"]:
        raise ValueError(
            "Turn-level credit research supports only data_plane.enabled=false"
        )
    if master_config.env.get("should_use_nemo_gym"):
        raise ValueError(
            "Native NeMo Gym step rewards require the versioned contract tracked "
            "by NVIDIA-NeMo/Gym#1298 and are not inferred by this project"
        )
    if _uses_async_rollouts(master_config):
        raise ValueError(
            "Turn-level credit research currently supports only synchronous "
            "native rollouts"
        )
