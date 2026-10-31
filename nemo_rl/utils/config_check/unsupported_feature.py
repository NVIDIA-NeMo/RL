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
"""Checks configuration for unsupported features before worker startup."""

from enum import Enum
from typing import Any


class EntryPoint(str, Enum):
    """Execution entrypoints with distinct feature support."""

    DISTILLATION = "distillation"
    DPO = "dpo"
    EVAL = "eval"
    GRPO = "grpo"
    PPO = "ppo"
    RM = "rm"
    SFT = "sft"
    SFT_V2 = "sft_v2"
    SINGLE_CONTROLLER = "single_controller"
    VLM_GRPO = "vlm_grpo"
    VLM_MPO = "vlm_mpo"
    XTOKEN_OFF_POLICY_DISTILLATION = "xtoken_off_policy_distillation"


def check_common(config: dict[str, Any]) -> None:
    """Reject unsupported configurations shared by all entrypoints."""
    pass


def _check_algo_distillation(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for distillation."""
    pass


def _check_algo_dpo(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for dpo."""
    pass


def _check_algo_eval(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for eval."""
    pass


def _check_algo_grpo(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for grpo."""
    pass


def _check_algo_ppo(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for ppo."""
    pass


def _check_algo_rm(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for rm."""
    pass


def _check_algo_sft(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for sft."""
    pass


def _check_algo_sft_v2(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for sft_v2."""
    pass


def _check_algo_single_controller(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for single_controller."""
    pass


def _check_algo_vlm_grpo(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for vlm_grpo."""
    pass


def _check_algo_vlm_mpo(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for vlm_mpo."""
    pass


def _check_algo_xtoken_off_policy_distillation(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for xtoken_off_policy_distillation."""
    pass


def check_algo(config: dict[str, Any], *, entrypoint: EntryPoint) -> None:
    """Reject unsupported configurations for the selected algorithm path.

    Args:
        config: The config resolved to plain dicts by OmegaConf.to_container.
        entrypoint: The execution entrypoint selecting the algorithm checks.
    """
    if entrypoint == EntryPoint.DISTILLATION:
        _check_algo_distillation(config)
    elif entrypoint == EntryPoint.DPO:
        _check_algo_dpo(config)
    elif entrypoint == EntryPoint.EVAL:
        _check_algo_eval(config)
    elif entrypoint == EntryPoint.GRPO:
        _check_algo_grpo(config)
    elif entrypoint == EntryPoint.PPO:
        _check_algo_ppo(config)
    elif entrypoint == EntryPoint.RM:
        _check_algo_rm(config)
    elif entrypoint == EntryPoint.SFT:
        _check_algo_sft(config)
    elif entrypoint == EntryPoint.SFT_V2:
        _check_algo_sft_v2(config)
    elif entrypoint == EntryPoint.SINGLE_CONTROLLER:
        _check_algo_single_controller(config)
    elif entrypoint == EntryPoint.VLM_GRPO:
        _check_algo_vlm_grpo(config)
    elif entrypoint == EntryPoint.VLM_MPO:
        _check_algo_vlm_mpo(config)
    elif entrypoint == EntryPoint.XTOKEN_OFF_POLICY_DISTILLATION:
        _check_algo_xtoken_off_policy_distillation(config)


def check_unsupported_feature(
    config: dict[str, Any], *, entrypoint: EntryPoint | None = None
) -> None:
    """Reject unsupported features and configuration combinations at startup.

    Call this on the resolved config before building MasterConfig. Common checks
    always run; entrypoint checks run when an entrypoint is provided. Each check
    should name the unsupported configuration and explain how to adjust it.

    Args:
        config: The config resolved to plain dicts by OmegaConf.to_container.
        entrypoint: The execution entrypoint, or None for entrypoint-independent checks.
    """
    check_common(config)
    if entrypoint is not None:
        check_algo(config, entrypoint=entrypoint)
