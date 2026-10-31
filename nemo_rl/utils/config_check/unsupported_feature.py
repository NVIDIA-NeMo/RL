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

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, NoReturn, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from nemo_rl.algorithms.distillation import MasterConfig as DistillationMasterConfig
    from nemo_rl.algorithms.ppo import MasterConfig as PPOMasterConfig
    from nemo_rl.algorithms.rm import MasterConfig as RMMasterConfig
    from nemo_rl.algorithms.sft import MasterConfig as SFTMasterConfig


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


def _raise_unsupported_feature(
    *,
    feature: str,
    condition: str | None = None,
    suggestion: str | None = None,
    tracking_url: str | None = None,
) -> NoReturn:
    """Raise ValueError with the feature, context, workaround, and tracking link."""
    message = f"Currently, {feature} is not supported"
    if condition:
        message += f" when {condition}"
    parts = [f"{message}."]
    if suggestion:
        parts.append(f"Please {suggestion}")
    if tracking_url:
        parts.append(f"Tracking: {tracking_url}")
    else:
        parts.append(
            "Please file an issue at https://github.com/NVIDIA-NeMo/RL if needed."
        )
    raise ValueError(" ".join(parts))


def check_common(config: BaseModel | dict[str, Any]) -> None:
    """Reject unsupported configurations shared by all entrypoints."""
    pass


def _check_algo_distillation(config: DistillationMasterConfig) -> None:
    """Reject unsupported configurations for distillation."""
    for section, who, backend_config in (
        ("policy", "student", config.policy),
        ("teacher", "teacher", config.teacher),
    ):
        automodel_cfg = backend_config.get("automodel_cfg")
        packing = backend_config.get("sequence_packing")
        if (
            automodel_cfg
            and automodel_cfg["enabled"]
            and automodel_cfg["sequence_parallel"]
            and packing
            and packing["enabled"]
        ):
            _raise_unsupported_feature(
                feature="sequence parallel + sequence packing",
                condition=f"distilling with an Automodel {who}",
                suggestion=f"set {section}.automodel_cfg.sequence_parallel=false.",
                tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/1178",
            )


def _check_algo_dpo(config: BaseModel) -> None:
    """Reject unsupported configurations for dpo."""
    pass


def _check_algo_eval(config: BaseModel) -> None:
    """Reject unsupported configurations for eval."""
    pass


def _check_algo_grpo(config: BaseModel) -> None:
    """Reject unsupported configurations for grpo."""
    pass


def _check_algo_ppo(config: PPOMasterConfig) -> None:
    """Reject unsupported configurations for ppo."""
    if config.loss_fn.seq_logprob_error_in_loss:
        _raise_unsupported_feature(
            feature="loss_fn.seq_logprob_error_in_loss",
            condition="using PPO",
            suggestion="disable it or use non-streaming GRPO.",
        )

    value_config = config.value
    megatron_cfg = value_config.get("megatron_cfg")
    if megatron_cfg and megatron_cfg["enabled"]:
        return
    automodel_cfg = value_config.get("automodel_cfg")
    if not automodel_cfg or not automodel_cfg["enabled"]:
        return
    packing = value_config.get("sequence_packing")
    if packing and packing["enabled"]:
        _raise_unsupported_feature(
            feature="sequence packing",
            condition="using the Automodel PPO value model",
            suggestion="set value.sequence_packing.enabled=false.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2951",
        )
    if automodel_cfg["context_parallel_size"] != 1:
        _raise_unsupported_feature(
            feature="context parallelism",
            condition="using the Automodel PPO value model",
            suggestion="set value.automodel_cfg.context_parallel_size=1.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2951",
        )


def _check_algo_rm(config: RMMasterConfig) -> None:
    """Reject unsupported configurations for rm."""
    policy_config = config.policy
    automodel_cfg = policy_config.get("automodel_cfg")
    if not automodel_cfg or not automodel_cfg["enabled"]:
        return
    if automodel_cfg["context_parallel_size"] > 1:
        _raise_unsupported_feature(
            feature="context parallelism",
            condition="using Automodel RM",
            suggestion="set policy.automodel_cfg.context_parallel_size=1.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2482",
        )


def _check_algo_sft(config: SFTMasterConfig) -> None:
    """Reject unsupported configurations for sft."""
    if config.data.get("backend") == "energon":
        _raise_unsupported_feature(
            feature="energon",
            condition="using SFT v1",
            suggestion="use SFT v2.",
        )
    if config.sft.defer_async_checkpoint_finalize:
        _raise_unsupported_feature(
            feature="sft.defer_async_checkpoint_finalize",
            condition="using SFT v1",
            suggestion="disable it or use SFT v2.",
        )


def _check_algo_sft_v2(config: BaseModel) -> None:
    """Reject unsupported configurations for sft_v2."""
    pass


def _check_algo_single_controller(config: BaseModel) -> None:
    """Reject unsupported configurations for single_controller."""
    pass


def _check_algo_vlm_grpo(config: BaseModel) -> None:
    """Reject unsupported configurations for vlm_grpo."""
    pass


def _check_algo_vlm_mpo(config: BaseModel) -> None:
    """Reject unsupported configurations for vlm_mpo."""
    pass


def _check_algo_xtoken_off_policy_distillation(config: BaseModel) -> None:
    """Reject unsupported configurations for xtoken_off_policy_distillation."""
    pass


def check_algo(config: BaseModel, *, entrypoint: EntryPoint) -> None:
    """Reject unsupported configurations for the selected algorithm path.

    Args:
        config: The validated MasterConfig for the selected algorithm.
        entrypoint: The execution entrypoint selecting the algorithm checks.
    """
    if entrypoint == EntryPoint.DISTILLATION:
        _check_algo_distillation(cast("DistillationMasterConfig", config))
    elif entrypoint == EntryPoint.DPO:
        _check_algo_dpo(config)
    elif entrypoint == EntryPoint.EVAL:
        _check_algo_eval(config)
    elif entrypoint == EntryPoint.GRPO:
        _check_algo_grpo(config)
    elif entrypoint == EntryPoint.PPO:
        _check_algo_ppo(cast("PPOMasterConfig", config))
    elif entrypoint == EntryPoint.RM:
        _check_algo_rm(cast("RMMasterConfig", config))
    elif entrypoint == EntryPoint.SFT:
        _check_algo_sft(cast("SFTMasterConfig", config))
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
    config: BaseModel | dict[str, Any], *, entrypoint: EntryPoint | None = None
) -> None:
    """Reject unsupported features and configuration combinations at startup.

    Call this after building MasterConfig. Common checks always run;
    entrypoint checks run when an entrypoint is provided. Each check
    should name the unsupported configuration and explain how to adjust it.

    Args:
        config: The validated MasterConfig, or a resolved dict for tools without a schema.
        entrypoint: The execution entrypoint, or None for entrypoint-independent checks.
    """
    check_common(config)
    if entrypoint is not None:
        check_algo(cast(BaseModel, config), entrypoint=entrypoint)
