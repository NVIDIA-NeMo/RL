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
from typing import Any, NoReturn


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


def check_common(config: dict[str, Any]) -> None:
    """Reject unsupported configurations shared by all entrypoints."""
    pass


def _check_algo_distillation(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for distillation."""
    for section, who in (("policy", "student"), ("teacher", "teacher")):
        backend_config = config.get(section)
        if not isinstance(backend_config, dict):
            continue
        automodel_cfg = backend_config.get("automodel_cfg")
        packing = backend_config.get("sequence_packing")
        if (
            isinstance(automodel_cfg, dict)
            and automodel_cfg.get("enabled")
            and automodel_cfg.get("sequence_parallel")
            and isinstance(packing, dict)
            and packing.get("enabled")
        ):
            _raise_unsupported_feature(
                feature="sequence parallel + sequence packing",
                condition=f"distilling with an Automodel {who}",
                suggestion=f"set {section}.automodel_cfg.sequence_parallel=false.",
                tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/1178",
            )


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
    loss_config = config.get("loss_fn")
    if isinstance(loss_config, dict) and loss_config.get("seq_logprob_error_in_loss"):
        _raise_unsupported_feature(
            feature="loss_fn.seq_logprob_error_in_loss",
            condition="using PPO",
            suggestion="disable it or use non-streaming GRPO.",
        )

    value_config = config.get("value")
    if not isinstance(value_config, dict):
        return
    megatron_cfg = value_config.get("megatron_cfg")
    if isinstance(megatron_cfg, dict) and megatron_cfg.get("enabled"):
        return
    automodel_cfg = value_config.get("automodel_cfg")
    if not isinstance(automodel_cfg, dict) or not automodel_cfg.get("enabled"):
        return
    packing = value_config.get("sequence_packing")
    if isinstance(packing, dict) and packing.get("enabled"):
        _raise_unsupported_feature(
            feature="sequence packing",
            condition="using the Automodel PPO value model",
            suggestion="set value.sequence_packing.enabled=false.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2951",
        )
    cp_size = automodel_cfg.get("context_parallel_size")
    if cp_size is not None and cp_size != 1:
        _raise_unsupported_feature(
            feature="context parallelism",
            condition="using the Automodel PPO value model",
            suggestion="set value.automodel_cfg.context_parallel_size=1.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2951",
        )


def _check_algo_rm(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for rm."""
    policy_config = config.get("policy")
    if not isinstance(policy_config, dict):
        return
    automodel_cfg = policy_config.get("automodel_cfg")
    if not isinstance(automodel_cfg, dict) or not automodel_cfg.get("enabled"):
        return
    cp_size = automodel_cfg.get("context_parallel_size")
    if cp_size is not None and cp_size > 1:
        _raise_unsupported_feature(
            feature="context parallelism",
            condition="using Automodel RM",
            suggestion="set policy.automodel_cfg.context_parallel_size=1.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2482",
        )


def _check_algo_sft(config: dict[str, Any]) -> None:
    """Reject unsupported configurations for sft."""
    data_config = config.get("data")
    if isinstance(data_config, dict) and data_config.get("backend") == "energon":
        _raise_unsupported_feature(
            feature="energon",
            condition="using SFT v1",
            suggestion="use SFT v2.",
        )
    sft_config = config.get("sft")
    if isinstance(sft_config, dict) and sft_config.get(
        "defer_async_checkpoint_finalize"
    ):
        _raise_unsupported_feature(
            feature="sft.defer_async_checkpoint_finalize",
            condition="using SFT v1",
            suggestion="disable it or use SFT v2.",
        )


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
