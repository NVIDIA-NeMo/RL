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
from typing import TYPE_CHECKING, NoReturn, cast

from pydantic import BaseModel

from nemo_rl.utils.config_check.utils import train_backend_configs

if TYPE_CHECKING:
    from nemo_rl.algorithms.distillation import MasterConfig as DistillationMasterConfig
    from nemo_rl.algorithms.dpo import MasterConfig as DPOMasterConfig
    from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
    from nemo_rl.algorithms.mpo import MasterConfig as MPOMasterConfig
    from nemo_rl.algorithms.ppo import MasterConfig as PPOMasterConfig
    from nemo_rl.algorithms.rm import MasterConfig as RMMasterConfig
    from nemo_rl.algorithms.sft import MasterConfig as SFTMasterConfig
    from nemo_rl.algorithms.sft_v2 import MasterConfig as SFTV2MasterConfig
    from nemo_rl.algorithms.single_controller_utils.config import (
        MasterConfig as SingleControllerMasterConfig,
    )
    from nemo_rl.models.policy import PolicyConfig


class EntryPoint(str, Enum):
    """Execution entrypoints with distinct feature support."""

    DISTILLATION = "distillation"
    DPO = "dpo"
    EVAL = "eval"
    GRPO = "grpo"
    MPO = "mpo"
    PPO = "ppo"
    RM = "rm"
    SFT = "sft"
    SFT_V2 = "sft_v2"
    SINGLE_CONTROLLER = "single_controller"
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


def _check_automodel(backend_config: PolicyConfig, *, path: str) -> None:
    """Reject unsupported Automodel training configurations."""
    # resolve config
    automodel_cfg = backend_config["automodel_cfg"]
    packing = backend_config.get("sequence_packing")
    hf_config_overrides = backend_config.get("hf_config_overrides")
    rope_scaling = (
        hf_config_overrides.get("rope_scaling") if hf_config_overrides else None
    )

    # yarn
    if rope_scaling and rope_scaling.get("rope_type") == "yarn":
        _raise_unsupported_feature(
            feature="YaRN RoPE scaling",
            condition=f"using Automodel for {path}",
            suggestion="use the Megatron backend.",
        )

    # context_parallel_size, sequence_packing
    if automodel_cfg["context_parallel_size"] > 1 and packing and packing["enabled"]:
        _raise_unsupported_feature(
            feature="context parallel + sequence packing",
            condition=f"using Automodel for {path}",
            suggestion=f"set {path}.automodel_cfg.context_parallel_size=1 or disable sequence packing.",
            tracking_url="https://github.com/NVIDIA/NeMo-RL/blob/main/docs/model-quirks.md#context-parallel-with-automodel",
        )


def _check_megatron(backend_config: PolicyConfig, *, path: str) -> None:
    """Reject unsupported Megatron training configurations."""
    # resolve config
    megatron_cfg = backend_config["megatron_cfg"]
    packing = backend_config.get("sequence_packing")
    generation = backend_config.get("generation")
    draft = backend_config.get("draft")
    ddp_config = megatron_cfg.get("distributed_data_parallel_config")

    # draft, context_parallel_size
    if (
        draft is not None
        and draft.enabled
        and megatron_cfg["context_parallel_size"] > 1
    ):
        _raise_unsupported_feature(
            feature="draft training + context parallel",
            condition=f"using Megatron for {path}",
            suggestion=f"set {path}.megatron_cfg.context_parallel_size=1 or disable {path}.draft.",
        )

    if megatron_cfg.get("use_fused_linear_logprobs"):
        # overlap_param_gather
        if ddp_config and ddp_config.get("overlap_param_gather"):
            _raise_unsupported_feature(
                feature="fused linear logprobs + overlap_param_gather",
                suggestion=f"set {path}.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false.",
            )

        # sequence_packing
        if packing and packing["enabled"]:
            _raise_unsupported_feature(
                feature="fused linear logprobs + sequence packing",
                suggestion=f"set {path}.megatron_cfg.use_fused_linear_logprobs=false or {path}.sequence_packing.enabled=false.",
            )

        # top_k, top_p
        if generation is not None and (
            (generation["top_k"] is not None and generation["top_k"] > 0)
            or (generation["top_p"] is not None and generation["top_p"] != 1.0)
        ):
            _raise_unsupported_feature(
                feature="fused linear logprobs + top-k/top-p filtering",
                suggestion=f"set {path}.megatron_cfg.use_fused_linear_logprobs=false or disable filtering ({path}.generation.top_k=null, {path}.generation.top_p=1.0).",
            )


def _check_policy(config: BaseModel) -> None:
    """Reject unsupported configurations for each training backend."""
    # resolve config
    backend_configs = train_backend_configs(config)

    for path, backend_config in backend_configs:
        # resolve config
        megatron_cfg = backend_config.get("megatron_cfg")
        automodel_cfg = backend_config.get("automodel_cfg")
        dynamic_batching = backend_config.get("dynamic_batching")
        pp_size = (
            megatron_cfg["pipeline_model_parallel_size"]
            if megatron_cfg and megatron_cfg["enabled"]
            else 1
        )

        # dispatch checks
        if megatron_cfg and megatron_cfg["enabled"]:
            _check_megatron(backend_config, path=path)
        elif automodel_cfg and automodel_cfg["enabled"]:
            _check_automodel(backend_config, path=path)

        # dynamic_batching, pipeline_model_parallel_size
        if dynamic_batching and dynamic_batching["enabled"] and pp_size > 1:
            _raise_unsupported_feature(
                feature="dynamic batching + pipeline parallel",
                condition=f"training {path}",
                suggestion=f"disable {path}.dynamic_batching or set {path}.megatron_cfg.pipeline_model_parallel_size=1.",
            )


def _check_megatron_generation(
    config: BaseModel, *, entrypoint: EntryPoint | None
) -> None:
    """Reject unsupported Megatron generation configurations."""
    # resolve config
    policy_config = getattr(config, "policy", None)
    router_replay = (
        policy_config.get("router_replay") if policy_config is not None else None
    )

    # router_replay
    if (
        router_replay
        and router_replay.get("enabled")
        and entrypoint != EntryPoint.SINGLE_CONTROLLER
    ):
        _raise_unsupported_feature(
            feature="Megatron-generation router replay outside SingleController",
            suggestion="use the SingleController token-capture path.",
        )


def _check_dynamo(config: BaseModel) -> None:
    """Reject unsupported managed Dynamo configurations."""
    # resolve config
    grpo_config = getattr(config, "grpo", None)
    if grpo_config is None:
        return
    async_grpo = grpo_config.async_grpo
    in_flight_weight_updates = (
        async_grpo is not None and async_grpo.in_flight_weight_updates
    )

    # in_flight_weight_updates
    if in_flight_weight_updates:
        _raise_unsupported_feature(
            feature="grpo.async_grpo.in_flight_weight_updates",
            condition="using managed Dynamo",
            suggestion="set grpo.async_grpo.in_flight_weight_updates=false.",
        )


def _check_generation(config: BaseModel, *, entrypoint: EntryPoint | None) -> None:
    """Reject unsupported generation configurations."""
    # resolve config
    policy_config = getattr(config, "policy", None)
    generation = getattr(config, "generation", None)
    env_config = getattr(config, "env", None)
    if policy_config is not None:
        generation = policy_config.get("generation")
    if generation is None:
        return
    backend = generation["backend"]
    enable_nemo_gym = bool(env_config and env_config.get("should_use_nemo_gym"))
    refit_transport = generation.get("refit_transport")
    val_sampling_overridden = (
        generation["val_temperature"] != generation["temperature"]
        or generation["val_top_p"] != generation["top_p"]
        or generation["val_top_k"] != generation["top_k"]
    )

    # dispatch checks
    if backend == "megatron":
        _check_megatron_generation(config, entrypoint=entrypoint)
    elif backend == "dynamo":
        _check_dynamo(config)

    # validation sampling
    if val_sampling_overridden:
        if backend != "vllm" or not enable_nemo_gym:
            _raise_unsupported_feature(
                feature="validation sampling overrides without vLLM + NeMo-Gym",
                suggestion="match validation sampling parameters to training or use vLLM NeMo-Gym rollouts.",
            )

        # val_top_k
        if generation["val_top_k"]:
            _raise_unsupported_feature(
                feature="generation.val_top_k overrides",
                condition="using NeMo-Gym",
                suggestion="leave policy.generation.val_top_k=null.",
            )

    # refit_transport
    if refit_transport is not None and not (
        backend == "vllm"
        or (backend == "megatron" and refit_transport in ("mcore", "nccl_reshard"))
    ):
        _raise_unsupported_feature(
            feature=f"refit_transport={refit_transport!r}",
            condition=f"using generation backend={backend!r}",
            suggestion="set policy.generation.refit_transport=null, or use vLLM or Megatron (mcore/nccl_reshard).",
        )


def check_common(config: BaseModel, *, entrypoint: EntryPoint | None = None) -> None:
    """Reject unsupported configurations shared by all entrypoints."""
    _check_policy(config)
    _check_generation(config, entrypoint=entrypoint)


def _check_algo_distillation(config: DistillationMasterConfig) -> None:
    """Reject unsupported configurations for distillation."""
    # resolve config
    backend_configs = (
        ("policy", "student", config.policy),
        ("teacher", "teacher", config.teacher),
    )
    generation = config.policy.get("generation")

    # sequence parallel + sequence packing
    for section, who, backend_config in backend_configs:
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

    # non-colocated megatron generation
    if (
        generation is not None
        and generation["backend"] == "megatron"
        and not generation["colocated"]["enabled"]
    ):
        _raise_unsupported_feature(
            feature="non-colocated Megatron generation",
            condition="using distillation",
            suggestion="set policy.generation.colocated.enabled=true or use vLLM.",
        )


def _check_algo_dpo(config: DPOMasterConfig) -> None:
    """Reject unsupported configurations for dpo."""
    # resolve config
    policy_config = config.policy
    dynamic_batching = policy_config["dynamic_batching"]
    packing = policy_config.get("sequence_packing")

    # dynamic_batching
    if dynamic_batching["enabled"]:
        _raise_unsupported_feature(
            feature="dynamic batching",
            condition="using DPO",
            suggestion="set policy.dynamic_batching.enabled=false.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/719",
        )

    # sequence_packing
    if packing and packing["enabled"]:
        _raise_unsupported_feature(
            feature="sequence packing",
            condition="using DPO",
            suggestion="set policy.sequence_packing.enabled=false.",
            tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/719",
        )


def _check_algo_eval(config: BaseModel) -> None:
    """Reject unsupported configurations for eval."""
    pass


def _check_algo_grpo(config: GRPOMasterConfig) -> None:
    """Reject unsupported configurations for grpo."""
    # resolve config
    grpo_config = config.grpo

    # async grpo
    if grpo_config.async_grpo.enabled:
        unsupported = [
            name
            for name, enabled in (
                ("grpo.use_dynamic_sampling", grpo_config.use_dynamic_sampling),
                ("grpo.reward_scaling", grpo_config.reward_scaling.enabled),
                ("grpo.reward_shaping", grpo_config.reward_shaping.enabled),
                (
                    "data.use_multiple_dataloader",
                    config.data["use_multiple_dataloader"],
                ),
            )
            if enabled
        ]
        if unsupported:
            _raise_unsupported_feature(
                feature=", ".join(unsupported),
                condition="using async GRPO",
                suggestion="disable these options.",
            )


def _check_algo_mpo(config: MPOMasterConfig) -> None:
    """Reject unsupported configurations for mpo."""
    # resolve config
    policy_config = config.policy
    dynamic_batching = policy_config["dynamic_batching"]
    packing = policy_config.get("sequence_packing")
    megatron_cfg = policy_config.get("megatron_cfg")

    # dynamic_batching
    if dynamic_batching["enabled"]:
        _raise_unsupported_feature(
            feature="dynamic batching",
            condition="using MPO",
            suggestion="set policy.dynamic_batching.enabled=false.",
        )

    if packing and packing["enabled"]:
        # sequence_packing
        if not megatron_cfg or not megatron_cfg["enabled"]:
            _raise_unsupported_feature(
                feature="sequence packing without Megatron",
                condition="using MPO",
                suggestion="use the Megatron backend or disable sequence packing.",
            )


def _check_algo_ppo(config: PPOMasterConfig) -> None:
    """Reject unsupported configurations for ppo."""
    # resolve config
    value_config = config.value
    value_automodel_cfg = value_config.get("automodel_cfg")
    value_packing = value_config.get("sequence_packing")
    loss_config = config.loss_fn

    # seq_logprob_error_in_loss
    if loss_config.seq_logprob_error_in_loss:
        _raise_unsupported_feature(
            feature="loss_fn.seq_logprob_error_in_loss",
            condition="using PPO",
            suggestion="disable it or use non-streaming GRPO.",
        )

    if value_automodel_cfg and value_automodel_cfg["enabled"]:
        # sequence_packing
        if value_packing and value_packing["enabled"]:
            _raise_unsupported_feature(
                feature="sequence packing",
                condition="using the Automodel PPO value model",
                suggestion="set value.sequence_packing.enabled=false.",
                tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2951",
            )

        # context_parallel_size
        if value_automodel_cfg["context_parallel_size"] != 1:
            _raise_unsupported_feature(
                feature="context parallelism",
                condition="using the Automodel PPO value model",
                suggestion="set value.automodel_cfg.context_parallel_size=1.",
                tracking_url="https://github.com/NVIDIA-NeMo/RL/issues/2951",
            )


def _check_algo_rm(config: RMMasterConfig) -> None:
    """Reject unsupported configurations for rm."""
    # resolve config
    policy_config = config.policy
    automodel_cfg = policy_config.get("automodel_cfg")

    # context_parallel_size
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
    # resolve config
    data_config = config.data
    sft_config = config.sft

    # energon
    if data_config.get("backend") == "energon":
        _raise_unsupported_feature(
            feature="energon",
            condition="using SFT v1",
            suggestion="use SFT v2.",
        )

    # defer_async_checkpoint_finalize
    if sft_config.defer_async_checkpoint_finalize:
        _raise_unsupported_feature(
            feature="sft.defer_async_checkpoint_finalize",
            condition="using SFT v1",
            suggestion="disable it or use SFT v2.",
        )


def _check_algo_sft_v2(config: SFTV2MasterConfig) -> None:
    """Reject unsupported configurations for sft_v2."""
    # resolve config
    policy_config = config.policy
    data_config = config.data
    sft_config = config.sft
    energon = data_config.get("energon")
    packing = energon.task_encoder.packing if energon is not None else None
    dynamic_batching = policy_config["dynamic_batching"]
    metric_name = config.checkpointing["metric_name"]
    megatron_cfg = policy_config.get("megatron_cfg")

    # dynamic_batching
    if packing is not None and dynamic_batching["enabled"]:
        _raise_unsupported_feature(
            feature="dynamic batching",
            condition="using Energon packing",
            suggestion="set policy.dynamic_batching.enabled=false.",
        )

    # validation
    if sft_config.val_period != 0 or sft_config.val_at_start or sft_config.val_at_end:
        _raise_unsupported_feature(
            feature="validation",
            condition="using SFT v2",
            suggestion="set sft.val_period=0, sft.val_at_start=false and sft.val_at_end=false.",
        )

    # data.validation
    if data_config.get("validation") is not None:
        _raise_unsupported_feature(
            feature="validation data",
            condition="using SFT v2",
            suggestion="set data.validation=null.",
        )

    # checkpointing.metric_name
    if metric_name is not None and metric_name.startswith("val:"):
        _raise_unsupported_feature(
            feature=f"checkpointing.metric_name={metric_name!r}",
            condition="using SFT v2",
            suggestion="use 'train:<metric>' or null.",
        )

    # hybridep
    if (
        packing is not None
        and megatron_cfg
        and megatron_cfg["enabled"]
        and megatron_cfg.get("moe_token_dispatcher_type") == "flex"
        and megatron_cfg.get("moe_flex_dispatcher_backend") == "hybridep"
    ):
        _raise_unsupported_feature(
            feature="hybridep flex dispatch",
            condition="using Energon packing",
            suggestion="disable Energon packing or use a different MoE dispatcher.",
        )


def _check_algo_single_controller(config: SingleControllerMasterConfig) -> None:
    """Reject unsupported configurations for single_controller."""
    from nemo_rl.algorithms.single_controller_utils.config import algo_config

    # resolve config
    algo = algo_config(config)
    algo_name = "ppo" if config.ppo is not None else "grpo"
    grpo_config = config.grpo
    loss_config = config.loss_fn
    backend_configs = train_backend_configs(config)

    # dynamic sampling, reward scaling/shaping, length penalty
    unsupported = [
        f"{algo_name}.{name}"
        for name, enabled in (
            ("use_dynamic_sampling", algo.use_dynamic_sampling),
            ("reward_scaling", algo.reward_scaling.enabled),
            ("reward_shaping", algo.reward_shaping.enabled),
        )
        if enabled
    ]
    if grpo_config is not None and grpo_config.length_penalty is not None:
        unsupported.append("grpo.length_penalty")
    if unsupported:
        _raise_unsupported_feature(
            feature=f"{', '.join(unsupported)} configuration",
            condition="using SingleController",
            suggestion="disable these options.",
        )

    # automodel
    for section, backend_config in backend_configs:
        automodel_cfg = backend_config.get("automodel_cfg")
        if automodel_cfg and automodel_cfg["enabled"]:
            _raise_unsupported_feature(
                feature=f"Automodel {section}",
                condition="using SingleController",
                suggestion=f"use the Megatron backend for {section}.",
            )

    # seq_logprob_error_in_loss
    if loss_config.seq_logprob_error_in_loss:
        _raise_unsupported_feature(
            feature="loss_fn.seq_logprob_error_in_loss",
            condition="using SingleController",
            suggestion="disable it or use non-streaming GRPO.",
        )

    # gdpo
    if algo.adv_estimator.name == "gdpo":
        _raise_unsupported_feature(
            feature="gdpo advantage estimation",
            condition="using SingleController",
            suggestion=f"set {algo_name}.adv_estimator.name=grpo.",
        )


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
        _check_algo_dpo(cast("DPOMasterConfig", config))
    elif entrypoint == EntryPoint.EVAL:
        _check_algo_eval(config)
    elif entrypoint == EntryPoint.GRPO:
        _check_algo_grpo(cast("GRPOMasterConfig", config))
    elif entrypoint == EntryPoint.MPO:
        _check_algo_mpo(cast("MPOMasterConfig", config))
    elif entrypoint == EntryPoint.PPO:
        _check_algo_ppo(cast("PPOMasterConfig", config))
    elif entrypoint == EntryPoint.RM:
        _check_algo_rm(cast("RMMasterConfig", config))
    elif entrypoint == EntryPoint.SFT:
        _check_algo_sft(cast("SFTMasterConfig", config))
    elif entrypoint == EntryPoint.SFT_V2:
        _check_algo_sft_v2(cast("SFTV2MasterConfig", config))
    elif entrypoint == EntryPoint.SINGLE_CONTROLLER:
        _check_algo_single_controller(cast("SingleControllerMasterConfig", config))
    elif entrypoint == EntryPoint.XTOKEN_OFF_POLICY_DISTILLATION:
        _check_algo_xtoken_off_policy_distillation(config)


def check_unsupported_feature(
    config: BaseModel, *, entrypoint: EntryPoint | None = None
) -> None:
    """Reject unsupported features and configuration combinations at startup.

    Call this after building MasterConfig. Common checks always run;
    entrypoint checks run when an entrypoint is provided. Each check
    should name the unsupported configuration and explain how to adjust it.

    Args:
        config: The validated MasterConfig.
        entrypoint: The execution entrypoint, or None for entrypoint-independent checks.
    """
    check_common(config, entrypoint=entrypoint)
    if entrypoint is not None:
        check_algo(config, entrypoint=entrypoint)
