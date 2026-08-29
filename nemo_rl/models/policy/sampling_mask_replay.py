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

"""Configuration and message-log helpers for sampling-mask replay."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch

from nemo_rl.data_plane.schema import (
    SAMPLING_MASK_SIZES_FIELD,
    SAMPLING_MASK_TOKEN_IDS_FIELD,
)
from nemo_rl.models.policy import SamplingMaskReplayConfig


def _replay_config(config: Mapping[str, Any]) -> SamplingMaskReplayConfig:
    raw = config.get("sampling_mask_replay")
    if isinstance(raw, SamplingMaskReplayConfig):
        return raw
    return SamplingMaskReplayConfig.model_validate(raw or {})


def sampling_mask_replay_enabled(config: Mapping[str, Any]) -> bool:
    """Whether sampling-mask replay is enabled (false when absent)."""
    return _replay_config(config).enabled


def configure_vllm_for_sampling_mask_replay(
    config: Mapping[str, Any],
    *,
    use_nemo_gym: bool = False,
) -> None:
    """Validate replay constraints and request processed masks from vLLM."""
    if not isinstance(config, MutableMapping):
        raise TypeError("sampling-mask replay policy config must be mutable.")
    generation = config.get("generation") or {}
    vllm_kwargs = generation.get("vllm_kwargs") or {}
    if not sampling_mask_replay_enabled(config):
        if vllm_kwargs.get("return_sampling_mask", False):
            raise ValueError(
                "generation.vllm_kwargs.return_sampling_mask must not be set "
                "directly; enable policy.sampling_mask_replay.enabled so the "
                "rollout, data-plane, and trainer consumers are configured too."
            )
        return

    if generation.get("backend") != "vllm":
        raise ValueError("sampling_mask_replay.enabled requires vLLM generation.")

    temperature = generation.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError(
            "sampling_mask_replay.enabled requires generation.temperature > 0."
        )

    top_k = generation.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(
            "sampling_mask_replay.enabled requires a finite positive integer "
            "generation.top_k."
        )

    vllm_kwargs = generation.setdefault("vllm_kwargs", {})
    if vllm_kwargs.get("speculative_config") is not None:
        raise ValueError(
            "sampling_mask_replay.enabled does not support speculative decoding."
        )

    custom_processor_keys = (
        "logits_processors",
        "logits_processor",
        "logits_processor_pattern",
    )
    if any(
        owner.get(key)
        for owner in (generation, vllm_kwargs)
        for key in custom_processor_keys
    ):
        raise ValueError(
            "sampling_mask_replay.enabled does not support custom logits processors."
        )

    megatron_cfg = config.get("megatron_cfg") or {}
    if megatron_cfg.get("use_fused_linear_logprobs", False):
        raise ValueError(
            "sampling_mask_replay.enabled does not support fused linear logprobs."
        )

    sequence_packing = config.get("sequence_packing") or {}
    if sequence_packing.get("enabled", False):
        raise ValueError(
            "sampling_mask_replay.enabled does not support "
            "policy.sequence_packing.enabled=true yet."
        )

    vllm_cfg = generation.setdefault("vllm_cfg", {})
    if use_nemo_gym:
        raise ValueError(
            "sampling_mask_replay.enabled does not support NeMo-Gym rollouts."
        )
    if vllm_cfg.get("expose_http_server", False):
        raise ValueError(
            "sampling_mask_replay.enabled does not support OpenAI-compatible HTTP "
            "chat rollouts."
        )

    raw_env_vars = vllm_cfg.get("env_vars")
    if raw_env_vars is None:
        env_vars: MutableMapping[str, Any] = {}
        vllm_cfg["env_vars"] = env_vars
    elif isinstance(raw_env_vars, MutableMapping):
        env_vars = raw_env_vars
    else:
        raise TypeError(
            "sampling_mask_replay.enabled requires generation.vllm_cfg.env_vars "
            "to be a mapping or null."
        )
    configured_model_runner = env_vars.get("VLLM_USE_V2_MODEL_RUNNER")
    if configured_model_runner not in (None, 1, True, "1"):
        raise ValueError(
            "sampling_mask_replay.enabled requires vLLM Model Runner V2, but "
            "generation.vllm_cfg.env_vars.VLLM_USE_V2_MODEL_RUNNER explicitly "
            "disables it."
        )

    vllm_kwargs["return_sampling_mask"] = True
    vllm_cfg["logprobs_mode"] = "processed_logprobs"
    env_vars["VLLM_USE_V2_MODEL_RUNNER"] = "1"


def _validate_sampling_mask_pair(
    token_ids: Any,
    sizes: Any,
    *,
    token_count: int,
    context: str,
) -> int:
    if not isinstance(token_ids, torch.Tensor) or not isinstance(sizes, torch.Tensor):
        raise TypeError(f"{context} sampling-mask fields must both be tensors.")
    if token_ids.dtype != torch.int32 or sizes.dtype != torch.int32:
        raise TypeError(f"{context} sampling-mask fields must both use torch.int32.")
    if token_ids.device != sizes.device:
        raise ValueError(f"{context} sampling-mask fields must use the same device.")
    if token_ids.ndim != 2:
        raise ValueError(
            f"{context} {SAMPLING_MASK_TOKEN_IDS_FIELD} must have shape [tokens, K], "
            f"got {tuple(token_ids.shape)}."
        )
    if sizes.ndim != 1:
        raise ValueError(
            f"{context} {SAMPLING_MASK_SIZES_FIELD} must have shape [tokens], "
            f"got {tuple(sizes.shape)}."
        )
    if token_ids.shape[0] != token_count or sizes.shape[0] != token_count:
        raise ValueError(
            f"{context} sampling-mask token dimension must equal {token_count}; got "
            f"{token_ids.shape[0]} and {sizes.shape[0]}."
        )
    width = int(token_ids.shape[1])
    if width <= 0:
        raise ValueError(f"{context} sampling-mask width K must be positive.")
    if sizes.numel() and torch.any((sizes < 0) | (sizes > width)).item():
        raise ValueError(f"{context} sampling-mask sizes must be in [0, {width}].")
    return width


def attach_sampling_mask_to_assistant_message(
    assistant_message: dict[str, Any],
    generation_outputs: Mapping[str, Any],
    *,
    batch_index: int,
    input_length: int,
    total_length: int,
) -> bool:
    """Attach one generated response's support; reject a torn field pair."""
    has_ids = SAMPLING_MASK_TOKEN_IDS_FIELD in generation_outputs
    has_sizes = SAMPLING_MASK_SIZES_FIELD in generation_outputs
    if has_ids != has_sizes:
        raise RuntimeError(
            "Generation returned only one sampling-mask field; "
            f"{SAMPLING_MASK_TOKEN_IDS_FIELD} and {SAMPLING_MASK_SIZES_FIELD} "
            "must be returned together."
        )
    if not has_ids:
        return False

    token_ids = generation_outputs[SAMPLING_MASK_TOKEN_IDS_FIELD][
        batch_index, input_length:total_length
    ]
    sizes = generation_outputs[SAMPLING_MASK_SIZES_FIELD][
        batch_index, input_length:total_length
    ]
    response_tokens = total_length - input_length
    _validate_sampling_mask_pair(
        token_ids,
        sizes,
        token_count=response_tokens,
        context="generated assistant message",
    )
    assistant_message[SAMPLING_MASK_TOKEN_IDS_FIELD] = token_ids
    assistant_message[SAMPLING_MASK_SIZES_FIELD] = sizes
    return True


def backfill_missing_sampling_masks(
    message_logs: Sequence[list[dict[str, Any]]],
) -> None:
    """Zero-fill non-generated token rows once the batch's width K is known."""
    width: int | None = None
    mask_device: torch.device | None = None
    for log_index, message_log in enumerate(message_logs):
        for message_index, message in enumerate(message_log):
            has_ids = SAMPLING_MASK_TOKEN_IDS_FIELD in message
            has_sizes = SAMPLING_MASK_SIZES_FIELD in message
            if has_ids != has_sizes:
                raise RuntimeError(
                    f"message_log[{log_index}][{message_index}] has a torn "
                    "sampling-mask field pair."
                )
            if not has_ids:
                continue
            message_token_ids = message.get("token_ids")
            if not isinstance(message_token_ids, torch.Tensor):
                raise TypeError("Sampling-mask messages require tensor token_ids.")
            current_width = _validate_sampling_mask_pair(
                message[SAMPLING_MASK_TOKEN_IDS_FIELD],
                message[SAMPLING_MASK_SIZES_FIELD],
                token_count=int(message_token_ids.shape[0]),
                context=f"message_log[{log_index}][{message_index}]",
            )
            current_device = message[SAMPLING_MASK_TOKEN_IDS_FIELD].device
            if mask_device is not None and current_device != mask_device:
                raise ValueError("The message-log batch mixes sampling-mask devices.")
            if width is not None and current_width != width:
                raise ValueError(
                    "The message-log batch mixes sampling-mask widths "
                    f"{width} and {current_width}."
                )
            width = current_width
            mask_device = current_device

    if width is None:
        return
    assert mask_device is not None

    for message_log in message_logs:
        for message in message_log:
            if SAMPLING_MASK_TOKEN_IDS_FIELD in message:
                continue
            message_token_ids = message.get("token_ids")
            if not isinstance(message_token_ids, torch.Tensor):
                continue
            token_count = int(message_token_ids.shape[0])
            message[SAMPLING_MASK_TOKEN_IDS_FIELD] = torch.zeros(
                (token_count, width),
                dtype=torch.int32,
                device=mask_device,
            )
            message[SAMPLING_MASK_SIZES_FIELD] = torch.zeros(
                token_count,
                dtype=torch.int32,
                device=mask_device,
            )
