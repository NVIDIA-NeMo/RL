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

"""Soft-token conditioning helpers."""

from contextlib import contextmanager
from collections.abc import Mapping, MutableMapping
from typing import Any, Iterator

import torch
from torch import nn
from torch.distributed.tensor import DTensor

SOFT_TOKEN_EMBEDDINGS_KEY = "soft_token_embeddings"
SOFT_TOKEN_POSITIONS_KEY = "soft_token_positions"
VLLM_PROMPT_EMBEDS_KEY = "vllm_prompt_embeds"
SOFT_TOKEN_KEYS = (SOFT_TOKEN_EMBEDDINGS_KEY, SOFT_TOKEN_POSITIONS_KEY)


def has_soft_token_inputs(data: Mapping[str, Any]) -> bool:
    has_embeddings = SOFT_TOKEN_EMBEDDINGS_KEY in data
    has_positions = SOFT_TOKEN_POSITIONS_KEY in data
    if has_embeddings != has_positions:
        raise ValueError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} and {SOFT_TOKEN_POSITIONS_KEY} must be "
            "provided together."
        )
    return has_embeddings


def copy_soft_token_inputs(
    source: Mapping[str, Any], destination: MutableMapping[str, Any]
) -> None:
    if not has_soft_token_inputs(source):
        return

    for key in SOFT_TOKEN_KEYS:
        destination[key] = source[key]


def has_vllm_prompt_embeds(data: Mapping[str, Any]) -> bool:
    return VLLM_PROMPT_EMBEDS_KEY in data


def copy_vllm_prompt_embeds(
    source: Mapping[str, Any], destination: MutableMapping[str, Any]
) -> None:
    if has_vllm_prompt_embeds(source):
        destination[VLLM_PROMPT_EMBEDS_KEY] = source[VLLM_PROMPT_EMBEDS_KEY]


def get_sample_soft_token_inputs(
    source: Mapping[str, Any], sample_idx: int
) -> dict[str, torch.Tensor]:
    if not has_soft_token_inputs(source):
        return {}

    sample_inputs = {}
    for key in SOFT_TOKEN_KEYS:
        value = source[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{key} must be a torch.Tensor, got {type(value)}")
        sample_inputs[key] = value[sample_idx : sample_idx + 1]
    return sample_inputs


def get_sample_vllm_prompt_embeds(
    source: Mapping[str, Any], sample_idx: int
) -> dict[str, torch.Tensor]:
    if not has_vllm_prompt_embeds(source):
        return {}

    prompt_embeds = source[VLLM_PROMPT_EMBEDS_KEY]
    if not torch.is_tensor(prompt_embeds):
        raise TypeError(
            f"{VLLM_PROMPT_EMBEDS_KEY} must be a torch.Tensor, got "
            f"{type(prompt_embeds)}"
        )
    return {VLLM_PROMPT_EMBEDS_KEY: prompt_embeds[sample_idx : sample_idx + 1]}


def get_vllm_prompt_embeds_for_sample(
    source: Mapping[str, Any], sample_idx: int, valid_length: int
) -> torch.Tensor:
    prompt_embeds = source[VLLM_PROMPT_EMBEDS_KEY]
    if not torch.is_tensor(prompt_embeds):
        raise TypeError(
            f"{VLLM_PROMPT_EMBEDS_KEY} must be a torch.Tensor, got "
            f"{type(prompt_embeds)}"
        )
    if prompt_embeds.ndim != 3:
        raise ValueError(
            f"{VLLM_PROMPT_EMBEDS_KEY} must have shape [B, S, H], got "
            f"{tuple(prompt_embeds.shape)}"
        )
    if sample_idx >= prompt_embeds.shape[0]:
        raise ValueError(
            f"sample_idx {sample_idx} is outside {VLLM_PROMPT_EMBEDS_KEY} batch "
            f"size {prompt_embeds.shape[0]}"
        )
    if valid_length > prompt_embeds.shape[1]:
        raise ValueError(
            f"{VLLM_PROMPT_EMBEDS_KEY} sequence length {prompt_embeds.shape[1]} is "
            f"shorter than input length {valid_length}"
        )
    return prompt_embeds[sample_idx, :valid_length]


def _validate_soft_token_shapes(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    soft_token_embeddings: torch.Tensor,
    soft_token_positions: torch.Tensor,
) -> torch.Tensor:
    if soft_token_embeddings.ndim != 3:
        raise ValueError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} must have shape [B, K, H], got "
            f"{tuple(soft_token_embeddings.shape)}"
        )
    if soft_token_positions.ndim != 2:
        raise ValueError(
            f"{SOFT_TOKEN_POSITIONS_KEY} must have shape [B, K], got "
            f"{tuple(soft_token_positions.shape)}"
        )
    if soft_token_embeddings.shape[:2] != soft_token_positions.shape:
        raise ValueError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} shape {tuple(soft_token_embeddings.shape)} "
            f"does not match {SOFT_TOKEN_POSITIONS_KEY} shape "
            f"{tuple(soft_token_positions.shape)}"
        )
    if soft_token_embeddings.shape[0] != input_ids.shape[0]:
        raise ValueError(
            f"soft-token batch size {soft_token_embeddings.shape[0]} does not "
            f"match input_ids batch size {input_ids.shape[0]}"
        )
    if soft_token_embeddings.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError(
            f"soft-token hidden size {soft_token_embeddings.shape[-1]} does not "
            f"match model hidden size {inputs_embeds.shape[-1]}"
        )

    positions = soft_token_positions.to(device=input_ids.device, dtype=torch.long)
    if torch.any(positions < -1):
        raise ValueError(f"{SOFT_TOKEN_POSITIONS_KEY} may only use -1 as padding")
    valid_positions = positions >= 0
    if torch.any(positions[valid_positions] >= input_ids.shape[1]):
        raise ValueError(
            f"{SOFT_TOKEN_POSITIONS_KEY} contains positions outside the input "
            f"sequence length {input_ids.shape[1]}"
        )
    return positions


def apply_soft_token_embeddings(
    model: nn.Module, model_args: dict[str, Any]
) -> dict[str, Any]:
    if not has_soft_token_inputs(model_args):
        return model_args

    input_ids = model_args.pop("input_ids", None)
    if input_ids is None:
        raise ValueError("input_ids are required when soft-token inputs are present")

    soft_token_embeddings = model_args.pop(SOFT_TOKEN_EMBEDDINGS_KEY)
    soft_token_positions = model_args.pop(SOFT_TOKEN_POSITIONS_KEY)
    if not torch.is_tensor(soft_token_embeddings):
        raise TypeError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} must be a torch.Tensor, got "
            f"{type(soft_token_embeddings)}"
        )
    if not torch.is_tensor(soft_token_positions):
        raise TypeError(
            f"{SOFT_TOKEN_POSITIONS_KEY} must be a torch.Tensor, got "
            f"{type(soft_token_positions)}"
        )

    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if get_input_embeddings is None:
        raise ValueError("model must expose an input embedding module")

    embedding_module = get_input_embeddings()
    if embedding_module is None:
        raise ValueError("model must expose an input embedding module")

    inputs_embeds = embedding_module(input_ids)
    if isinstance(inputs_embeds, DTensor):
        raise NotImplementedError(
            "soft-token embeddings currently require tensor_parallel_size=1"
        )

    positions = _validate_soft_token_shapes(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        soft_token_embeddings=soft_token_embeddings,
        soft_token_positions=soft_token_positions,
    )

    valid_positions = positions >= 0
    if torch.any(valid_positions):
        batch_indices = torch.arange(
            input_ids.shape[0], device=input_ids.device, dtype=torch.long
        ).unsqueeze(1)
        batch_indices = batch_indices.expand_as(positions)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[batch_indices[valid_positions], positions[valid_positions]] = (
            soft_token_embeddings.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )[valid_positions]
        )

    model_args["inputs_embeds"] = inputs_embeds
    return model_args


def _replace_embedding_output(
    input_ids: torch.Tensor | DTensor,
    inputs_embeds: torch.Tensor | DTensor,
    soft_token_embeddings: torch.Tensor,
    soft_token_positions: torch.Tensor,
) -> torch.Tensor | DTensor:
    output_is_dtensor = isinstance(inputs_embeds, DTensor)
    local_input_ids = (
        input_ids.to_local() if isinstance(input_ids, DTensor) else input_ids
    )
    local_inputs_embeds = (
        inputs_embeds.to_local() if output_is_dtensor else inputs_embeds
    )

    positions = _validate_soft_token_shapes(
        input_ids=local_input_ids,
        inputs_embeds=local_inputs_embeds,
        soft_token_embeddings=soft_token_embeddings,
        soft_token_positions=soft_token_positions,
    )

    valid_positions = positions >= 0
    if torch.any(valid_positions):
        batch_indices = torch.arange(
            local_input_ids.shape[0], device=local_input_ids.device, dtype=torch.long
        ).unsqueeze(1)
        batch_indices = batch_indices.expand_as(positions)
        local_inputs_embeds = local_inputs_embeds.clone()
        local_inputs_embeds[
            batch_indices[valid_positions], positions[valid_positions]
        ] = soft_token_embeddings.to(
            device=local_inputs_embeds.device, dtype=local_inputs_embeds.dtype
        )[valid_positions]

    if output_is_dtensor:
        return DTensor.from_local(
            local_inputs_embeds,
            device_mesh=inputs_embeds.device_mesh,
            placements=inputs_embeds.placements,
        )
    return local_inputs_embeds


@contextmanager
def soft_token_embedding_override(
    model: nn.Module, model_args: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    if not has_soft_token_inputs(model_args):
        yield model_args
        return

    input_ids = model_args.get("input_ids", None)
    if input_ids is None:
        raise ValueError("input_ids are required when soft-token inputs are present")

    soft_token_embeddings = model_args.pop(SOFT_TOKEN_EMBEDDINGS_KEY)
    soft_token_positions = model_args.pop(SOFT_TOKEN_POSITIONS_KEY)
    if not torch.is_tensor(soft_token_embeddings):
        raise TypeError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} must be a torch.Tensor, got "
            f"{type(soft_token_embeddings)}"
        )
    if not torch.is_tensor(soft_token_positions):
        raise TypeError(
            f"{SOFT_TOKEN_POSITIONS_KEY} must be a torch.Tensor, got "
            f"{type(soft_token_positions)}"
        )

    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if get_input_embeddings is None:
        raise ValueError("model must expose an input embedding module")

    embedding_module = get_input_embeddings()
    if embedding_module is None:
        raise ValueError("model must expose an input embedding module")

    def hook(
        module: nn.Module, args: tuple[Any, ...], output: torch.Tensor | DTensor
    ) -> torch.Tensor | DTensor:
        del module
        hook_input_ids = args[0] if args else input_ids
        return _replace_embedding_output(
            input_ids=hook_input_ids,
            inputs_embeds=output,
            soft_token_embeddings=soft_token_embeddings,
            soft_token_positions=soft_token_positions,
        )

    handle = embedding_module.register_forward_hook(hook)
    try:
        yield model_args
    finally:
        handle.remove()
