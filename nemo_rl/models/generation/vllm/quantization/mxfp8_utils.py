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

from typing import Any

import torch


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def flashinfer_scale_k_pad_width(k: int) -> int:
    if k < 0:
        raise ValueError("MXFP8 scale K dimension must be non-negative")
    return (-k) % 4


def pad_flashinfer_scale_k(input_tensor: Any) -> Any:
    pad_width = flashinfer_scale_k_pad_width(input_tensor.shape[-1])
    if pad_width == 0:
        return input_tensor

    padded_shape = (*input_tensor.shape[:-1], input_tensor.shape[-1] + pad_width)
    padded = input_tensor.new_zeros(padded_shape)
    padded[..., : input_tensor.shape[-1]] = input_tensor
    return padded


def flashinfer_mxfp8_moe_padding_plan(
    hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    """Compute FlashInfer TRTLLM MXFP8 MoE execution dimensions.

    Args:
        hidden_size: Unpadded model hidden size.
        intermediate_size: Unpadded TP-local MoE intermediate size.

    Returns:
        A ``(padded_hidden_size, padded_intermediate_size)`` tuple, rounded up to the
        kernel's required alignment (512 / 128 respectively).
    """
    """Compute FlashInfer TRTLLM MXFP8 MoE execution dimensions.

    Args:
        hidden_size: Unpadded model hidden size.
        intermediate_size: Unpadded TP-local MoE intermediate size.

    Returns:
        The padded hidden and intermediate sizes required by the kernel.
    """
    if hidden_size <= 0:
        raise ValueError("MXFP8 MoE hidden_size must be positive")
    if hidden_size % 32 != 0:
        raise ValueError(
            "FlashInfer TRTLLM MXFP8 MoE requires hidden_size divisible by 32, "
            f"got {hidden_size}."
        )
    if intermediate_size <= 0:
        raise ValueError("MXFP8 MoE intermediate_size must be positive")
    if intermediate_size % 32 != 0:
        raise ValueError(
            "FlashInfer TRTLLM MXFP8 MoE requires intermediate_size divisible "
            f"by 32, got {intermediate_size}."
        )

    return _round_up(hidden_size, 512), _round_up(intermediate_size, 128)


def pad_tensor_dim(
    input_tensor: torch.Tensor,
    dim: int,
    padded_size: int,
    pad_value: int | float = 0,
) -> torch.Tensor:
    """Pad one tensor dimension without changing its existing values."""
    current_size = input_tensor.shape[dim]
    if current_size == padded_size:
        return input_tensor
    if current_size > padded_size:
        raise ValueError(
            f"Cannot pad MXFP8 tensor dim {dim} from {current_size} to {padded_size}."
        )

    padded_shape = list(input_tensor.shape)
    padded_shape[dim] = padded_size
    padded = input_tensor.new_full(padded_shape, pad_value)
    padded.narrow(dim, 0, current_size).copy_(input_tensor)
    return padded


def pad_w13_intermediate(
    input_tensor: torch.Tensor,
    padded_intermediate_size: int,
    is_gated: bool,
    pad_value: int | float = 0,
) -> torch.Tensor:
    """Pad W13 while preserving separate gate and up-projection halves."""
    if not is_gated:
        return pad_tensor_dim(
            input_tensor,
            dim=1,
            padded_size=padded_intermediate_size,
            pad_value=pad_value,
        )
    if input_tensor.shape[1] % 2 != 0:
        raise ValueError(
            "Gated MXFP8 W13 must contain equal gate and up-projection halves."
        )

    intermediate_size = input_tensor.shape[1] // 2
    sharded = input_tensor.reshape(
        input_tensor.shape[0],
        2,
        intermediate_size,
        *input_tensor.shape[2:],
    )
    padded_shape = list(sharded.shape)
    padded_shape[2] = padded_intermediate_size
    padded = input_tensor.new_full(padded_shape, pad_value)
    padded[:, :, :intermediate_size].copy_(sharded)
    return padded.reshape(
        input_tensor.shape[0],
        2 * padded_intermediate_size,
        *input_tensor.shape[2:],
    )


def assign_or_replace_parameter(
    layer: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    *,
    force_replace: bool = False,
) -> None:
    """Update compatible parameter storage or install new runtime storage.

    Args:
        layer: Module that owns the parameter.
        name: Parameter attribute name.
        value: New parameter value.
        force_replace: Replace compatible storage instead of copying into it.
            This separates runtime storage from an aliased checkpoint parameter.
    """
    value = value.contiguous()
    parameter = getattr(layer, name, None)
    can_copy = (
        not force_replace
        and isinstance(parameter, torch.nn.Parameter)
        and parameter.shape == value.shape
        and parameter.dtype == value.dtype
        and parameter.device == value.device
    )
    if can_copy:
        with torch.no_grad():
            parameter.copy_(value)
        return

    # New runtime storage must not alias reusable scratch or checkpoint storage.
    setattr(layer, name, torch.nn.Parameter(value.clone(), requires_grad=False))
