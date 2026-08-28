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

"""Canonical Transformer Engine MXFP8 source storage for refit."""

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NativeMXFP8Components:
    """Canonical rowwise MXFP8 weight bytes and compact E8M0 scales."""

    weight: torch.Tensor
    weight_scale: torch.Tensor


def extract_native_mxfp8_components(tensor: Any) -> NativeMXFP8Components:
    """Return view-only canonical MXFP8 components from TE metadata.

    The source object's nominal dtype is deliberately ignored because Transformer
    Engine exposes a fake high-precision dtype for quantized parameters.
    """
    metadata_getter = getattr(tensor, "get_metadata", None)
    if not callable(metadata_getter):
        raise ValueError("Native MXFP8 refit source must provide get_metadata()")
    metadata = metadata_getter()
    if not isinstance(metadata, Mapping):
        raise ValueError("Native MXFP8 refit metadata must be a mapping")
    if metadata.get("with_gemm_swizzled_scales") is not False:
        raise ValueError("Native MXFP8 refit requires compact rowwise scales")
    _validate_e4m3_format(metadata)

    shape = _logical_shape(getattr(tensor, "shape", None))
    k = shape[-1]
    if k % 32:
        raise ValueError(
            f"Native MXFP8 refit requires K divisible by 32; got logical shape {shape}"
        )
    rows = math.prod(shape[:-1])
    scale_columns = k // 32
    data = metadata.get("rowwise_data")
    scale = metadata.get("rowwise_scale_inv")
    _validate_native_storage(data, scale, rows, k, scale_columns)
    assert isinstance(data, torch.Tensor)
    assert isinstance(scale, torch.Tensor)

    return NativeMXFP8Components(
        weight=data.view(torch.float8_e4m3fn).view(shape),
        weight_scale=scale[:rows, :scale_columns].view((*shape[:-1], scale_columns)),
    )


def _logical_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Native MXFP8 refit source must provide a logical shape")
    shape = tuple(value)
    if not shape:
        raise ValueError("Native MXFP8 refit logical shape must be non-empty")
    if any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in shape
    ):
        raise ValueError(
            "Native MXFP8 refit logical shape must contain positive integers"
        )
    return shape


def _validate_e4m3_format(metadata: Mapping[str, Any]) -> None:
    if "fp8_dtype" not in metadata:
        raise ValueError("Native MXFP8 refit metadata must include fp8_dtype")
    try:
        expected_dtype = importlib.import_module(
            "transformer_engine_torch"
        ).DType.kFloat8E4M3
    except ModuleNotFoundError as error:
        if error.name != "transformer_engine_torch":
            raise
        raise ValueError(
            "Native MXFP8 refit cannot validate fp8_dtype because "
            "transformer_engine_torch.DType.kFloat8E4M3 is unavailable"
        ) from error
    except AttributeError as error:
        raise ValueError(
            "Native MXFP8 refit cannot validate fp8_dtype because "
            "transformer_engine_torch.DType.kFloat8E4M3 is unavailable"
        ) from error
    if metadata["fp8_dtype"] is not expected_dtype:
        raise ValueError(
            "Native MXFP8 refit fp8_dtype must be "
            "transformer_engine_torch.DType.kFloat8E4M3"
        )


def _validate_native_storage(
    data: object,
    scale: object,
    rows: int,
    k: int,
    scale_columns: int,
) -> None:
    if not isinstance(data, torch.Tensor):
        raise ValueError("Native MXFP8 refit rowwise_data must be a torch.Tensor")
    if not isinstance(scale, torch.Tensor):
        raise ValueError("Native MXFP8 refit rowwise_scale_inv must be a torch.Tensor")
    if data.dtype is not torch.uint8:
        raise ValueError("Native MXFP8 refit rowwise_data must have torch.uint8 dtype")
    if scale.dtype is not torch.uint8:
        raise ValueError(
            "Native MXFP8 refit rowwise_scale_inv must have torch.uint8 dtype"
        )
    expected_data_elements = rows * k
    if data.numel() != expected_data_elements:
        raise ValueError(
            "Native MXFP8 refit rowwise_data must contain exactly "
            f"{expected_data_elements} value bytes"
        )
    if not data.is_contiguous():
        raise ValueError("Native MXFP8 refit rowwise_data must be contiguous")
    if scale.ndim != 2:
        raise ValueError("Native MXFP8 refit rowwise_scale_inv must be two-dimensional")
    if not scale.is_contiguous():
        raise ValueError("Native MXFP8 refit rowwise_scale_inv must be contiguous")
    if scale.shape[0] < rows or scale.shape[1] < scale_columns:
        raise ValueError(
            "Native MXFP8 refit rowwise_scale_inv must be at least "
            f"[{rows}, {scale_columns}]"
        )
