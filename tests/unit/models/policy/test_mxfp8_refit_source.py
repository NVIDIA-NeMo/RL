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

"""Tests for canonical Transformer Engine MXFP8 source storage extraction."""

import sys
from collections.abc import Mapping
from types import ModuleType

import pytest
import torch

from nemo_rl.models.policy.workers.mxfp8_refit_source import (
    extract_native_mxfp8_components,
)

_DEFAULT_METADATA = object()


class DType:
    """Structural stand-in for ``transformer_engine_torch.DType``."""

    def __str__(self) -> str:
        return "DType.kFloat8E4M3"


PINNED_TE_E4M3 = DType()
DType.__module__ = "transformer_engine_torch"
DType.kFloat8E4M3 = PINNED_TE_E4M3


class ForgedDType:
    """Matches the old structural predicate without being the TE binding."""

    def __str__(self) -> str:
        return "DType.kFloat8E4M3"


ForgedDType.__module__ = "transformer_engine_torch"
ForgedDType.__name__ = "DType"
FORGED_TE_E4M3 = ForgedDType()


@pytest.fixture(autouse=True)
def transformer_engine_torch_binding(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install the minimal pinned TE enum binding only while a test executes."""
    module = ModuleType("transformer_engine_torch")
    module.DType = DType
    monkeypatch.setitem(sys.modules, "transformer_engine_torch", module)
    return module


class FakeMXFP8Tensor:
    """Structural stand-in for the supported Transformer Engine metadata API."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        rowwise_data: object = None,
        rowwise_scale_inv: object = None,
        with_gemm_swizzled_scales: object = False,
        fp8_dtype: object = PINNED_TE_E4M3,
        metadata: object = _DEFAULT_METADATA,
        columnwise_data: object = None,
        columnwise_scale_inv: object = None,
    ) -> None:
        self.shape = shape
        self._metadata = (
            {
                "rowwise_data": rowwise_data,
                "rowwise_scale_inv": rowwise_scale_inv,
                "with_gemm_swizzled_scales": with_gemm_swizzled_scales,
                "fp8_dtype": fp8_dtype,
                "columnwise_data": columnwise_data,
                "columnwise_scale_inv": columnwise_scale_inv,
            }
            if metadata is _DEFAULT_METADATA
            else metadata
        )

    def get_metadata(self) -> object:
        return self._metadata


def _bytes(count: int) -> torch.Tensor:
    return torch.arange(count, dtype=torch.int64).to(torch.uint8)


def _source(
    *,
    shape: tuple[int, ...] = (3, 64),
    rowwise_data: object | None = None,
    rowwise_scale_inv: object | None = None,
    **kwargs: object,
) -> FakeMXFP8Tensor:
    rows = int(torch.tensor(shape[:-1]).prod()) if len(shape) > 1 else 1
    data = _bytes(rows * shape[-1]) if rowwise_data is None else rowwise_data
    scale = (
        _bytes((rows + 2) * (shape[-1] // 32 + 2)).reshape(
            rows + 2, shape[-1] // 32 + 2
        )
        if rowwise_scale_inv is None
        else rowwise_scale_inv
    )
    return FakeMXFP8Tensor(
        shape=shape,
        rowwise_data=data,
        rowwise_scale_inv=scale,
        **kwargs,
    )


def test_extract_native_mxfp8_components_packs_padded_scale_rows() -> None:
    data = _bytes(3 * 64).reshape(3, 64)
    scale = _bytes(5 * 4).reshape(5, 4)
    source = _source(rowwise_data=data, rowwise_scale_inv=scale)

    result = extract_native_mxfp8_components(source)

    assert result.weight.dtype == torch.float8_e4m3fn
    assert result.weight.shape == (3, 64)
    assert torch.equal(result.weight.view(torch.uint8), data)
    assert result.weight.view(torch.uint8).data_ptr() == data.data_ptr()
    assert result.weight_scale.shape == (3, 2)
    assert torch.equal(result.weight_scale, scale[:3, :2])
    assert result.weight_scale.is_contiguous()
    assert result.weight_scale.data_ptr() != scale.data_ptr()


def test_extract_native_mxfp8_components_keeps_exact_scale_storage_zero_copy() -> None:
    data = _bytes(3 * 64).reshape(3, 64)
    scale = _bytes(3 * 2).reshape(3, 2)

    result = extract_native_mxfp8_components(
        _source(rowwise_data=data, rowwise_scale_inv=scale)
    )

    assert result.weight_scale.is_contiguous()
    assert result.weight_scale.data_ptr() == scale.data_ptr()


def test_extract_native_mxfp8_components_restores_grouped_leading_dimensions() -> None:
    data = _bytes(2 * 3 * 64).reshape(2, 3, 64)
    scale = _bytes(8 * 4).reshape(8, 4)
    source = _source(shape=(2, 3, 64), rowwise_data=data, rowwise_scale_inv=scale)

    result = extract_native_mxfp8_components(source)

    assert result.weight.shape == (2, 3, 64)
    assert torch.equal(result.weight.view(torch.uint8), data)
    assert result.weight_scale.shape == (2, 3, 2)
    assert torch.equal(result.weight_scale, scale[:6, :2].view(2, 3, 2))
    assert result.weight_scale.is_contiguous()


def test_extract_native_mxfp8_components_does_not_fall_back_to_columnwise_storage() -> (
    None
):
    columnwise_data = _bytes(3 * 64)
    columnwise_scale = _bytes(3 * 2).reshape(3, 2)
    source = FakeMXFP8Tensor(
        shape=(3, 64),
        rowwise_data=None,
        rowwise_scale_inv=None,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale,
    )

    with pytest.raises(ValueError, match="rowwise_data"):
        extract_native_mxfp8_components(source)


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (FakeMXFP8Tensor(shape=(3, 64), metadata=None), "metadata"),
        (
            FakeMXFP8Tensor(shape=(3, 64), metadata=[]),
            "metadata",
        ),
        (
            FakeMXFP8Tensor(
                shape=(3, 64),
                rowwise_data="not a tensor",
                rowwise_scale_inv=_bytes(20).reshape(5, 4),
            ),
            "rowwise_data",
        ),
        (
            FakeMXFP8Tensor(
                shape=(3, 64),
                rowwise_data=_bytes(192),
                rowwise_scale_inv="not a tensor",
            ),
            "rowwise_scale_inv",
        ),
        (
            _source(rowwise_data=torch.zeros(192, dtype=torch.float32)),
            "rowwise_data.*torch.uint8",
        ),
        (
            _source(rowwise_scale_inv=torch.zeros(5, 4, dtype=torch.float32)),
            "rowwise_scale_inv.*torch.uint8",
        ),
        (
            _source(rowwise_data=_bytes(191)),
            "exactly 192",
        ),
        (
            _source(rowwise_data=_bytes(193)),
            "exactly 192",
        ),
        (
            _source(rowwise_data=_bytes(192).reshape(3, 64).transpose(0, 1)),
            "contiguous",
        ),
        (
            _source(rowwise_scale_inv=_bytes(20).reshape(4, 5).transpose(0, 1)),
            "contiguous",
        ),
        (
            _source(rowwise_scale_inv=_bytes(4).reshape(2, 2)),
            "at least",
        ),
        (
            _source(rowwise_scale_inv=_bytes(5)),
            "two-dimensional",
        ),
        (_source(shape=(3, 63)), "divisible by 32"),
        (
            FakeMXFP8Tensor(
                shape=(),
                metadata={
                    "with_gemm_swizzled_scales": False,
                    "fp8_dtype": PINNED_TE_E4M3,
                },
            ),
            "non-empty",
        ),
        (_source(with_gemm_swizzled_scales=True), "compact rowwise scales"),
        (_source(with_gemm_swizzled_scales=None), "compact rowwise scales"),
        (_source(fp8_dtype=FORGED_TE_E4M3), "fp8_dtype"),
        (_source(fp8_dtype="DType.kFloat8E4M3"), "fp8_dtype"),
        (_source(fp8_dtype="DType.kFloat8E5M2"), "E4M3"),
        (_source(fp8_dtype="DType.kFloat8E4M3-compatible"), "fp8_dtype"),
    ],
)
def test_extract_native_mxfp8_components_rejects_invalid_storage(
    source: FakeMXFP8Tensor, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        extract_native_mxfp8_components(source)


def test_extract_native_mxfp8_components_rejects_missing_fp8_dtype() -> None:
    source = _source()
    metadata = source.get_metadata()
    assert isinstance(metadata, Mapping)
    del metadata["fp8_dtype"]

    with pytest.raises(ValueError, match="fp8_dtype"):
        extract_native_mxfp8_components(source)


def test_extract_native_mxfp8_components_accepts_pinned_te_e4m3_dtype(
    transformer_engine_torch_binding: ModuleType,
) -> None:
    assert transformer_engine_torch_binding.DType.kFloat8E4M3 is PINNED_TE_E4M3

    result = extract_native_mxfp8_components(_source(fp8_dtype=PINNED_TE_E4M3))

    assert result.weight.dtype == torch.float8_e4m3fn


def test_extract_native_mxfp8_components_requires_te_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "transformer_engine_torch")

    with pytest.raises(ValueError, match="transformer_engine_torch"):
        extract_native_mxfp8_components(_source())


def test_extract_native_mxfp8_components_rejects_object_without_metadata_api() -> None:
    class MissingMetadataAPI:
        shape = (3, 64)

    with pytest.raises(ValueError, match="get_metadata"):
        extract_native_mxfp8_components(MissingMetadataAPI())
