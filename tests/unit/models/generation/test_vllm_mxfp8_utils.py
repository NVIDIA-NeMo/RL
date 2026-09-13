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

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_mxfp8_utils():
    module_path = (
        Path(__file__).parents[4]
        / "nemo_rl/models/generation/vllm/quantization/mxfp8_utils.py"
    )
    spec = importlib.util.spec_from_file_location("mxfp8_utils", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("k", "expected_pad_width"),
    [
        (0, 0),
        (1, 3),
        (4, 0),
        (6, 2),
        (8, 0),
    ],
)
def test_flashinfer_scale_k_pad_width_aligns_to_multiple_of_four(
    k: int, expected_pad_width: int
) -> None:
    flashinfer_scale_k_pad_width = _load_mxfp8_utils().flashinfer_scale_k_pad_width
    assert flashinfer_scale_k_pad_width(k) == expected_pad_width


def test_flashinfer_scale_k_pad_width_rejects_negative_k() -> None:
    flashinfer_scale_k_pad_width = _load_mxfp8_utils().flashinfer_scale_k_pad_width
    with pytest.raises(ValueError, match="non-negative"):
        flashinfer_scale_k_pad_width(-1)


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size", "expected"),
    [
        (2688, 1856, (3072, 1920)),
        (2688, 928, (3072, 1024)),
        (96, 128, (512, 128)),
        (4096, 2048, (4096, 2048)),
    ],
)
def test_flashinfer_mxfp8_moe_padding_plan(
    hidden_size: int,
    intermediate_size: int,
    expected: tuple[int, int],
) -> None:
    padding_plan = _load_mxfp8_utils().flashinfer_mxfp8_moe_padding_plan

    padded_hidden_size, padded_intermediate_size = padding_plan(
        hidden_size, intermediate_size
    )

    assert (padded_hidden_size, padded_intermediate_size) == expected
    assert padded_hidden_size % 512 == 0
    assert padded_hidden_size // 32 % 4 == 0
    assert padded_intermediate_size % 128 == 0
    assert padded_intermediate_size // 32 % 4 == 0


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size"),
    [(0, 128), (128, 0), (127, 128), (128, 127)],
)
def test_flashinfer_mxfp8_moe_padding_plan_rejects_invalid_sizes(
    hidden_size: int, intermediate_size: int
) -> None:
    padding_plan = _load_mxfp8_utils().flashinfer_mxfp8_moe_padding_plan

    with pytest.raises(ValueError):
        padding_plan(hidden_size, intermediate_size)


def test_pad_tensor_dim_preserves_values_and_fills_padding() -> None:
    pad_tensor_dim = _load_mxfp8_utils().pad_tensor_dim
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)

    padded = pad_tensor_dim(tensor, dim=1, padded_size=4, pad_value=127)

    assert tuple(padded.shape) == (2, 4)
    torch.testing.assert_close(padded[:, :2], tensor)
    assert torch.all(padded[:, 2:] == 127)


def test_pad_tensor_dim_rejects_smaller_target() -> None:
    pad_tensor_dim = _load_mxfp8_utils().pad_tensor_dim

    with pytest.raises(ValueError, match="Cannot pad MXFP8 tensor dim"):
        pad_tensor_dim(torch.ones(2, 3), dim=1, padded_size=2)


def test_assign_or_replace_parameter_copies_when_storage_is_compatible() -> None:
    assign_or_replace_parameter = _load_mxfp8_utils().assign_or_replace_parameter
    layer = torch.nn.Module()
    parameter = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    parameter.weight_loader = object()
    layer.register_parameter("weight", parameter)

    assign_or_replace_parameter(layer, "weight", torch.ones(2, 3))

    assert layer.weight is parameter
    assert layer.weight.weight_loader is parameter.weight_loader
    assert torch.all(layer.weight == 1)


def test_assign_or_replace_parameter_can_force_new_runtime_storage() -> None:
    assign_or_replace_parameter = _load_mxfp8_utils().assign_or_replace_parameter
    layer = torch.nn.Module()
    checkpoint_parameter = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    layer.register_parameter("weight", checkpoint_parameter)

    assign_or_replace_parameter(
        layer,
        "weight",
        torch.ones(2, 3),
        force_replace=True,
    )

    assert layer.weight is not checkpoint_parameter
    assert torch.all(checkpoint_parameter == 0)
    assert torch.all(layer.weight == 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["new", "resize", "force"])
def test_runtime_parameter_owns_shared_scratch_storage(device: str, mode: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    assign = _load_mxfp8_utils().assign_or_replace_parameter
    layers = [torch.nn.Module(), torch.nn.Module()]
    scratch = torch.ones(2, 4, 8, dtype=torch.uint8, device=device)
    for index, layer in enumerate(layers):
        if mode != "new":
            shape = scratch.shape if mode == "force" else (1,)
            layer.weight = torch.nn.Parameter(
                torch.zeros(shape, dtype=scratch.dtype, device=device),
                requires_grad=False,
            )
        scratch.fill_(index + 1)
        assign(layer, "weight", scratch, force_replace=mode == "force")
    pointers = [layer.weight.data_ptr() for layer in layers]
    assert len(set(pointers + [scratch.data_ptr()])) == 3
    assert torch.all(layers[0].weight == 1)
    assert torch.all(layers[1].weight == 2)
    for index, layer in enumerate(layers):
        scratch.fill_(index + 3)
        assign(layer, "weight", scratch)
        assert layer.weight.data_ptr() == pointers[index]
    scratch.zero_()
    assert torch.all(layers[0].weight == 3)
    assert torch.all(layers[1].weight == 4)


@pytest.mark.parametrize("is_gated", [False, True])
def test_pad_w13_intermediate_preserves_gate_halves(is_gated: bool) -> None:
    pad_w13_intermediate = _load_mxfp8_utils().pad_w13_intermediate
    rows = 4 if is_gated else 2
    tensor = torch.arange(rows * 3).reshape(1, rows, 3)

    padded = pad_w13_intermediate(
        tensor,
        padded_intermediate_size=4,
        is_gated=is_gated,
        pad_value=-1,
    )

    expected_rows = 8 if is_gated else 4
    assert tuple(padded.shape) == (1, expected_rows, 3)
    if is_gated:
        torch.testing.assert_close(padded[:, :2], tensor[:, :2])
        torch.testing.assert_close(padded[:, 4:6], tensor[:, 2:])
        assert torch.all(padded[:, 2:4] == -1)
        assert torch.all(padded[:, 6:] == -1)
    else:
        torch.testing.assert_close(padded[:, :2], tensor)
        assert torch.all(padded[:, 2:] == -1)
