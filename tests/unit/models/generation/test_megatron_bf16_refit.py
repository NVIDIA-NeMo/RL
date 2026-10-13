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

"""Canonical HF receive hooks for Megatron inference storage."""

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.megatron.megatron_worker import (
    MegatronGenerationRefitMixin,
    _MegatronRefitTask,
)

pytestmark = pytest.mark.mcore

GATE = "model.layers.0.mlp.gate_proj.weight"
UP = "model.layers.0.mlp.up_proj.weight"


@dataclass(frozen=True)
class _LocalSpec:
    name: str
    part: int

    def select(self, tensor):
        return tensor.chunk(2, dim=-2)[self.part]

    def selected_shape(self, shape):
        return torch.Size((*shape[:-2], shape[-2] // 2, shape[-1]))


def _make_map(destination, wire_dtype=torch.bfloat16):
    local_specs = (_LocalSpec(GATE, 0), _LocalSpec(UP, 1))
    conversion = SimpleNamespace(
        param_name="decoder.layers.0.mlp.linear_fc1.weight",
        hf_param_names=(GATE, UP),
        local_hf_param_specs=lambda: local_specs,
        combine_local_hf_weights=lambda values: torch.cat(
            [values[GATE], values[UP]], dim=-2
        ),
    )
    task = _MegatronRefitTask(conversion, destination, id(destination))
    info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {"name": GATE, "dtype": str(wire_dtype)},
                {"name": UP, "dtype": str(wire_dtype)},
            ]
        },
    }
    worker = object.__new__(MegatronGenerationRefitMixin)
    worker._generation_m2n_pending = {}
    return worker, worker._build_destination_hf_to_local_param_map(info, [task])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("wire_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_destination_receives_wire_dtype_and_casts_live_storage(dtype, wire_dtype):
    destination = torch.nn.Parameter(
        torch.full((8, 4), -1, dtype=dtype), requires_grad=False
    )
    _, specs = _make_map(destination, wire_dtype)
    old_storage = destination.data
    destination.data = torch.full_like(destination, -2)
    with torch.no_grad():
        for index, name in enumerate((GATE, UP)):
            spec = specs.get(name)
            ctx = spec.materialize()
            assert spec.wire_dtype == ctx.buf.dtype == wire_dtype
            assert ctx.buf.shape == (4, 4)
            # These values distinguish FP16/FP32 from a lossy BF16 round trip.
            expected = torch.full((4, 4), index + 1.003, dtype=wire_dtype)
            ctx.buf.copy_(expected)
            if spec.post is not None:
                spec.post(ctx)
            torch.testing.assert_close(
                destination.chunk(2)[index], expected.to(dtype), rtol=0, atol=0
            )
    assert torch.all(old_storage == -1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_matching_destination_dtype_retains_direct_live_view(dtype):
    destination = torch.nn.Parameter(
        torch.zeros((8, 4), dtype=dtype), requires_grad=False
    )
    _, specs = _make_map(destination, dtype)
    spec = specs.get(GATE)
    ctx = spec.materialize()
    assert spec.wire_dtype == ctx.buf.dtype == dtype
    assert ctx.buf.data_ptr() == destination.data_ptr()
    assert spec.post is None


def test_destination_rejects_quantized_wire_metadata():
    worker = object.__new__(MegatronGenerationRefitMixin)
    info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [{"name": GATE, "dtype": "torch.float8_e4m3fn"}]
        },
    }
    with pytest.raises(ValueError, match="Unsupported Megatron bulk refit wire dtype"):
        worker._build_destination_hf_to_local_param_map(info, [])


@pytest.mark.parametrize("storage", ["bf16", "te", "grouped"])
def test_packed_refit_preserves_destination_copy(
    storage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from megatron.core import fp8_utils

    destination = SimpleNamespace(shape=(8, 4), copy_=MagicMock())
    monkeypatch.setattr(fp8_utils, "is_float8tensor", lambda _: storage == "te")
    monkeypatch.setattr(
        fp8_utils,
        "is_grouped_tensor_with_quantized_storage",
        lambda _: storage == "grouped",
    )
    quantized_copy = MagicMock()
    monkeypatch.setattr(fp8_utils, "copy_tensor_to_quantized_param", quantized_copy)
    conversion = SimpleNamespace(
        param_name="linear_fc1.weight", hf_param_names=(GATE, UP)
    )
    task = _MegatronRefitTask(conversion, destination, id(destination))
    assert task.is_quantized == (storage != "bf16")
    gate = torch.full((4, 4), 3, dtype=torch.bfloat16)
    up = torch.full((4, 4), 67, dtype=torch.bfloat16)
    converted = torch.cat((gate, up))
    worker = object.__new__(MegatronGenerationRefitMixin)
    worker._generation_refit_tasks = [task]
    worker._generation_refit_task_index = 0
    worker._generation_refit_model_chunks = []
    worker._generation_refit_pending_weights = {}
    worker._generation_refit_pending_streams = {}
    worker._generation_refit_remaining_dependencies = Counter(task.dependencies)
    worker.megatron_bridge = SimpleNamespace(
        stream_weights_hf_to_megatron=MagicMock(
            return_value=iter([SimpleNamespace(weight=converted)])
        )
    )

    # This is the packed collective/null loader, also used by NCCL misc weights.
    worker._load_generation_refit_batch([(GATE, gate)])
    destination.copy_.assert_not_called()
    worker._load_generation_refit_batch([(UP, up)])
    destination.copy_.assert_called_once_with(converted)
    quantized_copy.assert_not_called()
    assert worker._generation_refit_task_index == 1
    assert worker._generation_refit_pending_weights == {}


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("wire_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_bulk_quantized_writer_waits_for_complete_fused_weight(
    grouped: bool, wire_dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    from megatron.core import fp8_utils

    destination = torch.zeros((2, 8, 4) if grouped else (8, 4), dtype=torch.bfloat16)
    monkeypatch.setattr(fp8_utils, "is_float8tensor", lambda _: not grouped)
    monkeypatch.setattr(
        fp8_utils, "is_grouped_tensor_with_quantized_storage", lambda _: grouped
    )
    quantized_copy = MagicMock()
    monkeypatch.setattr(fp8_utils, "copy_tensor_to_quantized_param", quantized_copy)
    worker, specs = _make_map(destination, wire_dtype)
    worker._write_generation_refit_weight = MagicMock()
    for cycle in (1, 2):
        expected = torch.empty_like(destination, dtype=wire_dtype)
        expected.chunk(2, dim=-2)[0].fill_(3 * cycle)
        expected.chunk(2, dim=-2)[1].fill_(67 * cycle)
        for index, name in enumerate((GATE, UP)):
            spec = specs.get(name)
            ctx = spec.materialize()
            assert spec.wire_dtype == ctx.buf.dtype == wire_dtype
            ctx.buf.copy_(expected.chunk(2, dim=-2)[index])
            spec.post(ctx)
            assert quantized_copy.call_count == cycle - 1 + index
        actual_destination, assembled = quantized_copy.call_args.args
        assert actual_destination is destination
        torch.testing.assert_close(assembled, expected)
        assert worker._generation_m2n_pending == {}
    worker._write_generation_refit_weight.assert_not_called()


@pytest.mark.parametrize("wire_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_grouped_expert_receiver_preserves_wire_dtype(wire_dtype):
    destinations = [torch.zeros((8, 4), dtype=wire_dtype) for _ in range(2)]
    tasks = []
    for expert, destination in enumerate(destinations):
        names = tuple(
            f"model.layers.0.mlp.experts.{expert}.{projection}.weight"
            for projection in ("gate_proj", "up_proj")
        )
        local_specs = tuple(_LocalSpec(name, part) for part, name in enumerate(names))
        conversion = SimpleNamespace(
            param_name=f"linear_fc1.weight{expert}",
            local_hf_param_specs=lambda specs=local_specs: specs,
        )
        tasks.append(_MegatronRefitTask(conversion, destination, id(destination)))
    metadata = [
        {
            "name": f"model.layers.0.mlp.experts.{projection}.weight",
            "dtype": str(wire_dtype),
            "grouped_expert_proj": projection,
        }
        for projection in ("gate_proj", "up_proj")
    ]
    info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {"model.layers.0": metadata},
    }
    worker = object.__new__(MegatronGenerationRefitMixin)
    specs = worker._build_destination_hf_to_local_param_map(info, tasks)
    for cycle in (1, 2):
        for part, param_info in enumerate(metadata):
            spec = specs.get(param_info["name"])
            ctx = spec.materialize()
            assert spec.wire_dtype == ctx.buf.dtype == wire_dtype
            assert ctx.buf.shape == (2, 4, 4)
            for expert in range(2):
                ctx.buf[expert].fill_(cycle + part + expert + 0.003)
            spec.post(ctx)
            for expert, destination in enumerate(destinations):
                torch.testing.assert_close(
                    destination.chunk(2)[part], ctx.buf[expert], rtol=0, atol=0
                )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TE quantization requires CUDA"
)
@pytest.mark.parametrize("recipe", ["blockwise", "mxfp8"])
@pytest.mark.parametrize("wire_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantized_fused_destination_commits_values_and_scales_on_current_stream(
    recipe,
    wire_dtype,
    monkeypatch,
):
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from megatron.core import fp8_utils
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
    )
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    if recipe == "mxfp8":
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)

    quantizer_cls = Float8BlockQuantizer if recipe == "blockwise" else MXFP8Quantizer
    quantizer = quantizer_cls(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    destination = quantizer.quantize(
        torch.zeros((256, 128), dtype=torch.bfloat16, device="cuda")
    )
    worker, specs = _make_map(destination, wire_dtype)
    storage_names = (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    )
    pointers = {name: getattr(destination, name).data_ptr() for name in storage_names}
    old_data = destination._rowwise_data.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    commit = fp8_utils.copy_tensor_to_quantized_param
    observed_streams = []

    def record_commit(target, received):
        assert target is destination
        observed_streams.append(torch.cuda.current_stream())
        commit(target, received)

    monkeypatch.setattr(fp8_utils, "copy_tensor_to_quantized_param", record_commit)
    for cycle in (1, 2):
        expected = torch.cat(
            (
                torch.full((128, 128), 3.0 * cycle, dtype=wire_dtype, device="cuda"),
                torch.full((128, 128), 67.0 * cycle, dtype=wire_dtype, device="cuda"),
            )
        )
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(stream):
            for index, name in enumerate((GATE, UP)):
                spec = specs.get(name)
                ctx = spec.materialize()
                assert spec.wire_dtype == ctx.buf.dtype == wire_dtype
                ctx.buf.copy_(expected.chunk(2)[index])
                spec.post(ctx)
                if index == 0:
                    assert torch.equal(destination._rowwise_data, old_data)
        stream.synchronize()
        reference = quantizer.quantize(expected)
        for name in storage_names:
            assert getattr(destination, name).data_ptr() == pointers[name]
            actual_storage = getattr(destination, name)
            reference_storage = getattr(reference, name)
            if recipe == "blockwise" and name.endswith("_scale_inv"):
                # TE pads the scale grid's final dimension to a multiple of four.
                # Those unused entries are uninitialized; compare every live scale.
                rows = expected.shape[0] // quantizer.block_len
                columns = expected.shape[1] // quantizer.block_len
                if name.startswith("_columnwise"):
                    rows, columns = columns, rows
                actual_storage = actual_storage[:rows, :columns]
                reference_storage = reference_storage[:rows, :columns]
            torch.testing.assert_close(
                actual_storage,
                reference_storage,
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        torch.testing.assert_close(
            destination.dequantize().float(), expected.float(), rtol=0.07, atol=0
        )
        old_data = destination._rowwise_data.clone()
        assert worker._generation_m2n_pending == {}
    assert observed_streams == [stream, stream]
