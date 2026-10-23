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

"""Unit tests for train-side expert source grouping and materialization.

``MegatronReshardParamSpecCollection._grouped_expert_spec`` creates a ``pre`` hook
that prepares this rank's full expert group before selecting one HF projection.
Plain CPU tensors suffice for unquantized sources.

Importing ``megatron_policy_worker`` pulls in megatron.core, so this is
mcore-marked and skipped where mcore is unavailable.
"""

import math
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

# megatron_policy_worker imports both megatron.core and megatron.bridge at
# module top, so guard on both: an env can have megatron.core but not
# megatron.bridge, and importing this test module would otherwise raise a
# collection error (not skip) in non-mcore lanes.
pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    LocalHFParamSpec,
)

import nemo_rl.models.policy.workers.megatron_policy_worker as worker_module  # noqa: E402
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
    MegatronReshardParamSpecCollection,
)
from nemo_rl.weight_sync.nccl_reshard_utils import LocalParamSpec, RefitCtx

pytestmark = pytest.mark.mcore


def _source(
    tensor: torch.Tensor, hf_spec: LocalHFParamSpec | None = None
) -> LocalParamSpec:
    return MegatronReshardParamSpecCollection._default_param_spec(
        tensor, hf_spec if hf_spec is not None else LocalHFParamSpec("weight")
    )


def _group(proj, grouped_name, expert_groups):
    grouped = MegatronReshardParamSpecCollection._grouped_expert_spec(
        proj, grouped_name, expert_groups
    )
    return grouped.materialize().buf


def test_grouped_expert_spec_missing_group_raises():
    groups = {("other.experts", "gate_proj"): [_source(torch.randn(8, 8))]}
    with pytest.raises(AssertionError, match="no local experts.*PP-filter"):
        _group("gate_proj", "model.layers.0.mlp.experts.gate_proj.weight", groups)


def test_grouped_expert_spec_empty_group_raises():
    prefix = "model.layers.0.mlp.experts"
    with pytest.raises(AssertionError, match="no local experts.*PP-filter"):
        _group("gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): []})


@pytest.mark.parametrize("mismatch", ["shape", "split"])
def test_grouped_expert_spec_rejects_incompatible_hf_views(
    mismatch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = "model.layers.0.mlp.experts"
    members = [
        _source(torch.ones(4, 4), LocalHFParamSpec("first", -2, 0, 2)),
        _source(
            torch.ones(8 if mismatch == "shape" else 4, 4),
            LocalHFParamSpec("second", -2, int(mismatch == "split"), 2),
        ),
    ]
    prepare = MagicMock(side_effect=AssertionError("incompatible group was prepared"))
    monkeypatch.setattr(worker_module, "_dequantize_refit_source_group", prepare)
    with pytest.raises(ValueError, match="incompatible HF views"):
        _group(
            "gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): members}
        )
    prepare.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("split_index", [0, 1])
def test_grouped_expert_spec_selects_equivalent_axes_after_group_preparation(
    dtype: torch.dtype, split_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = "model.layers.0.mlp.experts"
    proj = "gate_proj" if split_index == 0 else "up_proj"
    sources = tuple(
        torch.nn.Parameter(
            1 + torch.arange(24, dtype=dtype).reshape(6, 4) / 1024 + index,
            requires_grad=False,
        )
        for index in range(2)
    )
    members = [
        _source(source, LocalHFParamSpec(f"expert-{index}", dim, split_index, 2))
        for index, (source, dim) in enumerate(zip(sources, (0, -2)))
    ]
    for member in members:
        member.pre = MagicMock(side_effect=AssertionError("individual hook called"))
    prepare = MagicMock(wraps=worker_module._dequantize_refit_source_group)
    monkeypatch.setattr(worker_module, "_dequantize_refit_source_group", prepare)
    grouped = MegatronReshardParamSpecCollection._grouped_expert_spec(
        proj, f"{prefix}.{proj}.weight", {(prefix, proj): members}
    )
    assert grouped.wire_dtype == dtype
    for cycle in range(2):
        result = grouped.materialize().buf
        expected = torch.stack(sources).chunk(2, dim=1)[split_index].contiguous()
        assert result.dtype == dtype and result.is_contiguous()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        if dtype != torch.bfloat16:
            assert not torch.equal(result, result.to(torch.bfloat16).to(dtype))
        assert prepare.call_count == cycle + 1
        assert all(
            actual is original
            for actual, original in zip(prepare.call_args.args[0], sources)
        )
        for source in sources:
            source.data = source.data + 1
    for member in members:
        member.pre.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TE quantization requires CUDA"
)
@pytest.mark.parametrize(
    "recipe,block_dim,shape",
    [
        pytest.param("mxfp8", None, (96, 160), id="mxfp8-padded-scales"),
        pytest.param("mxfp8", None, (128, 128), id="mxfp8-aligned"),
        pytest.param("blockwise", 1, (256, 128), id="blockwise-1d"),
        pytest.param("blockwise", 2, (256, 128), id="blockwise-2d"),
        pytest.param("current", None, (256, 128), id="scalar-current-scaling"),
        pytest.param("delayed", None, (256, 128), id="scalar-delayed-scaling"),
    ],
)
def test_quantized_expert_group_decodes_once_with_current_scales(
    recipe: str,
    block_dim: int | None,
    shape: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The optional TE quantizers and kernels are needed only for this GPU test.
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
        Float8Quantizer,
    )
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
        Float8BlockwiseQTensorStorage,
    )
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
        MXFP8TensorStorage,
    )

    rows, columns = shape
    if recipe == "mxfp8":
        available, reason = te.is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(reason)

    if recipe == "mxfp8":
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        storage_type = MXFP8TensorStorage
    elif recipe == "blockwise":
        quantizer = Float8BlockQuantizer(
            tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_dim,
        )
        storage_type = Float8BlockwiseQTensorStorage
    if recipe in ("current", "delayed"):
        quantizers = tuple(
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=torch.device("cuda"),
                rowwise=True,
                columnwise=False,
            )
            if recipe == "current"
            else Float8Quantizer(
                scale=torch.tensor([0.5 / (index + 1)], device="cuda"),
                amax=torch.zeros(1, device="cuda"),
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
            )
            for index in range(2)
        )
        storage_type = None
    else:
        quantizers = (quantizer, quantizer)
    column_indices = torch.arange(columns, device="cuda")
    column_blocks = torch.tensor(
        [0.0625, 0.125, 0.25, 0.5, 1.0], device="cuda", dtype=torch.bfloat16
    )[(column_indices // 32) % 5]
    within_block = (1 + (column_indices % 8).float() / 8).to(torch.bfloat16)
    # Vary scale blocks along both axes and values within each row, so crossed
    # scale rows or columns cannot match the reference.
    column_pattern = column_blocks * within_block
    values = tuple(
        torch.cat(
            [
                torch.full(
                    (rows // 2, columns), value, device="cuda", dtype=torch.bfloat16
                )
                for value in pair
            ]
        )
        * column_pattern
        for pair in ((3.0, 67.0), (0.125, 129.0))
    )
    members = tuple(q.quantize(value) for q, value in zip(quantizers, values))
    original_decode = storage_type.dequantize if storage_type is not None else None
    previous = None
    for cycle in (1, 2):
        with torch.no_grad():
            for index, (member, value, q) in enumerate(
                zip(members, values, quantizers)
            ):
                if recipe == "delayed":
                    q.scale.fill_(0.5 / (index + 1) * cycle)
                q.update_quantized(value * cycle, member)
        expected = torch.stack(
            [member.dequantize(dtype=torch.bfloat16) for member in members]
        )
        decoded_groups = []

        def decode_once(tensor, *args, **kwargs):
            decoded_groups.append(tensor)
            # Only scale metadata may need padding; raw expert weights must
            # remain a direct concatenation of their original logical rows.
            assert tensor._rowwise_data.shape == (len(members) * rows, columns)
            assert torch.equal(
                tensor._rowwise_data,
                torch.cat([member._rowwise_data for member in members]),
            )
            if recipe == "mxfp8":
                assert torch.equal(
                    tensor._rowwise_scale_inv[: len(members) * rows, : columns // 32],
                    torch.cat(
                        [
                            member._rowwise_scale_inv[:rows, : columns // 32]
                            for member in members
                        ]
                    ),
                )
            assert original_decode is not None
            return original_decode(tensor, *args, **kwargs)

        member_decode = MagicMock(
            side_effect=AssertionError("per-expert decode called")
        )
        with monkeypatch.context() as hooks:
            hooks.setattr(type(members[0]), "dequantize", member_decode)
            if storage_type is not None:
                hooks.setattr(storage_type, "dequantize", decode_once)
            actual = worker_module._dequantize_refit_source_group(members)
        member_decode.assert_not_called()
        assert len(decoded_groups) == (1 if storage_type is not None else 0)
        assert actual.dtype == torch.bfloat16
        assert actual.shape == (2, rows, columns)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual
    if storage_type is not None:
        failed_decode = MagicMock(side_effect=RuntimeError("group decode failed"))
        with monkeypatch.context() as hooks:
            hooks.setattr(type(members[0]), "dequantize", member_decode)
            hooks.setattr(storage_type, "dequantize", failed_decode)
            with pytest.raises(RuntimeError, match="group decode failed"):
                worker_module._dequantize_refit_source_group(members)
        failed_decode.assert_called_once()
        member_decode.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TE quantization requires CUDA"
)
@pytest.mark.parametrize("block_dim", [1, 2])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((96, 128), id="partial-row-block"),
        pytest.param((128, 96), id="partial-column-block"),
        pytest.param((1856, 2688), id="nano-expert-shard"),
    ],
)
def test_unaligned_blockwise_experts_preserve_edge_blocks_and_current_scales(
    block_dim: int, shape: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The optional TE quantizer and kernels are needed only for this GPU test.
    import transformer_engine_torch as tex
    from megatron.core import fp8_utils
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
    )

    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=block_dim,
    )
    rows, columns = shape
    row_indices = torch.arange(rows, device="cuda").unsqueeze(1)
    column_indices = torch.arange(columns, device="cuda").unsqueeze(0)
    # Distinct expert and edge-block scales expose accidental block sharing
    # when the last partial block of one expert is packed with the next.
    pattern = (
        2.0 ** ((row_indices // 128) % 3 + (column_indices // 128) % 3)
        * (1 + (row_indices % 8 + column_indices % 8).float() / 16)
    ).to(torch.bfloat16)
    values = (pattern, pattern * 32)
    members = tuple(quantizer.quantize(value) for value in values)
    # Exercise the common logical-export entry point with real TE members;
    # only discovery of their enclosing GroupedTensor is stubbed.
    grouped_source = torch.empty(0)
    monkeypatch.setattr(
        fp8_utils,
        "is_grouped_tensor_with_quantized_storage",
        lambda tensor: tensor is grouped_source,
    )
    member_views = MagicMock(return_value=members)
    monkeypatch.setattr(fp8_utils, "get_grouped_quantized_members", member_views)
    previous = None
    for cycle in (1, 2):
        with torch.no_grad():
            for member, value in zip(members, values):
                quantizer.update_quantized(value * cycle, member)
        expected = torch.stack(
            [member.dequantize(dtype=torch.bfloat16) for member in members]
        )
        decode = MagicMock(wraps=fp8_utils.dequantize_fp8_tensor)
        with monkeypatch.context() as hooks:
            hooks.setattr(fp8_utils, "dequantize_fp8_tensor", decode)
            actual = worker_module._dequantize_refit_source_group(members)
            grouped_actual = worker_module._dequantize_refit_source(grouped_source)
        assert decode.call_count == 2 * len(members)
        assert all(
            invocation.args[0] is member
            for invocation, member in zip(decode.call_args_list, members * 2)
        )
        for result in (actual, grouped_actual):
            assert result.dtype == torch.bfloat16
            assert result.shape == (len(members), rows, columns)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual
        assert member_views.call_count == cycle
        member_views.assert_called_with(grouped_source, create_if_missing=True)


@pytest.mark.parametrize(
    "layout,reason",
    [
        ("columnwise", "rowwise primary-weight storage"),
        ("swizzled", "compact MXFP8 primary-weight scales"),
        ("unaligned-mxfp8", "aligned to 32-element quantization blocks"),
    ],
)
def test_quantized_expert_group_requires_primary_weight_layout(
    layout: str, reason: str
) -> None:
    # Real TE storage descriptors suffice; rejection runs before CUDA decoding.
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (
        MXFP8TensorStorage,
    )

    data = torch.zeros(
        (96, 144 if layout == "unaligned-mxfp8" else 160), dtype=torch.uint8
    )
    source = MXFP8TensorStorage(
        rowwise_data=None if layout == "columnwise" else data,
        rowwise_scale_inv=torch.ones((128, 8), dtype=torch.uint8),
        columnwise_data=data,
        columnwise_scale_inv=torch.ones((4, 256), dtype=torch.uint8),
        fp8_dtype=tex.DType.kFloat8E4M3,
        quantizer=None,
        with_gemm_swizzled_scales=layout == "swizzled",
    )
    with pytest.raises(ValueError, match=reason):
        worker_module._dequantize_refit_source_group((source, source))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("grouped", [False, True])
def test_source_map_rejects_metadata_dtype_mismatch(
    dtype: torch.dtype, grouped: bool
) -> None:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.my_pp_stage = 0
    prefix = "model.layers.0.mlp.experts"
    name = (
        f"{prefix}.gate_proj.weight"
        if grouped
        else "model.layers.0.mlp.gate_proj.weight"
    )
    source_name = f"{prefix}.0.gate_proj.weight" if grouped else name

    def pre(base: torch.Tensor) -> RefitCtx:
        pytest.fail("map validation must not materialize source buffers")

    source = _source(torch.empty(2, 2, dtype=dtype))
    source.pre = pre
    worker._iter_local_hf_param_shards = lambda: [(source_name, source)]
    metadata_dtype = torch.float32 if dtype != torch.float32 else torch.float16
    param = {"name": name, "dtype": str(metadata_dtype)}
    if grouped:
        param["grouped_expert_proj"] = "gate_proj"
    info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {"model.layers.0": [param]},
    }
    with pytest.raises(ValueError, match="declares wire dtype.*from metadata"):
        worker._build_source_hf_to_local_param_map(info)


# --------------------------------------------------------------------------
# build_hf_to_local_param_map (train/src side) — folds this rank's local
# shards (_iter_local_hf_param_shards) into LocalParamSpecs.  Fake the shard
# iterator; _build_expert_groups / _grouped_expert_spec run for real.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_build_hf_to_local_param_map_train_side(dtype: torch.dtype):
    from nemo_rl.weight_sync.nccl_reshard_utils import HFToLocalParamMap

    w = object.__new__(MegatronPolicyWorkerImpl)  # no __init__ / no megatron state
    w.my_pp_stage = 0
    # opd_full off, mirroring __init__ when the config block is absent.
    w._opd_full_enabled = False
    w._opd_full_lm_head_lifecycle = None
    w._opd_full_teacher_lm_heads = {}
    w._opd_full_teacher_checkpoint_paths = {}
    w._opd_full_lm_head_evicted = False
    prefix = "model.layers.0.mlp.experts"
    direct = torch.randn(8, 16, dtype=dtype)  # dense FFN down_proj local shard view
    e9 = torch.randn(128, 16, dtype=dtype)  # global experts on a later EP rank
    e10 = torch.randn(128, 16, dtype=dtype)
    e11 = torch.randn(128, 16, dtype=dtype)
    w._iter_local_hf_param_shards = lambda: [
        (
            "model.layers.0.mlp.down_proj.weight",
            _source(direct),
        ),
        (f"{prefix}.11.gate_proj.weight", _source(e11)),
        (f"{prefix}.9.gate_proj.weight", _source(e9)),
        (f"{prefix}.10.gate_proj.weight", _source(e10)),
    ]
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [8, 16],
                    "dtype": str(dtype),
                },
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "global_shape": [12, 128, 16],
                    "dtype": str(dtype),
                    "grouped_expert_proj": "gate_proj",
                },
            ]
        },
    }

    pmap = w.build_hf_to_local_param_map(refit_info)
    assert isinstance(pmap, HFToLocalParamMap)

    # Direct: the typed descriptor retains the live local parameter.
    d = pmap.get("model.layers.0.mlp.down_proj.weight")
    assert d.base.tensor is direct and d.pre is not None and d.post is None
    assert d.wire_dtype == dtype

    # Global expert indices sort numerically, independently of insertion order.
    g = pmap.get(f"{prefix}.gate_proj.weight")
    assert tuple(source.tensor for source in g.base) == (e9, e10, e11)
    assert g.pre is not None and g.post is None
    assert g.wire_dtype == dtype
    ctx = g.materialize()
    assert ctx.buf.shape == (3, 128, 16)
    assert ctx.buf.dtype == dtype
    assert torch.equal(ctx.buf, torch.stack((e9, e10, e11)))

    direct.add_(1)
    e9.add_(2)
    e10.add_(3)
    e11.add_(4)
    assert d.materialize().buf is direct
    refreshed = g.materialize().buf
    assert refreshed.dtype == dtype
    assert torch.equal(refreshed, torch.stack((e9, e10, e11)))
    assert not torch.equal(refreshed, ctx.buf)


class _FakeMXFP8Tensor:
    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        import transformer_engine_torch

        self.shape = data.shape
        self._metadata = {
            "rowwise_data": data,
            "rowwise_scale_inv": scale,
            "with_gemm_swizzled_scales": False,
            "fp8_dtype": transformer_engine_torch.DType.kFloat8E4M3,
        }

    def get_metadata(self) -> dict[str, object]:
        return self._metadata


def _native_tensor(
    shape: tuple[int, ...],
    *,
    value_marker: int,
    scale_marker: int,
) -> _FakeMXFP8Tensor:
    rows = math.prod(shape[:-1])
    return _FakeMXFP8Tensor(
        torch.full(shape, value_marker, dtype=torch.uint8),
        torch.full(
            (rows, shape[-1] // 32),
            scale_marker,
            dtype=torch.uint8,
        ),
    )


def _native_worker(
    tasks: list[SimpleNamespace],
    *,
    grouped_tasks: list[SimpleNamespace] | None = None,
) -> MegatronPolicyWorkerImpl:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.fp8_cfg = {"enabled": True, "fp8_param": True, "fp8_recipe": "mxfp8"}
    worker.megatron_cfg = SimpleNamespace(
        optimizer=SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=False),
        ddp=SimpleNamespace(overlap_param_gather=False),
    )
    worker.cfg = cast(
        Any,
        {
            "megatron_cfg": {"native_mxfp8_refit": True},
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"precision": "fp8", "is_mx": True},
            },
        },
    )
    worker.refit_conversion_tasks = tasks
    worker._native_grouped_mxfp8_tasks = grouped_tasks or []
    return worker


def _native_components(
    shape: tuple[int, ...],
) -> list[dict[str, object]]:
    return [
        {
            "role": "weight",
            "global_shape": shape,
            "dtype": "torch.float8_e4m3fn",
        },
        {
            "role": "weight_scale",
            "global_shape": (*shape[:-1], shape[-1] // 32),
            "dtype": "torch.uint8",
        },
    ]


def _refit_info(
    params: list[tuple[str, tuple[int, ...], str | None]],
) -> dict[str, Any]:
    return {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": name,
                    "components": _native_components(shape),
                    **(
                        {"grouped_expert_proj": grouped_projection}
                        if grouped_projection is not None
                        else {}
                    ),
                }
                for name, shape, grouped_projection in params
            ]
        },
    }


def test_native_mxfp8_dense_fc1_split_and_fc2_direct_refresh() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    gate_name = "model.layers.0.mlp.gate_proj.weight"
    up_name = "model.layers.0.mlp.up_proj.weight"
    down_name = "model.layers.0.mlp.down_proj.weight"
    fc1 = _native_tensor((16, 64), value_marker=11, scale_marker=12)
    fc2 = _native_tensor((64, 32), value_marker=21, scale_marker=22)
    tasks = [
        SimpleNamespace(
            mapping=GatedMLPMapping(
                "decoder.layers.0.mlp.linear_fc1.weight",
                gate=gate_name,
                up=up_name,
            ),
            param_weight=fc1,
            global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
        ),
        SimpleNamespace(
            mapping=AutoMapping(
                "decoder.layers.0.mlp.linear_fc2.weight",
                down_name,
            ),
            param_weight=fc2,
            global_param_name="decoder.layers.0.mlp.linear_fc2.weight",
        ),
    ]
    worker = _native_worker(tasks)

    source_map = worker.build_hf_to_local_param_map(
        _refit_info(
            [
                (gate_name, (8, 64), None),
                (up_name, (8, 64), None),
                (down_name, (64, 32), None),
            ]
        )
    )

    expected_shapes = {
        gate_name: {"weight": (8, 64), "weight_scale": (8, 2)},
        up_name: {"weight": (8, 64), "weight_scale": (8, 2)},
        down_name: {"weight": (64, 32), "weight_scale": (64, 1)},
    }
    for name, roles in expected_shapes.items():
        for role, shape in roles.items():
            spec = source_map.get(name, role=role)
            assert spec is not None
            assert spec.base.shape == shape
            assert spec.pre is None

    down_weight = source_map.get(down_name, role="weight")
    assert down_weight is not None
    first = down_weight.base
    replacement = torch.full((64, 32), 91, dtype=torch.uint8)
    fc2._metadata["rowwise_data"] = replacement
    worker._refresh_local_native_mxfp8_param_components()
    second = down_weight.base
    assert first.data_ptr() != second.data_ptr()
    assert second.view(torch.uint8).data_ptr() == replacement.data_ptr()
    assert torch.equal(second.view(torch.uint8), replacement)


def test_native_mxfp8_task_builder_uses_public_bridge_api() -> None:
    fc1_name = "decoder.layers.0.mlp.experts.linear_fc1.weight"
    fc2_name = "decoder.layers.0.mlp.experts.linear_fc2.weight"
    tasks = [
        SimpleNamespace(
            global_param_name="decoder.layers.0.self_attention.linear_qkv.weight"
        ),
        SimpleNamespace(global_param_name=fc1_name),
        SimpleNamespace(global_param_name=f"{fc1_name}0"),
        SimpleNamespace(global_param_name=fc2_name),
    ]
    model = object()
    calls: list[list[object]] = []

    class FakeBridge:
        def get_export_mxfp8_tasks(self, models: list[object]) -> list[SimpleNamespace]:
            calls.append(models)
            return tasks

    worker = _native_worker([])
    worker.model = model
    worker.megatron_bridge = FakeBridge()

    result = worker._build_native_mxfp8_conversion_tasks()

    assert calls == [[model]]
    assert result is tasks
    assert worker._native_grouped_mxfp8_tasks == [tasks[1], tasks[3]]


@pytest.mark.parametrize(
    "global_name",
    [
        "decoder.layers.0.mlp.experts.local_experts.3.linear_fc1.weight",
        "decoder.layers.0.mlp.experts.linear_fc1.weight3",
    ],
)
def test_native_mxfp8_simple_expert_fc1_up_projection_is_direct(
    global_name: str,
) -> None:
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    expert_name = "model.layers.0.mlp.experts.3.up_proj.weight"
    source = _native_tensor((8, 64), value_marker=31, scale_marker=32)
    worker = _native_worker(
        [
            SimpleNamespace(
                mapping=AutoMapping(
                    global_name,
                    expert_name,
                ),
                param_weight=source,
                global_param_name=global_name,
            )
        ]
    )

    components = list(worker._iter_local_native_mxfp8_param_components())

    assert [(name, role) for name, role, _ in components] == [
        (expert_name, "weight"),
        (expert_name, "weight_scale"),
    ]
    assert [tuple(tensor.shape) for _, _, tensor in components] == [
        (8, 64),
        (8, 2),
    ]


@pytest.mark.parametrize("with_weight_suffix", [False, True])
def test_native_mxfp8_fused_expert_names_normalize_optional_weight_suffix(
    with_weight_suffix: bool,
) -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
    )

    prefix = "model.layers.0.mlp.experts"
    suffix = ".weight" if with_weight_suffix else ""
    worker = _native_worker([])
    fc1 = SimpleNamespace(
        mapping=FusedGatedExpertMapping(
            "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            f"{prefix}.gate_up_proj{suffix}",
        ),
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
    )
    fc2 = SimpleNamespace(
        mapping=FusedExpertMapping(
            "decoder.layers.0.mlp.experts.linear_fc2.weight0",
            f"{prefix}.down_proj{suffix}",
        ),
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
    )

    assert worker._native_task_projections(fc1, grouped=True) == (
        (f"{prefix}.gate_proj.weight", "gate"),
        (f"{prefix}.up_proj.weight", "up"),
    )
    assert worker._native_task_projections(fc2, grouped=True) == (
        (f"{prefix}.down_proj.weight", "down"),
    )


def test_native_mxfp8_component_iterator_extracts_once_per_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.bridge.models.conversion.param_mapping import GatedMLPMapping

    import nemo_rl.models.policy.workers.megatron_policy_worker as worker_module

    source = _native_tensor((16, 64), value_marker=1, scale_marker=2)
    task = SimpleNamespace(
        mapping=GatedMLPMapping(
            "decoder.layers.0.mlp.linear_fc1.weight",
            gate="model.layers.0.mlp.gate_proj.weight",
            up="model.layers.0.mlp.up_proj.weight",
        ),
        param_weight=source,
        global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
    )
    worker = _native_worker([task])
    real_extract = worker_module.extract_native_mxfp8_components
    extracted = []

    def record_extract(param: object):
        extracted.append(param)
        return real_extract(param)

    monkeypatch.setattr(
        worker_module, "extract_native_mxfp8_components", record_extract
    )

    assert len(list(worker._iter_local_native_mxfp8_param_components())) == 4
    assert extracted == [source]


def test_native_mxfp8_source_map_uses_shared_component_iterator() -> None:
    name = "model.layers.0.mlp.down_proj.weight"
    weight = torch.empty((64, 32), dtype=torch.float8_e4m3fn)
    scale = torch.empty((64, 1), dtype=torch.uint8)
    worker = _native_worker([])
    worker._iter_local_native_mxfp8_param_components = lambda: iter(
        [(name, "weight", weight), (name, "weight_scale", scale)]
    )

    source_map = worker.build_hf_to_local_param_map(
        _refit_info([(name, (64, 32), None)])
    )

    weight_spec = source_map.get(name, role="weight")
    scale_spec = source_map.get(name, role="weight_scale")
    assert weight_spec is not None and weight_spec.base is weight
    assert scale_spec is not None and scale_spec.base is scale


def test_native_mxfp8_per_expert_fc1_fc2_group_both_roles_numerically() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    prefix = "model.layers.0.mlp.experts"
    tasks = []
    for expert, marker in ((10, 100), (2, 20)):
        gate_name = f"{prefix}.{expert}.gate_proj.weight"
        up_name = f"{prefix}.{expert}.up_proj.weight"
        down_name = f"{prefix}.{expert}.down_proj.weight"
        fc1_data = torch.empty((8, 64), dtype=torch.uint8)
        fc1_data[:4].fill_(marker + 1)
        fc1_data[4:].fill_(marker + 2)
        fc1_scale = torch.empty((8, 2), dtype=torch.uint8)
        fc1_scale[:4].fill_(marker + 3)
        fc1_scale[4:].fill_(marker + 4)
        tasks.extend(
            [
                SimpleNamespace(
                    mapping=GatedMLPMapping(
                        f"decoder.layers.0.mlp.experts.local_experts.{expert}.linear_fc1.weight",
                        gate=gate_name,
                        up=up_name,
                    ),
                    param_weight=_FakeMXFP8Tensor(fc1_data, fc1_scale),
                    global_param_name=f"decoder.layers.0.mlp.experts.local_experts.{expert}.linear_fc1.weight",
                ),
                SimpleNamespace(
                    mapping=AutoMapping(
                        f"decoder.layers.0.mlp.experts.local_experts.{expert}.linear_fc2.weight",
                        down_name,
                    ),
                    param_weight=_native_tensor(
                        (64, 32),
                        value_marker=marker + 5,
                        scale_marker=marker + 6,
                    ),
                    global_param_name=f"decoder.layers.0.mlp.experts.local_experts.{expert}.linear_fc2.weight",
                ),
            ]
        )
    worker = _native_worker(tasks)
    params: list[tuple[str, tuple[int, ...], str | None]] = [
        (f"{prefix}.{projection}.weight", shape, projection)
        for projection, shape in (
            ("gate_proj", (2, 4, 64)),
            ("up_proj", (2, 4, 64)),
            ("down_proj", (2, 64, 32)),
        )
    ]

    source_map = worker.build_hf_to_local_param_map(_refit_info(params))

    expected = {
        "gate_proj": {"weight": (21, 101), "weight_scale": (23, 103)},
        "up_proj": {"weight": (22, 102), "weight_scale": (24, 104)},
        "down_proj": {"weight": (25, 105), "weight_scale": (26, 106)},
    }
    for projection, roles in expected.items():
        name = f"{prefix}.{projection}.weight"
        for role, markers in roles.items():
            spec = source_map.get(name, role=role)
            assert spec is not None and spec.pre is not None
            grouped = spec.pre(spec.base).buf
            storage = grouped.view(torch.uint8) if role == "weight" else grouped
            assert tuple(int(storage[index].flatten()[0]) for index in range(2)) == (
                markers
            )


def test_native_mxfp8_grouped_members_refresh_without_aggregate_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
    )
    from megatron.core import fp8_utils

    import nemo_rl.models.policy.workers.megatron_policy_worker as worker_module

    prefix = "model.layers.0.mlp.experts"
    fc1_grouped = object()
    fc2_grouped = object()
    fc1_members = [
        _native_tensor((8, 64), value_marker=11, scale_marker=12),
        _native_tensor((8, 64), value_marker=41, scale_marker=42),
    ]
    fc2_members = [
        _native_tensor((64, 32), value_marker=15, scale_marker=16),
        _native_tensor((64, 32), value_marker=45, scale_marker=46),
    ]
    member_calls = []

    def get_members(param: object, *, create_if_missing: bool):
        member_calls.append((param, create_if_missing))
        return fc1_members if param is fc1_grouped else fc2_members

    monkeypatch.setattr(fp8_utils, "get_grouped_quantized_members", get_members)
    extracted = []
    real_extract = worker_module.extract_native_mxfp8_components

    def record_extract(source: object):
        extracted.append(source)
        assert source not in (fc1_grouped, fc2_grouped)
        return real_extract(source)

    monkeypatch.setattr(
        worker_module,
        "extract_native_mxfp8_components",
        record_extract,
    )
    grouped_tasks = [
        SimpleNamespace(
            mapping=FusedGatedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc1.weight0",
                f"{prefix}.gate_up_proj",
            ),
            param_weight=fc1_grouped,
            global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
        ),
        SimpleNamespace(
            mapping=FusedExpertMapping(
                "decoder.layers.0.mlp.experts.linear_fc2.weight0",
                f"{prefix}.down_proj",
            ),
            param_weight=fc2_grouped,
            global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
        ),
    ]
    worker = _native_worker([], grouped_tasks=grouped_tasks)
    params: list[tuple[str, tuple[int, ...], str | None]] = [
        (f"{prefix}.{projection}.weight", shape, projection)
        for projection, shape in (
            ("gate_proj", (2, 4, 64)),
            ("up_proj", (2, 4, 64)),
            ("down_proj", (2, 64, 32)),
        )
    ]

    source_map = worker.build_hf_to_local_param_map(_refit_info(params))

    assert member_calls == []
    expected_shapes = {
        "gate_proj": {"weight": (2, 4, 64), "weight_scale": (2, 4, 2)},
        "up_proj": {"weight": (2, 4, 64), "weight_scale": (2, 4, 2)},
        "down_proj": {"weight": (2, 64, 32), "weight_scale": (2, 64, 1)},
    }
    for projection, roles in expected_shapes.items():
        name = f"{prefix}.{projection}.weight"
        for role, shape in roles.items():
            spec = source_map.get(name, role=role)
            assert spec is not None and spec.pre is not None
            assert spec.pre(spec.base).buf.shape == shape

    gate_spec = source_map.get(f"{prefix}.gate_proj.weight", role="weight")
    assert gate_spec is not None and gate_spec.pre is not None
    gate_pre = gate_spec.pre
    first = gate_pre(gate_spec.base).buf
    replacement = torch.full((8, 64), 99, dtype=torch.uint8)
    fc1_members[0]._metadata["rowwise_data"] = replacement
    second = gate_pre(gate_spec.base).buf
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(second[0].view(torch.uint8), replacement[:4])
    assert all(create_if_missing is False for _, create_if_missing in member_calls)
    assert extracted


def test_native_mxfp8_grouped_partition_initializes_missing_member_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.bridge.models.conversion.param_mapping import FusedExpertMapping
    from megatron.core import fp8_utils

    grouped_param = object()
    member = _native_tensor((64, 32), value_marker=1, scale_marker=2)
    task = SimpleNamespace(
        mapping=FusedExpertMapping(
            "decoder.layers.0.mlp.experts.linear_fc2.weight0",
            "model.layers.0.mlp.experts.down_proj",
        ),
        param_weight=grouped_param,
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
    )
    worker = _native_worker([], grouped_tasks=[task])
    calls: list[bool] = []

    def get_members(param: object, *, create_if_missing: bool):
        assert param is grouped_param
        calls.append(create_if_missing)
        if not create_if_missing:
            raise RuntimeError("member cache is not initialized")
        return [member]

    monkeypatch.setattr(fp8_utils, "get_grouped_quantized_members", get_members)
    monkeypatch.setattr(
        fp8_utils, "is_grouped_mxfp8tensor", lambda param: param is grouped_param
    )

    native, grouped, misc = worker._partition_native_mxfp8_conversion_tasks([task])

    assert native == []
    assert grouped == [task]
    assert misc == []
    assert calls == [False, True]


def test_native_mxfp8_grouped_validation_fails_before_any_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.bridge.models.conversion.param_mapping import FusedExpertMapping
    from megatron.core import fp8_utils

    import nemo_rl.weight_sync.xferdtensor as xfer_module
    from nemo_rl.weight_sync.nccl_reshard_utils import (
        HFToLocalParamMap,
        LocalParamSpec,
    )

    direct_name = "model.layers.0.mlp.down_proj.weight"
    grouped_name = "model.layers.0.mlp.experts.down_proj.weight"
    grouped_param = object()
    invalid_member = _native_tensor((64, 32), value_marker=1, scale_marker=2)
    invalid_member._metadata["rowwise_scale_inv"] = torch.ones(
        (64, 1), dtype=torch.float32
    )
    grouped_task = SimpleNamespace(
        mapping=FusedExpertMapping(
            "decoder.layers.0.mlp.experts.linear_fc2.weight0",
            "model.layers.0.mlp.experts.down_proj",
        ),
        param_weight=grouped_param,
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight",
    )
    worker = _native_worker([], grouped_tasks=[grouped_task])
    worker.my_pp_stage = 0
    worker.pp_comm_group = cast(Any, object())
    worker._broadcast_misc_params_packed = cast(Any, lambda **_: None)

    def grouped_pre(_base: object, *, role: str):
        from nemo_rl.weight_sync.nccl_reshard_utils import RefitCtx

        return RefitCtx(
            buf=worker._materialize_native_grouped_component(grouped_task, "down", role)
        )

    worker.hf_to_local_param_map = HFToLocalParamMap(
        specs={
            (direct_name, "weight"): LocalParamSpec(base=torch.empty(64, 32)),
            (grouped_name, "weight"): LocalParamSpec(
                base=None,
                pre=lambda base: grouped_pre(base, role="weight"),
            ),
            (grouped_name, "weight_scale"): LocalParamSpec(
                base=None,
                pre=lambda base: grouped_pre(base, role="weight_scale"),
            ),
        }
    )
    worker.nccl_reshard_refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": direct_name,
                    "pp_stage": 0,
                    "src_mesh_info": "src",
                    "dst_mesh_info": "dst",
                    "components": [
                        {
                            "role": "weight",
                            "global_shape": [64, 32],
                            "src_placements": [],
                            "dst_placements": [],
                        }
                    ],
                },
                {
                    "name": grouped_name,
                    "pp_stage": 0,
                    "src_mesh_info": "src",
                    "dst_mesh_info": "dst",
                    "components": _native_components((2, 64, 32)),
                },
            ]
        },
    }
    transfers = []

    monkeypatch.setattr(
        fp8_utils,
        "get_grouped_quantized_members",
        lambda param, *, create_if_missing: (
            [invalid_member]
            if param is grouped_param and create_if_missing is False
            else []
        ),
    )
    monkeypatch.setattr(
        xfer_module,
        "xferdtensor",
        lambda *args: transfers.append(args),
    )
    monkeypatch.setattr(
        xfer_module,
        "DTensorRef",
        lambda *, local_tensor, global_shape: SimpleNamespace(
            local_tensor=local_tensor,
            global_shape=global_shape,
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    with pytest.raises(ValueError, match=f"{grouped_name!r}.*role"):
        worker._nccl_reshard_refit()

    assert transfers == []


def test_native_mxfp8_skips_pp_placeholders_and_misc_mappings() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    worker = _native_worker(
        [
            SimpleNamespace(
                mapping=GatedMLPMapping(
                    "decoder.layers.0.mlp.linear_fc1.weight",
                    gate="model.layers.0.mlp.gate_proj.weight",
                    up="model.layers.0.mlp.up_proj.weight",
                ),
                param_weight=None,
                global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
            ),
            SimpleNamespace(
                mapping=AutoMapping(
                    "decoder.layers.0.self_attention.linear_qkv.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                ),
                param_weight=_native_tensor((64, 64), value_marker=1, scale_marker=2),
                global_param_name="decoder.layers.0.self_attention.linear_qkv.weight",
            ),
        ]
    )

    assert list(worker._iter_local_native_mxfp8_param_components()) == []


def test_native_mxfp8_rejects_unsupported_bulk_mapping() -> None:
    mapping = SimpleNamespace(
        hf_param="model.layers.0.mlp.down_proj.weight",
        is_expert=False,
    )
    worker = _native_worker(
        [
            SimpleNamespace(
                mapping=mapping,
                param_weight=_native_tensor((64, 32), value_marker=1, scale_marker=2),
                global_param_name="decoder.layers.0.mlp.linear_fc2.weight",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"model\.layers\.0\.mlp\.down_proj\.weight.*weight",
    ):
        list(worker._iter_local_native_mxfp8_param_components())


def test_native_mxfp8_metadata_has_ordered_component_shapes() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    gate_name = "model.layers.0.mlp.gate_proj.weight"
    up_name = "model.layers.0.mlp.up_proj.weight"
    down_name = "model.layers.0.mlp.down_proj.weight"
    worker = _native_worker(
        [
            SimpleNamespace(
                mapping=GatedMLPMapping(
                    "decoder.layers.0.mlp.linear_fc1.weight",
                    gate=gate_name,
                    up=up_name,
                ),
                param_weight=_native_tensor((16, 64), value_marker=1, scale_marker=2),
                global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
            ),
            SimpleNamespace(
                mapping=AutoMapping(
                    "decoder.layers.0.mlp.linear_fc2.weight",
                    down_name,
                ),
                param_weight=_native_tensor((64, 32), value_marker=3, scale_marker=4),
                global_param_name="decoder.layers.0.mlp.linear_fc2.weight",
            ),
        ]
    )

    metadata = worker._build_native_mxfp8_shape_metadata(
        {"tp_size": 2, "ep_size": 1, "pp_size": 1}
    )

    assert list(metadata) == [gate_name, up_name, down_name]
    assert metadata[gate_name]["shape"] == [16, 64]
    assert metadata[down_name]["shape"] == [64, 64]
    for name in (gate_name, up_name, down_name):
        components = metadata[name]["components"]
        assert [component["role"] for component in components] == [
            "weight",
            "weight_scale",
        ]
        assert components[0]["shape"] == metadata[name]["shape"]
        assert components[1]["shape"] == [
            *metadata[name]["shape"][:-1],
            metadata[name]["shape"][-1] // 32,
        ]


def test_native_mxfp8_metadata_keeps_bf16_ignored_experts_in_misc() -> None:
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    native_prefix = "model.layers.0.mlp.experts.0"
    ignored_prefix = "model.layers.1.mlp.experts.0"
    native_fc1 = WeightConversionTask(
        param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        global_param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
        mapping=GatedMLPMapping(
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
            gate=f"{native_prefix}.gate_proj.weight",
            up=f"{native_prefix}.up_proj.weight",
        ),
        param_weight=_native_tensor((8, 64), value_marker=1, scale_marker=2),
    )
    native_fc2 = WeightConversionTask(
        param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
        global_param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
        mapping=AutoMapping(
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
            f"{native_prefix}.down_proj.weight",
        ),
        param_weight=_native_tensor((64, 32), value_marker=3, scale_marker=4),
    )
    ignored_fc1 = WeightConversionTask(
        param_name="decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight",
        global_param_name="decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight",
        mapping=GatedMLPMapping(
            "decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight",
            gate=f"{ignored_prefix}.gate_proj.weight",
            up=f"{ignored_prefix}.up_proj.weight",
        ),
        param_weight=torch.zeros((8, 64), dtype=torch.bfloat16),
    )
    ignored_fc2 = WeightConversionTask(
        param_name="decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.weight",
        global_param_name="decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.weight",
        mapping=AutoMapping(
            "decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.weight",
            f"{ignored_prefix}.down_proj.weight",
        ),
        param_weight=torch.zeros((64, 32), dtype=torch.bfloat16),
    )
    tasks = [native_fc1, native_fc2, ignored_fc1, ignored_fc2]
    worker = _native_worker(tasks)
    worker._calculate_refit_param_info = lambda: []
    worker.draft_model = None
    worker.model = SimpleNamespace(config=SimpleNamespace(num_layers=2))

    def export_hf_weights(_models: Any, **kwargs: Any):
        exported_tasks = kwargs["conversion_tasks"]
        for task in exported_tasks:
            hf_param = task.mapping.hf_param
            names = hf_param.values() if isinstance(hf_param, dict) else (hf_param,)
            for name in names:
                yield str(name), torch.zeros((1,), dtype=torch.bfloat16)

    worker.megatron_bridge = SimpleNamespace(export_hf_weights=export_hf_weights)

    refit_info = worker.prepare_nccl_reshard_refit_info(
        {"tp_size": 1, "ep_size": 1, "pp_size": 1},
        {"tp_size": 1, "ep_size": 1, "pp_size": 1},
        1,
        1,
    )

    native_names = [
        param["name"]
        for params in refit_info["per_layer_params"].values()
        for param in params
    ]
    assert native_names == [
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
    ]
    assert list(refit_info["misc_meta"]) == [
        f"{ignored_prefix}.gate_proj.weight",
        f"{ignored_prefix}.up_proj.weight",
        f"{ignored_prefix}.down_proj.weight",
    ]
    assert worker._misc_conversion_tasks == [ignored_fc1, ignored_fc2]


def test_native_grouped_bf16_experts_route_to_misc_instead_of_raising() -> None:
    """A first/last-BF16 boundary layer must not fail the plan.

    Bridge builds the grouped task list now, and it names a BF16 expert weight
    exactly the way it names a quantized one, so the suffix classifier feeding
    ``_native_grouped_mxfp8_tasks`` cannot tell them apart. Megatron's
    ``get_grouped_quantized_members`` raises on a weight with no quantized
    storage -- asserted below so the premise is visible -- so a partition that
    reached for members before checking storage would turn every mixed-precision
    run into a planning-time failure on its first BF16 MoE layer.
    """
    from megatron.bridge.models.conversion.param_mapping import FusedGatedExpertMapping
    from megatron.core import fp8_utils

    bf16_weight = torch.zeros((2, 8, 64), dtype=torch.bfloat16)
    task = SimpleNamespace(
        mapping=FusedGatedExpertMapping(
            "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            "model.layers.0.mlp.experts.gate_up_proj.weight",
        ),
        param_weight=bf16_weight,
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight",
    )
    worker = _native_worker([], grouped_tasks=[task])

    with pytest.raises(ValueError):
        fp8_utils.get_grouped_quantized_members(bf16_weight, create_if_missing=False)

    native, grouped, misc = worker._partition_native_mxfp8_conversion_tasks([task])

    assert native == []
    assert grouped == []
    assert misc == [task]


def test_native_conversion_builder_routes_expanded_bf16_experts_to_misc() -> None:
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
    from megatron.bridge.models.conversion.param_mapping import (
        FusedGatedExpertMapping,
    )

    global_name = "decoder.layers.0.mlp.experts.linear_fc1.weight"
    members = [
        torch.zeros((8, 64), dtype=torch.bfloat16),
        torch.ones((8, 64), dtype=torch.bfloat16),
    ]
    tasks = [
        WeightConversionTask(
            param_name=f"{global_name}{expert}",
            global_param_name=f"{global_name}{expert}",
            mapping=FusedGatedExpertMapping(
                f"{global_name}{expert}",
                "model.layers.0.mlp.experts.gate_up_proj.weight",
            ),
            param_weight=member,
        )
        for expert, member in enumerate(members)
    ]
    calls: list[list[object]] = []

    class FakeBridge:
        def get_export_mxfp8_tasks(
            self, models: list[object]
        ) -> list[WeightConversionTask]:
            calls.append(models)
            return tasks

    worker = _native_worker([])
    model = object()
    worker.model = model
    worker.megatron_bridge = FakeBridge()

    result = worker._build_native_mxfp8_conversion_tasks()
    native, grouped, misc = worker._partition_native_mxfp8_conversion_tasks(result)

    assert calls == [[model]]
    assert result is tasks
    assert worker._native_grouped_mxfp8_tasks == []
    assert native == []
    assert grouped == []
    assert misc == tasks


def test_native_mxfp8_per_expert_metadata_expands_global_expert_axis() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    from nemo_rl.weight_sync.nccl_reshard_utils import (
        group_expert_params_in_metadata,
    )

    prefix = "model.layers.0.mlp.experts"
    tasks = []
    for local_expert in range(2):
        tasks.extend(
            [
                SimpleNamespace(
                    mapping=GatedMLPMapping(
                        f"decoder.layers.0.mlp.experts.local_experts.{local_expert}.linear_fc1.weight",
                        gate=f"{prefix}.{local_expert}.gate_proj.weight",
                        up=f"{prefix}.{local_expert}.up_proj.weight",
                    ),
                    param_weight=_native_tensor(
                        (8, 64), value_marker=1, scale_marker=2
                    ),
                    global_param_name=f"decoder.layers.0.mlp.experts.local_experts.{local_expert}.linear_fc1.weight",
                ),
                SimpleNamespace(
                    mapping=AutoMapping(
                        f"decoder.layers.0.mlp.experts.local_experts.{local_expert}.linear_fc2.weight",
                        f"{prefix}.{local_expert}.down_proj.weight",
                    ),
                    param_weight=_native_tensor(
                        (64, 32), value_marker=3, scale_marker=4
                    ),
                    global_param_name=f"decoder.layers.0.mlp.experts.local_experts.{local_expert}.linear_fc2.weight",
                ),
            ]
        )
    worker = _native_worker(tasks)

    metadata = worker._build_native_mxfp8_shape_metadata(
        {"tp_size": 4, "ep_size": 2, "pp_size": 1}
    )
    grouped = group_expert_params_in_metadata(metadata)

    assert grouped[f"{prefix}.gate_proj.weight"]["shape"] == [4, 4, 64]
    assert grouped[f"{prefix}.up_proj.weight"]["shape"] == [4, 4, 64]
    assert grouped[f"{prefix}.down_proj.weight"]["shape"] == [4, 64, 32]
    assert grouped[f"{prefix}.down_proj.weight"]["components"][1]["shape"] == [
        4,
        64,
        1,
    ]
