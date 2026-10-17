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

``_group_experts`` (``MegatronPolicyWorkerImpl``) creates a ``pre`` hook that
prepares this rank's full expert group before selecting one HF projection.
Plain CPU tensors suffice for unquantized sources.

Importing ``megatron_policy_worker`` pulls in megatron.core, so this is
mcore-marked and skipped where mcore is unavailable.
"""

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
from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)
from nemo_rl.weight_sync.nccl_reshard_utils import (  # noqa: E402
    LocalParamSpec,
    RefitCtx,
)

pytestmark = pytest.mark.mcore


def _source(
    tensor: torch.Tensor, hf_view: LocalHFParamSpec | None = None
) -> LocalParamSpec:
    worker = object.__new__(MegatronPolicyWorkerImpl)
    return worker._default_param_spec(
        tensor, hf_view if hf_view is not None else LocalHFParamSpec("weight")
    )


def _group(proj, grouped_name, expert_groups):
    worker = object.__new__(MegatronPolicyWorkerImpl)
    grouped = worker._group_experts(proj, grouped_name, expert_groups)
    return grouped.materialize().buf


def test_group_experts_missing_group_raises():
    groups = {("other.experts", "gate_proj"): [_source(torch.randn(8, 8))]}
    with pytest.raises(AssertionError, match="no local experts.*PP-filter"):
        _group("gate_proj", "model.layers.0.mlp.experts.gate_proj.weight", groups)


def test_group_experts_empty_group_raises():
    prefix = "model.layers.0.mlp.experts"
    with pytest.raises(AssertionError, match="no local experts.*PP-filter"):
        _group("gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): []})


@pytest.mark.parametrize("mismatch", ["shape", "split"])
def test_group_experts_rejects_incompatible_hf_views(
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
def test_group_experts_selects_equivalent_axes_after_group_preparation(
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
    worker = object.__new__(MegatronPolicyWorkerImpl)
    grouped = worker._group_experts(
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
# iterator; _build_expert_groups / _group_experts run for real.
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
