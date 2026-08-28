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

"""Unit test for the train-side expert stacking (``_group_experts``).

``_group_experts`` (``MegatronPolicyWorkerImpl``) stacks this rank's local
per-expert tensors for one projection into ``[E_local, ...]``.  It doesn't use
``self`` and operates on plain tensors, so a dummy ``self`` + CPU tensors suffice.

Importing ``megatron_policy_worker`` pulls in megatron.core, so this is
mcore-marked and skipped where mcore is unavailable.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

# megatron_policy_worker imports both megatron.core and megatron.bridge at
# module top, so guard on both: an env can have megatron.core but not
# megatron.bridge, and importing this test module would otherwise raise a
# collection error (not skip) in non-mcore lanes.
pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")

from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


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
    rows = torch.tensor(shape[:-1]).prod().item()
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
    worker.fp8_cfg = {"fp8_param": True, "fp8_recipe": "mxfp8"}
    worker.cfg = {
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {"precision": "fp8", "is_mx": True},
        }
    }
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


def _group(proj, grouped_name, expert_groups):
    # _group_experts ignores self; pass a dummy.
    return MegatronPolicyWorkerImpl._group_experts(
        SimpleNamespace(), proj, grouped_name, expert_groups
    )


def test_group_experts_stacks_in_order():
    prefix = "model.layers.0.mlp.experts"
    e0 = torch.randn(1536, 4096)
    e1 = torch.randn(1536, 4096)
    e2 = torch.randn(1536, 4096)
    groups = {(prefix, "gate_proj"): [e0, e1, e2]}
    out = _group("gate_proj", f"{prefix}.gate_proj.weight", groups)
    assert out.shape == (3, 1536, 4096)
    # Order preserved (expert 0 first).
    assert torch.equal(out[0], e0)
    assert torch.equal(out[1], e1)
    assert torch.equal(out[2], e2)


def test_group_experts_missing_group_raises():
    groups = {("other.experts", "gate_proj"): [torch.randn(8, 8)]}
    with pytest.raises(AssertionError):
        _group("gate_proj", "model.layers.0.mlp.experts.gate_proj.weight", groups)


def test_group_experts_empty_group_raises():
    prefix = "model.layers.0.mlp.experts"
    with pytest.raises(AssertionError):
        _group("gate_proj", f"{prefix}.gate_proj.weight", {(prefix, "gate_proj"): []})


# --------------------------------------------------------------------------
# build_hf_to_local_param_map (train/src side) — folds this rank's local
# shards (_iter_local_hf_param_shards) into LocalParamSpecs.  Fake the shard
# iterator; _build_expert_groups / _group_experts run for real.
# --------------------------------------------------------------------------
def test_build_hf_to_local_param_map_train_side():
    from nemo_rl.weight_sync.nccl_reshard_utils import HFToLocalParamMap

    w = object.__new__(MegatronPolicyWorkerImpl)  # no __init__ / no megatron state
    prefix = "model.layers.0.mlp.experts"
    direct = torch.randn(8, 16)  # a dense FFN down_proj local shard view
    e0 = torch.randn(128, 16)  # this rank's local expert 0 gate_proj
    e1 = torch.randn(128, 16)  # local expert 1 gate_proj
    w._iter_local_hf_param_shards = lambda: [
        ("model.layers.0.mlp.down_proj.weight", direct),
        (f"{prefix}.0.gate_proj.weight", e0),
        (f"{prefix}.1.gate_proj.weight", e1),
    ]
    refit_info = {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.down_proj.weight",
                    "global_shape": [8, 16],
                },
                {
                    "name": f"{prefix}.gate_proj.weight",
                    "global_shape": [2, 128, 16],
                    "grouped_expert_proj": "gate_proj",
                },
            ]
        },
    }

    pmap = w.build_hf_to_local_param_map(refit_info)
    assert isinstance(pmap, HFToLocalParamMap)

    # Direct: base is the live local view, sent as-is (no hooks).
    d = pmap.get("model.layers.0.mlp.down_proj.weight")
    assert d.base is direct and d.pre is None and d.post is None

    # Grouped expert: pre stacks this rank's per-expert views into [E_local, ...]
    # fresh each refit (base unused — the views are captured in the hook).
    g = pmap.get(f"{prefix}.gate_proj.weight")
    assert g.pre is not None
    ctx = g.pre(g.base)
    assert ctx.buf.shape == (2, 128, 16)
    assert torch.equal(ctx.buf[0], e0) and torch.equal(ctx.buf[1], e1)


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
            assert spec.pre is not None

    down_weight = source_map.get(down_name, role="weight")
    first = down_weight.pre(down_weight.base).buf
    replacement = torch.full((64, 32), 91, dtype=torch.uint8)
    fc2._metadata["rowwise_data"] = replacement
    second = down_weight.pre(down_weight.base).buf
    assert first.data_ptr() != second.data_ptr()
    assert second.view(torch.uint8).data_ptr() == replacement.data_ptr()
    assert torch.equal(second.view(torch.uint8), replacement)


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
    params = [
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
    params = [
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
    first = gate_spec.pre(gate_spec.base).buf
    replacement = torch.full((8, 64), 99, dtype=torch.uint8)
    fc1_members[0]._metadata["rowwise_data"] = replacement
    second = gate_spec.pre(gate_spec.base).buf
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(second[0].view(torch.uint8), replacement[:4])
    assert all(create_if_missing is False for _, create_if_missing in member_calls)
    assert extracted


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
                param_weight=_native_tensor(
                    (64, 64), value_marker=1, scale_marker=2
                ),
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
                param_weight=_native_tensor(
                    (64, 32), value_marker=1, scale_marker=2
                ),
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
                param_weight=_native_tensor(
                    (16, 64), value_marker=1, scale_marker=2
                ),
                global_param_name="decoder.layers.0.mlp.linear_fc1.weight",
            ),
            SimpleNamespace(
                mapping=AutoMapping(
                    "decoder.layers.0.mlp.linear_fc2.weight",
                    down_name,
                ),
                param_weight=_native_tensor(
                    (64, 32), value_marker=3, scale_marker=4
                ),
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
