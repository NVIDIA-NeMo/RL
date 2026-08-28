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

import math
from types import SimpleNamespace
from typing import Any, cast

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
    worker.fp8_cfg = {"fp8_param": True, "fp8_recipe": "mxfp8"}
    worker.cfg = cast(
        Any,
        {
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"precision": "fp8", "is_mx": True},
            }
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


def _group(proj, grouped_name, expert_groups):
    # _group_experts ignores self; pass a dummy.
    return MegatronPolicyWorkerImpl._group_experts(
        cast(Any, SimpleNamespace()), proj, grouped_name, expert_groups
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
    w._iter_local_hf_param_shards = cast(
        Any,
        lambda: iter(
            [
                ("model.layers.0.mlp.down_proj.weight", direct),
                (f"{prefix}.0.gate_proj.weight", e0),
                (f"{prefix}.1.gate_proj.weight", e1),
            ]
        ),
    )
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
    assert d is not None
    assert d.base is direct and d.pre is None and d.post is None

    # Grouped expert: pre stacks this rank's per-expert views into [E_local, ...]
    # fresh each refit (base unused — the views are captured in the hook).
    g = pmap.get(f"{prefix}.gate_proj.weight")
    assert g is not None and g.pre is not None
    pre = g.pre
    ctx = pre(g.base)
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


@pytest.mark.parametrize("owns_grouped_params", [True, False])
def test_native_mxfp8_build_refit_tasks_bypasses_strict_grouped_names(
    monkeypatch: pytest.MonkeyPatch,
    owns_grouped_params: bool,
) -> None:
    from contextlib import nullcontext

    from megatron.bridge.models.conversion import model_bridge
    from megatron.bridge.models.conversion.param_mapping import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
    )
    from megatron.core import fp8_utils

    from nemo_rl.models.megatron import draft

    fc1_name = "decoder.layers.0.mlp.experts.linear_fc1.weight"
    fc2_name = "decoder.layers.0.mlp.experts.linear_fc2.weight"
    fc1_param = object()
    fc2_param = object()
    params = (
        [(fc1_name, fc1_param), (fc2_name, fc2_param)] if owns_grouped_params else []
    )
    mappings = {
        f"{fc1_name}0": FusedGatedExpertMapping(
            f"{fc1_name}0", "model.layers.0.mlp.experts.gate_up_proj"
        ),
        f"{fc2_name}0": FusedExpertMapping(
            f"{fc2_name}0", "model.layers.0.mlp.experts.down_proj"
        ),
    }
    lookups = []

    class FakeRegistry:
        def megatron_to_hf_lookup(self, name: str):
            lookups.append(name)
            return mappings.get(name)

    class FakeBridge:
        def _unwrap_name(self, name: str) -> str:
            return name

        def mapping_registry(self) -> FakeRegistry:
            return FakeRegistry()

        def _megatron_global_param_names_all_pp_ranks(self, _models):
            return [fc1_name, fc2_name]

    class FakeModel:
        config = SimpleNamespace(moe_single_grouped_weight=True)

        def named_parameters(self):
            return iter(params)

    strict_calls = []

    def strict_get_conversion_tasks(_models):
        strict_calls.append(True)
        raise ValueError(f"No mapping found for {fc1_name}")

    monkeypatch.setattr(draft, "draft_model_detached", lambda _models: nullcontext())
    monkeypatch.setattr(
        model_bridge,
        "_megatron_local_name_to_global",
        lambda _models, _config, local_name, _vp_stage: local_name,
    )
    monkeypatch.setattr(
        fp8_utils,
        "is_grouped_mxfp8tensor",
        lambda param: param in (fc1_param, fc2_param),
    )
    worker = _native_worker([])
    worker.model = FakeModel()
    worker.megatron_bridge = SimpleNamespace(
        _model_bridge=FakeBridge(),
        hf_pretrained=SimpleNamespace(),
        get_conversion_tasks=strict_get_conversion_tasks,
    )

    tasks = worker._build_refit_conversion_tasks()

    assert strict_calls == []
    assert [task.global_param_name for task in tasks] == [fc1_name, fc2_name]
    assert [task.param_weight for task in tasks] == (
        [fc1_param, fc2_param] if owns_grouped_params else [None, None]
    )
    assert lookups == [f"{fc1_name}0", f"{fc2_name}0"]


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
        worker.nccl_reshard_refit()

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
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        GatedMLPMapping,
    )

    native_prefix = "model.layers.0.mlp.experts.0"
    ignored_prefix = "model.layers.1.mlp.experts.0"
    native_fc1 = SimpleNamespace(
        mapping=GatedMLPMapping(
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
            gate=f"{native_prefix}.gate_proj.weight",
            up=f"{native_prefix}.up_proj.weight",
        ),
        param_weight=_native_tensor((8, 64), value_marker=1, scale_marker=2),
        global_param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight",
    )
    native_fc2 = SimpleNamespace(
        mapping=AutoMapping(
            "decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
            f"{native_prefix}.down_proj.weight",
        ),
        param_weight=_native_tensor((64, 32), value_marker=3, scale_marker=4),
        global_param_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight",
    )
    ignored_fc1 = SimpleNamespace(
        mapping=GatedMLPMapping(
            "decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight",
            gate=f"{ignored_prefix}.gate_proj.weight",
            up=f"{ignored_prefix}.up_proj.weight",
        ),
        param_weight=torch.zeros((8, 64), dtype=torch.bfloat16),
        global_param_name="decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight",
    )
    ignored_fc2 = SimpleNamespace(
        mapping=AutoMapping(
            "decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.weight",
            f"{ignored_prefix}.down_proj.weight",
        ),
        param_weight=torch.zeros((64, 32), dtype=torch.bfloat16),
        global_param_name="decoder.layers.1.mlp.experts.local_experts.0.linear_fc2.weight",
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
