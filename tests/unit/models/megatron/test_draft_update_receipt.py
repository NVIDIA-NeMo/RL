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

from __future__ import annotations

import ast
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.mcore


def test_pinned_distributed_optimizer_digest_matches_submodule_source() -> None:
    receipt = _receipt_module()
    source_path = (
        Path(__file__).parents[4]
        / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
        / "megatron/core/optimizer/distrib_optimizer.py"
    )
    source = source_path.read_text()
    source_lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    optimizer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DistributedOptimizer"
    )
    method_sources: list[str] = []
    for method_name in receipt._DISTRIBUTED_OPTIMIZER_METHODS:
        method = next(
            node
            for node in optimizer_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        start_line = min(
            [method.lineno, *(decorator.lineno for decorator in method.decorator_list)]
        )
        method_sources.append(
            textwrap.dedent("".join(source_lines[start_line - 1 : method.end_lineno]))
        )

    actual = hashlib.sha256("\n".join(method_sources).encode("utf-8")).hexdigest()

    assert actual == receipt._DISTRIBUTED_OPTIMIZER_SOURCE_SHA256


def _receipt_module() -> Any:
    module_name = "_isolated_draft_update_receipt"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).parents[4] / "nemo_rl/models/megatron/draft/receipt.py"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.fail("draft update receipt implementation is missing")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sharded(
    key: str,
    data: torch.Tensor,
    *,
    global_shape: tuple[int, ...] | None = None,
    global_offset: tuple[int, ...] | None = None,
    replica_id: int | tuple[int, ...] = 0,
) -> Any:
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    return ShardedTensor(
        key=key,
        data=data,
        dtype=data.dtype,
        local_shape=tuple(data.shape),
        global_shape=global_shape or tuple(data.shape),
        global_offset=global_offset or tuple(0 for _ in data.shape),
        axis_fragmentations=None,
        replica_id=replica_id,
    )


class _DraftModel(torch.nn.Module):
    def __init__(self, parameter: torch.nn.Parameter, state: dict[str, Any]) -> None:
        super().__init__()
        self.weight = parameter
        self._sharded_state = state

    def sharded_state_dict(self, **_: Any) -> dict[str, Any]:
        return self._sharded_state


def _decision() -> Any:
    return SimpleNamespace(
        global_step=3,
        decision_id=7,
        update_requested=True,
        draft_refit_requested=True,
        reason="always",
        observed_acceptance=None,
    )


def _install_fake_optimizer_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type[Any], type[Any], type[Any]]:
    class FakeChainedOptimizer:
        pass

    class FakeFloat16OptimizerWithFloat16Params:
        pass

    class FakeDistributedOptimizer:
        pass

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.optimizer.optimizer",
        SimpleNamespace(
            ChainedOptimizer=FakeChainedOptimizer,
            Float16OptimizerWithFloat16Params=FakeFloat16OptimizerWithFloat16Params,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.optimizer.distrib_optimizer",
        SimpleNamespace(DistributedOptimizer=FakeDistributedOptimizer),
    )
    return (
        FakeChainedOptimizer,
        FakeFloat16OptimizerWithFloat16Params,
        FakeDistributedOptimizer,
    )


def _install_fake_mapping_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type[Any], type[Any]]:
    @dataclass
    class FakeShardedTensor:
        key: str
        data: torch.Tensor
        dtype: torch.dtype
        local_shape: tuple[int, ...]
        global_shape: tuple[int, ...]
        global_offset: tuple[int, ...]
        axis_fragmentations: tuple[int, ...] | None
        replica_id: int | tuple[int, ...] = 0
        prepend_axis_num: int = 0

    @dataclass
    class FakeShardedTensorFactory:
        key: str
        data: torch.Tensor
        build_fn: Any
        merge_fn: Any
        replica_id: int | tuple[int, ...] = 0
        flattened_range: slice | None = None

        def build(self) -> Any:
            return self.build_fn(
                self.key,
                self.data,
                self.replica_id,
                self.flattened_range,
            )

    def apply_factories(container: Any) -> None:
        def expand(value: Any) -> Any:
            if isinstance(value, FakeShardedTensorFactory):
                return expand(value.build())
            if isinstance(value, dict):
                return {key: expand(item) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item) for item in value]
            if isinstance(value, tuple):
                return tuple(expand(item) for item in value)
            return value

        expanded = expand(container)
        container.clear()
        container.update(expanded)

    monkeypatch.setitem(
        sys.modules,
        "megatron.core.dist_checkpointing.mapping",
        SimpleNamespace(
            ShardedTensor=FakeShardedTensor,
            ShardedTensorFactory=FakeShardedTensorFactory,
            apply_factories=apply_factories,
            is_main_replica=lambda replica_id: (
                replica_id == 0
                if isinstance(replica_id, int)
                else all(item == 0 for item in replica_id)
            ),
        ),
    )
    return FakeShardedTensor, FakeShardedTensorFactory


def _root_with_group_records(
    receipt: Any,
    group_records: list[Any],
) -> str:
    records = [
        receipt.CanonicalDraftStateRecord.for_tensor(
            component="model",
            logical_key="draft.weight",
            global_shape=(1,),
            global_offset=(0,),
            local_tensor=torch.tensor([1.0]),
            replica_id=0,
        ),
        receipt.CanonicalDraftStateRecord.for_scalar(
            component="optimizer",
            logical_key="draft.weight/state_initialized",
            value=False,
            replica_id=0,
            record_kind="state_marker",
        ),
        *group_records,
    ]
    return receipt.canonical_draft_state_roots(records).optimizer_sha256


def test_float16_group_hyperparameters_follow_live_to_master_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_module()
    _, float16_cls, _ = _install_fake_optimizer_modules(monkeypatch)
    model_parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    master_parameter = torch.nn.Parameter(model_parameter.float())
    optimizer = object.__new__(float16_cls)
    optimizer.float16_groups = [[model_parameter]]
    optimizer.fp32_from_float16_groups = [[master_parameter]]
    optimizer.fp32_from_fp32_groups = [[]]
    optimizer.optimizer = SimpleNamespace(
        param_groups=[{"params": [master_parameter], "lr": 1.0e-4}]
    )

    before_records = receipt._optimizer_group_records(optimizer, {model_parameter})
    before = _root_with_group_records(receipt, before_records)
    optimizer.optimizer.param_groups[0]["lr"] = 2.0e-4
    after = _root_with_group_records(
        receipt,
        receipt._optimizer_group_records(optimizer, {model_parameter}),
    )

    assert sum(record.logical_key.endswith("/lr") for record in before_records) == 1
    assert before != after


def test_distributed_group_hyperparameters_follow_model_group_index_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_module()
    _, _, distributed_cls = _install_fake_optimizer_modules(monkeypatch)
    model_parameter = torch.nn.Parameter(torch.ones(1))
    local_main_slice = torch.nn.Parameter(torch.ones(1))
    optimizer = object.__new__(distributed_cls)
    optimizer.model_param_group_index_map = {model_parameter: (0, 0)}
    optimizer.optimizer = SimpleNamespace(
        param_groups=[{"params": [local_main_slice], "lr": 1.0e-4}]
    )

    before_records = receipt._optimizer_group_records(optimizer, {model_parameter})
    before = _root_with_group_records(receipt, before_records)
    optimizer.optimizer.param_groups[0]["lr"] = 2.0e-4
    after = _root_with_group_records(
        receipt,
        receipt._optimizer_group_records(optimizer, {model_parameter}),
    )

    assert sum(record.logical_key.endswith("/lr") for record in before_records) == 1
    assert before != after


def test_factory_transforms_regular_optimizer_moments_on_a_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_module()
    _install_fake_optimizer_modules(monkeypatch)
    sharded_tensor_cls, factory_cls = _install_fake_mapping_module(monkeypatch)
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))

    def build(
        key: str,
        data: torch.Tensor,
        replica_id: int | tuple[int, ...],
        flattened_range: slice | None,
    ) -> dict[str, Any]:
        assert flattened_range is None
        return {
            "left": sharded_tensor_cls(
                f"{key}.left",
                data[:2],
                data.dtype,
                (2,),
                (2,),
                (0,),
                None,
                replica_id,
            ),
            "right": sharded_tensor_cls(
                f"{key}.right",
                data[2:],
                data.dtype,
                (2,),
                (2,),
                (0,),
                None,
                replica_id,
            ),
        }

    factory = factory_cls(
        key="draft.weight",
        data=parameter,
        build_fn=build,
        merge_fn=lambda state: torch.cat([state["left"], state["right"]]),
    )
    model = _DraftModel(parameter, {"weight": factory})
    optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(
            param_groups=[{"params": [parameter], "lr": 0.1}],
            state={
                parameter: {
                    "exp_avg": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                }
            },
        )
    )

    records = receipt.canonical_draft_state_records(model, optimizer)
    moments = {
        record.logical_key: record
        for record in records
        if record.component == "optimizer" and "exp_avg" in record.logical_key
    }
    left_expected = receipt.CanonicalDraftStateRecord.for_tensor(
        component="optimizer",
        logical_key="unused",
        global_shape=(2,),
        global_offset=(0,),
        local_tensor=torch.tensor([1.0, 2.0]),
        replica_id=0,
    )
    right_expected = receipt.CanonicalDraftStateRecord.for_tensor(
        component="optimizer",
        logical_key="unused",
        global_shape=(2,),
        global_offset=(0,),
        local_tensor=torch.tensor([3.0, 4.0]),
        replica_id=0,
    )

    assert factory.data is parameter
    assert set(moments) == {
        "draft.weight/exp_avg.left",
        "draft.weight/exp_avg.right",
    }
    assert moments["draft.weight/exp_avg.left"].num_bytes == 8
    assert moments["draft.weight/exp_avg.right"].num_bytes == 8
    assert (
        moments["draft.weight/exp_avg.left"].tensor_sha256
        == left_expected.tensor_sha256
    )
    assert (
        moments["draft.weight/exp_avg.right"].tensor_sha256
        == right_expected.tensor_sha256
    )


@pytest.mark.parametrize(
    ("owned_range", "expected"),
    [
        ((0, 2), {"draft.weight/exp_avg.left": ((0, 2), [1.0, 2.0])}),
        ((2, 4), {"draft.weight/exp_avg.right": ((0, 2), [1.0, 2.0])}),
        (
            (1, 3),
            {
                "draft.weight/exp_avg.left": ((1, 2), [1.0]),
                "draft.weight/exp_avg.right": ((0, 1), [2.0]),
            },
        ),
    ],
)
def test_distributed_factory_partitions_local_slice_from_full_leaves(
    monkeypatch: pytest.MonkeyPatch,
    owned_range: tuple[int, int],
    expected: dict[str, tuple[tuple[int, int], list[float]]],
) -> None:
    receipt = _receipt_module()
    _, _, distributed_cls = _install_fake_optimizer_modules(monkeypatch)
    sharded_tensor_cls, factory_cls = _install_fake_mapping_module(monkeypatch)
    monkeypatch.setattr(
        receipt,
        "validate_pinned_distributed_optimizer_class",
        lambda cls: None,
    )
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    build_ranges: list[tuple[int, int] | None] = []

    def build(
        key: str,
        data: torch.Tensor,
        replica_id: int | tuple[int, ...],
        flattened_range: slice | None,
    ) -> dict[str, Any]:
        interval = (
            None
            if flattened_range is None
            else (int(flattened_range.start), int(flattened_range.stop))
        )
        build_ranges.append(interval)
        left, right = torch.chunk(data, 2)
        return {
            "left": sharded_tensor_cls(
                f"{key}.left",
                left,
                data.dtype,
                (2,),
                (2,),
                (0,),
                None,
                replica_id,
            ),
            "right": sharded_tensor_cls(
                f"{key}.right",
                right,
                data.dtype,
                (2,),
                (2,),
                (0,),
                None,
                replica_id,
            ),
        }

    factory = factory_cls(
        key="draft.weight",
        data=parameter,
        build_fn=build,
        merge_fn=lambda state: torch.cat([state["left"], state["right"]]),
        replica_id=(0, 0, 0),
    )
    model = _DraftModel(parameter, {"weight": factory})
    optimizer = object.__new__(distributed_cls)
    optimizer.distributed_optimizer_instance_id = 0
    optimizer.model_param_group_index_map = {parameter: (0, 0)}
    optimizer.optimizer = SimpleNamespace(
        param_groups=[{"params": [torch.nn.Parameter(torch.ones(2))], "lr": 0.1}]
    )
    optimizer._get_model_param_range_map = lambda _: {
        "param": SimpleNamespace(start=owned_range[0], end=owned_range[1])
    }
    optimizer._get_main_param_and_optimizer_states = lambda _: {
        "param": torch.tensor([10.0, 11.0]),
        "exp_avg": torch.tensor([1.0, 2.0]),
    }

    records = receipt.canonical_draft_state_records(model, optimizer)
    moments = {
        record.logical_key: record
        for record in records
        if record.component == "optimizer" and "exp_avg" in record.logical_key
    }

    assert all(interval is None for interval in build_ranges)
    assert set(moments) == set(expected)
    for key, (expected_range, expected_values) in expected.items():
        expected_record = receipt.CanonicalDraftStateRecord.for_flattened_tensor(
            component="optimizer",
            logical_key=key,
            global_shape=(2,),
            global_offset=(0,),
            base_local_shape=(2,),
            flattened_range=expected_range,
            local_tensor=torch.tensor(expected_values),
            replica_id=0,
        )
        assert moments[key].flattened_range == expected_range
        assert moments[key].tensor_sha256 == expected_record.tensor_sha256


def test_distributed_factory_rejects_nonpartitioning_full_leaves() -> None:
    receipt = _receipt_module()
    source = torch.tensor([0.0, 1.0, 2.0, 3.0])
    full_leaves = [
        SimpleNamespace(key="draft.weight.left", data=source[:2]),
        SimpleNamespace(key="draft.weight.right", data=source[1:3]),
    ]

    with pytest.raises(RuntimeError, match="gap, overlap, or source-order"):
        receipt._factory_flattened_leaf_ranges(
            full_leaves,
            model_key="draft.weight",
            state_key="draft.weight/exp_avg",
            source_tensor=source,
            local_state_tensor=torch.tensor([1.0, 2.0]),
            source_local_numel=4,
            flattened_range=(0, 2),
        )


def test_schema_rejects_gapped_flattened_optimizer_coverage() -> None:
    receipt = _receipt_module()
    records = [
        receipt.CanonicalDraftStateRecord.for_tensor(
            component="model",
            logical_key="draft.weight",
            global_shape=(4,),
            global_offset=(0,),
            local_tensor=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            replica_id=0,
        ),
        receipt.CanonicalDraftStateRecord.for_flattened_tensor(
            component="optimizer",
            logical_key="draft.weight/exp_avg",
            global_shape=(4,),
            global_offset=(0,),
            base_local_shape=(4,),
            flattened_range=(0, 2),
            local_tensor=torch.tensor([0.1, 0.2]),
            replica_id=0,
        ),
        receipt.CanonicalDraftStateRecord.for_flattened_tensor(
            component="optimizer",
            logical_key="draft.weight/exp_avg",
            global_shape=(4,),
            global_offset=(0,),
            base_local_shape=(4,),
            flattened_range=(3, 4),
            local_tensor=torch.tensor([0.4]),
            replica_id=0,
        ),
    ]

    with pytest.raises(RuntimeError, match="gapped flattened"):
        receipt.canonical_draft_state_roots(records)


def test_roots_are_order_independent_and_domain_separated() -> None:
    receipt = _receipt_module()
    records = [
        receipt.CanonicalDraftStateRecord.for_tensor(
            component="model",
            logical_key="draft.weight",
            global_shape=(2,),
            global_offset=(0,),
            local_tensor=torch.tensor([1, 2], dtype=torch.int32),
            replica_id=0,
        ),
        receipt.CanonicalDraftStateRecord.for_tensor(
            component="optimizer",
            logical_key="draft.weight/exp_avg",
            global_shape=(2,),
            global_offset=(0,),
            local_tensor=torch.tensor([3, 4], dtype=torch.int32),
            replica_id=0,
        ),
        receipt.CanonicalDraftStateRecord.for_scalar(
            component="optimizer",
            logical_key="optimizer.0.group.1/lr",
            value=1.0e-5,
            replica_id=0,
        ),
    ]

    roots = receipt.canonical_draft_state_roots(records)
    reversed_roots = receipt.canonical_draft_state_roots(list(reversed(records)))

    assert roots == reversed_roots
    assert roots.model_sha256 != roots.optimizer_sha256
    assert len(roots.model_sha256) == len(roots.optimizer_sha256) == 64


def test_factory_expands_on_a_container_copy() -> None:
    receipt = _receipt_module()
    from megatron.core.dist_checkpointing.mapping import (
        ShardedTensor,
        ShardedTensorFactory,
    )
    from megatron.core.optimizer.optimizer import FP32Optimizer

    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))

    def build(
        key: str,
        data: torch.Tensor,
        replica_id: int | tuple[int, ...],
        flattened_range: slice | None,
    ) -> dict[str, ShardedTensor]:
        assert flattened_range is None
        return {
            "left": _sharded(f"{key}.left", data[:2], replica_id=replica_id),
            "right": _sharded(f"{key}.right", data[2:], replica_id=replica_id),
        }

    factory = ShardedTensorFactory(
        key="draft.weight",
        data=parameter,
        build_fn=build,
        merge_fn=lambda state: torch.cat([state["left"], state["right"]]),
        replica_id=0,
    )
    state = {"weight": factory}
    model = _DraftModel(parameter, state)
    base = torch.optim.AdamW([{"params": [parameter]}], lr=0.1)
    base.state[parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "exp_avg_sq": torch.tensor([2.0, 3.0, 4.0, 5.0]),
    }
    optimizer = object.__new__(FP32Optimizer)
    optimizer.optimizer = base

    records = receipt.canonical_draft_state_records(model, optimizer)

    assert state["weight"] is factory
    model_keys = {
        record.logical_key for record in records if record.component == "model"
    }
    assert model_keys == {"draft.weight.left", "draft.weight.right"}
    moments = {
        record.logical_key: record
        for record in records
        if record.component == "optimizer" and "exp_avg" in record.logical_key
    }
    assert moments["draft.weight/exp_avg.left"].num_bytes == 8
    assert moments["draft.weight/exp_avg.right"].num_bytes == 8


def test_uninitialized_adam_emits_false_state_marker_without_fabrication() -> None:
    receipt = _receipt_module()
    from megatron.core.optimizer.optimizer import FP32Optimizer

    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad_norm_group = "draft"
    model = _DraftModel(parameter, {"weight": _sharded("draft.weight", parameter)})
    base = torch.optim.AdamW([{"params": [parameter], "lr": 0.1}])
    optimizer = object.__new__(FP32Optimizer)
    optimizer.optimizer = base

    records = receipt.canonical_draft_state_records(model, optimizer)

    markers = [record for record in records if record.record_kind == "state_marker"]
    assert len(markers) == 1
    assert markers[0].logical_key == "draft.weight/state_initialized"
    assert markers[0].scalar_value is False
    assert not any("exp_avg" in record.logical_key for record in records)


def test_float16_adapter_uses_live_model_to_master_identity() -> None:
    receipt = _receipt_module()
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    parameter.grad_norm_group = "draft"
    master = torch.nn.Parameter(parameter.float())
    master.grad_norm_group = "draft"
    base = torch.optim.AdamW([{"params": [master]}], lr=0.1)
    base.state[master] = {
        "step": torch.tensor(4.0),
        "exp_avg": torch.tensor([0.25, 0.5]),
        "exp_avg_sq": torch.tensor([0.125, 0.25]),
    }
    optimizer = object.__new__(Float16OptimizerWithFloat16Params)
    optimizer.optimizer = base
    optimizer.float16_groups = [[parameter]]
    optimizer.fp32_from_float16_groups = [[master]]
    optimizer.fp32_from_fp32_groups = [[]]
    model = _DraftModel(parameter, {"weight": _sharded("draft.weight", parameter)})

    records = receipt.canonical_draft_state_records(model, optimizer)

    assert any(record.logical_key == "draft.weight/exp_avg" for record in records)
    assert any(
        record.logical_key == "draft.weight/state_initialized"
        and record.scalar_value is True
        for record in records
    )


def test_distributed_adapter_reads_only_local_private_slice_without_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_module()
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    receipt.validate_pinned_distributed_optimizer_class(DistributedOptimizer)
    parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    parameter.grad_norm_group = "draft"
    main = torch.nn.Parameter(torch.tensor([10.0, 11.0]))
    main.grad_norm_group = "draft"
    inner = SimpleNamespace(
        param_groups=[{"params": [main], "lr": 0.1}],
        state={
            main: {
                "step": torch.tensor(3.0),
                "exp_avg": torch.tensor([0.5, 0.75]),
                "exp_avg_sq": torch.tensor([0.25, 0.5]),
            }
        },
    )
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.optimizer = inner
    optimizer.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False
    )
    optimizer.model_param_group_index_map = {parameter: (0, 0)}
    dtype_key = (torch.float32, torch.float32)
    optimizer.model_param_gbuf_map = {parameter: (0, dtype_key, 0)}
    optimizer.gbuf_ranges = [
        {
            dtype_key: [
                {"param_map": {parameter: {"param": SimpleNamespace(start=1, end=3)}}}
            ]
        }
    ]
    optimizer.distributed_optimizer_instance_id = 0
    model = _DraftModel(
        parameter,
        {
            "weight": _sharded(
                "draft.weight",
                parameter,
                replica_id=(0, 0, 0),
            )
        },
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        MagicMock(side_effect=AssertionError("full DP gather must not run")),
    )
    optimizer.get_parameter_state_dp_zero = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("full parameter-state gather must not run")
    )
    optimizer.sharded_state_dict = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("full target optimizer state must not be built")
    )

    records = receipt.canonical_draft_state_records(model, optimizer)

    exp_avg = next(
        record for record in records if record.logical_key == "draft.weight/exp_avg"
    )
    assert exp_avg.record_kind == "flattened_tensor"
    assert exp_avg.flattened_range == (1, 3)
    assert exp_avg.base_local_shape == (4,)
    optimizer.get_parameter_state_dp_zero.assert_not_called()
    optimizer.sharded_state_dict.assert_not_called()


def test_distributed_adapter_rejects_source_or_type_drift() -> None:
    receipt = _receipt_module()
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    class DriftedDistributedOptimizer(DistributedOptimizer):
        pass

    with pytest.raises(RuntimeError, match="pinned MCore"):
        receipt.validate_pinned_distributed_optimizer_class(DriftedDistributedOptimizer)


def test_disabled_capture_calls_no_factory_or_receipt_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_module()
    shard_factory = MagicMock(side_effect=AssertionError("factory called"))
    gather = MagicMock(side_effect=AssertionError("receipt collective called"))
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    result = receipt.maybe_capture_draft_update_receipt(
        capture_draft_update_receipt=False,
        decision=_decision(),
        draft_update_successful=True,
        shard_factory=shard_factory,
        wrapper_visible=True,
    )

    assert result is None
    shard_factory.assert_not_called()
    gather.assert_not_called()


def test_receipt_capture_world_consenses_remote_factory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_module()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output: list[Any], local: dict[str, Any]) -> None:
        output[0] = local
        output[1] = {
            "rank": 1,
            "records": [],
            "error": "RuntimeError: pinned MCore drift",
            "wrapper_visible": True,
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="rank 1: RuntimeError: pinned MCore drift"):
        receipt.maybe_capture_draft_update_receipt(
            capture_draft_update_receipt=True,
            decision=_decision(),
            draft_update_successful=True,
            shard_factory=lambda: [],
            wrapper_visible=True,
        )


def test_receipt_capture_publishes_only_on_lowest_wrapper_visible_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_module()
    records = [
        receipt.CanonicalDraftStateRecord.for_tensor(
            component="model",
            logical_key="draft.weight",
            global_shape=(1,),
            global_offset=(0,),
            local_tensor=torch.tensor([1.0]),
            replica_id=0,
        ),
        receipt.CanonicalDraftStateRecord.for_scalar(
            component="optimizer",
            logical_key="draft.weight/state_initialized",
            value=False,
            replica_id=0,
            record_kind="state_marker",
        ),
    ]
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)

    def gather(output: list[Any], local: dict[str, Any]) -> None:
        output[0] = {
            "rank": 0,
            "records": records,
            "error": None,
            "wrapper_visible": False,
        }
        output[1] = local
        output[2] = {
            "rank": 2,
            "records": [],
            "error": None,
            "wrapper_visible": True,
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    captured = receipt.maybe_capture_draft_update_receipt(
        capture_draft_update_receipt=True,
        decision=_decision(),
        draft_update_successful=True,
        shard_factory=lambda: [],
        wrapper_visible=True,
    )

    assert captured is not None
    assert captured["publisher_rank"] == 1
    assert captured["receipt"] is not None


def test_select_published_receipt_requires_one_visible_publisher() -> None:
    receipt = _receipt_module()
    expected = {
        "successful": True,
        "decision_id": 7,
        "global_step": 3,
        "draft_model_sha256": "1" * 64,
        "draft_optimizer_sha256": "2" * 64,
    }
    rows = [
        {
            "world_rank": 0,
            "draft_update_receipt_publisher_rank": 1,
            "is_replica_leader": True,
        },
        {
            "world_rank": 1,
            "draft_update_receipt_publisher_rank": 1,
            "draft_update_receipt": expected,
            "is_replica_leader": True,
        },
        {
            "world_rank": 2,
            "draft_update_receipt_publisher_rank": 1,
            "is_replica_leader": False,
        },
    ]

    selected = receipt.select_published_draft_update_receipt(
        rows,
        capture_draft_update_receipt=True,
        receipt_required=True,
    )

    assert selected == expected


def test_selector_rejects_fabricated_receipt_when_capture_is_disabled() -> None:
    receipt = _receipt_module()

    with pytest.raises(RuntimeError, match="disabled receipt capture"):
        receipt.select_published_draft_update_receipt(
            [
                {
                    "world_rank": 0,
                    "draft_update_receipt_publisher_rank": 0,
                    "draft_update_receipt": {
                        "successful": True,
                        "decision_id": 7,
                        "global_step": 3,
                        "draft_model_sha256": "1" * 64,
                        "draft_optimizer_sha256": "2" * 64,
                    },
                }
            ],
            capture_draft_update_receipt=False,
            receipt_required=False,
        )


def test_optimizer_template_replica_is_rewritten_without_mutating_model() -> None:
    receipt = _receipt_module()
    parameter = torch.nn.Parameter(torch.ones(2))
    template = _sharded(
        "draft.weight",
        parameter,
        replica_id=(0, 0, 3),
    )

    rewritten = receipt.optimizer_replica_id(template.replica_id, instance_id=0)

    assert rewritten == (0, 0, 0)
    assert template.replica_id == (0, 0, 3)
