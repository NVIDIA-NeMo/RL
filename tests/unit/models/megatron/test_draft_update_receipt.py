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

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = pytest.mark.mcore


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
    optimizer = object.__new__(FP32Optimizer)
    optimizer.optimizer = base

    records = receipt.canonical_draft_state_records(model, optimizer)

    assert state["weight"] is factory
    model_keys = {
        record.logical_key for record in records if record.component == "model"
    }
    assert model_keys == {"draft.weight.left", "draft.weight.right"}


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
