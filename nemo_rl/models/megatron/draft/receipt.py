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
"""Canonical receipts for successful online draft-model optimizer updates."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import textwrap
from dataclasses import asdict, dataclass, replace
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypedDict,
)

import torch

_DISTRIBUTED_OPTIMIZER_SOURCE_SHA256 = (
    "c048e93bbcb079ff50f52ef366e685fab80115665941beac1cbaa358e88145d0"
)
_DISTRIBUTED_OPTIMIZER_METHODS = (
    "_build_model_and_main_param_groups",
    "_get_model_param_range_map",
    "_get_main_param_and_optimizer_states",
)


class DraftUpdateDecisionLike(Protocol):
    global_step: int
    decision_id: int
    update_requested: bool


class _ReceiptEnvelope(TypedDict):
    rank: int
    records: list[CanonicalDraftStateRecord]
    error: str | None
    wrapper_visible: bool


@dataclass(frozen=True)
class CanonicalDraftStateRoots:
    model_sha256: str
    optimizer_sha256: str


@dataclass(frozen=True)
class CanonicalDraftStateRecord:
    """Serializable, content-addressed description of one canonical shard."""

    component: Literal["model", "optimizer"]
    logical_key: str
    record_kind: Literal[
        "tensor", "flattened_tensor", "scalar", "group", "state_marker"
    ]
    replica_id: int | tuple[int, ...]
    global_shape: tuple[int, ...] | None = None
    global_offset: tuple[int, ...] | None = None
    base_local_shape: tuple[int, ...] | None = None
    flattened_range: tuple[int, int] | None = None
    dtype: str | None = None
    num_bytes: int | None = None
    tensor_sha256: str | None = None
    scalar_value: Any = None

    @classmethod
    def for_tensor(
        cls,
        *,
        component: Literal["model", "optimizer"],
        logical_key: str,
        global_shape: Sequence[int],
        global_offset: Sequence[int],
        local_tensor: torch.Tensor,
        replica_id: int | tuple[int, ...],
        base_local_shape: Sequence[int] | None = None,
    ) -> CanonicalDraftStateRecord:
        return cls(
            component=component,
            logical_key=logical_key,
            record_kind="tensor",
            replica_id=_canonical_replica_id(replica_id),
            global_shape=tuple(int(x) for x in global_shape),
            global_offset=tuple(int(x) for x in global_offset),
            base_local_shape=tuple(
                int(x)
                for x in (
                    local_tensor.shape if base_local_shape is None else base_local_shape
                )
            ),
            dtype=str(local_tensor.dtype),
            num_bytes=local_tensor.numel() * local_tensor.element_size(),
            tensor_sha256=_tensor_sha256(local_tensor),
        )

    @classmethod
    def for_flattened_tensor(
        cls,
        *,
        component: Literal["model", "optimizer"],
        logical_key: str,
        global_shape: Sequence[int],
        global_offset: Sequence[int],
        base_local_shape: Sequence[int],
        flattened_range: tuple[int, int],
        local_tensor: torch.Tensor,
        replica_id: int | tuple[int, ...],
    ) -> CanonicalDraftStateRecord:
        return cls(
            component=component,
            logical_key=logical_key,
            record_kind="flattened_tensor",
            replica_id=_canonical_replica_id(replica_id),
            global_shape=tuple(int(x) for x in global_shape),
            global_offset=tuple(int(x) for x in global_offset),
            base_local_shape=tuple(int(x) for x in base_local_shape),
            flattened_range=(int(flattened_range[0]), int(flattened_range[1])),
            dtype=str(local_tensor.dtype),
            num_bytes=local_tensor.numel() * local_tensor.element_size(),
            tensor_sha256=_tensor_sha256(local_tensor),
        )

    @classmethod
    def for_scalar(
        cls,
        *,
        component: Literal["model", "optimizer"],
        logical_key: str,
        value: Any,
        replica_id: int | tuple[int, ...],
        record_kind: Literal["scalar", "group", "state_marker"] = "scalar",
    ) -> CanonicalDraftStateRecord:
        return cls(
            component=component,
            logical_key=logical_key,
            record_kind=record_kind,
            replica_id=_canonical_replica_id(replica_id),
            scalar_value=_canonical_scalar(value),
        )


def _canonical_replica_id(value: Any) -> int | tuple[int, ...]:
    if isinstance(value, int):
        return value
    return tuple(int(x) for x in value)


def _canonical_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(
                "only zero-dimensional optimizer tensors are scalar records"
            )
        return _canonical_scalar(value.detach().cpu().item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer scalar records must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_scalar(x) for x in value]
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_scalar(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"unsupported optimizer scalar type: {type(value).__name__}")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    if sys.byteorder != "little":
        raise RuntimeError("canonical draft receipts require little-endian workers")
    detached = tensor.detach().to(device="cpu").contiguous()
    payload = detached.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def optimizer_replica_id(
    replica_id: int | tuple[int, ...], *, instance_id: int
) -> int | tuple[int, ...]:
    """Replace only the optimizer-instance replica coordinate."""
    if isinstance(replica_id, int):
        return int(instance_id)
    if not replica_id:
        return (int(instance_id),)
    return (*replica_id[:-1], int(instance_id))


def validate_pinned_distributed_optimizer_class(cls: type[Any]) -> None:
    """Fail closed unless the exact audited MCore private adapter is present."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    if cls is not DistributedOptimizer:
        raise RuntimeError(
            "pinned MCore DistributedOptimizer type drift; refusing receipt capture"
        )
    source = "\n".join(
        textwrap.dedent(inspect.getsource(getattr(cls, name)))
        for name in _DISTRIBUTED_OPTIMIZER_METHODS
    )
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != _DISTRIBUTED_OPTIMIZER_SOURCE_SHA256:
        raise RuntimeError(
            "pinned MCore DistributedOptimizer source drift; refusing receipt "
            f"capture (expected {_DISTRIBUTED_OPTIMIZER_SOURCE_SHA256}, got {actual})"
        )


@dataclass(frozen=True)
class _ModelTemplate:
    parameter: torch.nn.Parameter
    value: Any


def _iter_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_values(item)
    else:
        yield value


def _container_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _container_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_container_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_container_copy(item) for item in value)
    return value


def _expanded_sharded_values(value: Any) -> list[Any]:
    from megatron.core.dist_checkpointing.mapping import apply_factories

    copied = _container_copy(value)
    wrapper = {"state": copied}
    apply_factories(wrapper)
    return list(_iter_values(wrapper["state"]))


def _model_templates(draft_model: Any) -> list[_ModelTemplate]:
    from megatron.core.dist_checkpointing.mapping import (
        ShardedTensor,
        ShardedTensorFactory,
    )

    state = draft_model.sharded_state_dict()
    templates: list[_ModelTemplate] = []

    def visit(value: Any, inherited: torch.nn.Parameter | None = None) -> None:
        if isinstance(value, ShardedTensorFactory):
            parameter = (
                value.data if isinstance(value.data, torch.nn.Parameter) else inherited
            )
            if parameter is None:
                raise RuntimeError(f"draft factory {value.key!r} has no live parameter")
            for leaf in _expanded_sharded_values(value):
                if not isinstance(leaf, ShardedTensor):
                    raise TypeError(
                        f"draft factory {value.key!r} produced unsupported "
                        f"{type(leaf).__name__}"
                    )
                templates.append(_ModelTemplate(parameter, leaf))
            return
        if isinstance(value, ShardedTensor):
            parameter = (
                value.data if isinstance(value.data, torch.nn.Parameter) else inherited
            )
            if parameter is None:
                raise RuntimeError(f"draft shard {value.key!r} has no live parameter")
            templates.append(_ModelTemplate(parameter, value))
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item, inherited)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, inherited)

    visit(state)
    return templates


def _record_from_sharded_tensor(
    *,
    component: Literal["model", "optimizer"],
    logical_key: str,
    sharded: Any,
    tensor: torch.Tensor,
    replica_id: int | tuple[int, ...],
    flattened_range: tuple[int, int] | None = None,
    base_local_shape: tuple[int, ...] | None = None,
) -> CanonicalDraftStateRecord:
    geometry_shape = (1,) * int(getattr(sharded, "prepend_axis_num", 0)) + tuple(
        sharded.local_shape
    )
    if flattened_range is None:
        return CanonicalDraftStateRecord.for_tensor(
            component=component,
            logical_key=logical_key,
            global_shape=sharded.global_shape,
            global_offset=sharded.global_offset,
            local_tensor=tensor,
            replica_id=replica_id,
            base_local_shape=geometry_shape,
        )
    return CanonicalDraftStateRecord.for_flattened_tensor(
        component=component,
        logical_key=logical_key,
        global_shape=sharded.global_shape,
        global_offset=sharded.global_offset,
        base_local_shape=base_local_shape or geometry_shape,
        flattened_range=flattened_range,
        local_tensor=tensor,
        replica_id=replica_id,
    )


def _regular_param_state_map(optimizer: Any) -> dict[torch.nn.Parameter, Any]:
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        Float16OptimizerWithFloat16Params,
    )

    if isinstance(optimizer, ChainedOptimizer):
        merged: dict[torch.nn.Parameter, Any] = {}
        for child in optimizer.chained_optimizers:
            overlap = merged.keys() & _regular_param_state_map(child).keys()
            if overlap:
                raise RuntimeError(
                    "draft parameter appears in multiple chained optimizers"
                )
            merged.update(_regular_param_state_map(child))
        return merged

    model_to_main: dict[torch.nn.Parameter, torch.nn.Parameter] = {}
    if isinstance(optimizer, Float16OptimizerWithFloat16Params):
        for model_group, main_group in zip(
            optimizer.float16_groups, optimizer.fp32_from_float16_groups
        ):
            model_to_main.update(zip(model_group, main_group))
        for group in optimizer.fp32_from_fp32_groups:
            model_to_main.update((parameter, parameter) for parameter in group)
    else:
        for group in optimizer.optimizer.param_groups:
            model_to_main.update(
                (parameter, parameter) for parameter in group["params"]
            )
    return {
        model_parameter: optimizer.optimizer.state.get(main_parameter, {})
        for model_parameter, main_parameter in model_to_main.items()
    }


def _optimizer_group_records(
    optimizer: Any, draft_parameters: set[torch.nn.Parameter]
) -> list[CanonicalDraftStateRecord]:
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    children = (
        optimizer.chained_optimizers
        if isinstance(optimizer, ChainedOptimizer)
        else [optimizer]
    )
    records: list[CanonicalDraftStateRecord] = []
    for optimizer_index, child in enumerate(children):
        for group_index, group in enumerate(child.optimizer.param_groups):
            if not any(parameter in draft_parameters for parameter in group["params"]):
                continue
            for key, value in sorted(group.items()):
                if key == "params":
                    continue
                records.append(
                    CanonicalDraftStateRecord.for_scalar(
                        component="optimizer",
                        logical_key=f"optimizer.{optimizer_index}.group.{group_index}/{key}",
                        value=value,
                        replica_id=0,
                        record_kind="group",
                    )
                )
    return records


def _distributed_optimizer_records(
    optimizer: Any, templates: list[_ModelTemplate]
) -> list[CanonicalDraftStateRecord]:
    from megatron.core.dist_checkpointing.mapping import is_main_replica

    validate_pinned_distributed_optimizer_class(type(optimizer))
    instance_id = int(optimizer.distributed_optimizer_instance_id)
    records: list[CanonicalDraftStateRecord] = []
    seen_parameters: set[torch.nn.Parameter] = set()
    for template in templates:
        parameter = template.parameter
        if parameter not in optimizer.model_param_group_index_map:
            continue
        if parameter in seen_parameters:
            continue
        seen_parameters.add(parameter)
        sharded = template.value
        replica_id = optimizer_replica_id(sharded.replica_id, instance_id=instance_id)
        if not is_main_replica(replica_id):
            continue
        range_map = optimizer._get_model_param_range_map(parameter)["param"]
        flattened_range = (int(range_map.start), int(range_map.end))
        state = optimizer._get_main_param_and_optimizer_states(parameter)
        initialized = any(key != "param" for key in state)
        records.append(
            CanonicalDraftStateRecord.for_scalar(
                component="optimizer",
                logical_key=f"{sharded.key}/state_initialized",
                value=initialized,
                replica_id=replica_id,
                record_kind="state_marker",
            )
        )
        for key, tensor in sorted(state.items()):
            if key == "param" or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.numel() == 1 and key == "step":
                records.append(
                    CanonicalDraftStateRecord.for_scalar(
                        component="optimizer",
                        logical_key=f"{sharded.key}/{key}",
                        value=tensor,
                        replica_id=replica_id,
                    )
                )
                continue
            records.append(
                _record_from_sharded_tensor(
                    component="optimizer",
                    logical_key=f"{sharded.key}/{key}",
                    sharded=sharded,
                    tensor=tensor,
                    replica_id=replica_id,
                    flattened_range=flattened_range,
                    base_local_shape=tuple(sharded.local_shape),
                )
            )
    return records


def _regular_optimizer_records(
    optimizer: Any, templates: list[_ModelTemplate]
) -> list[CanonicalDraftStateRecord]:
    from megatron.core.dist_checkpointing.mapping import is_main_replica

    records: list[CanonicalDraftStateRecord] = []
    state_map = _regular_param_state_map(optimizer)
    for template in templates:
        if template.parameter not in state_map:
            continue
        sharded = template.value
        if not is_main_replica(sharded.replica_id):
            continue
        state = state_map[template.parameter]
        initialized = bool(state)
        records.append(
            CanonicalDraftStateRecord.for_scalar(
                component="optimizer",
                logical_key=f"{sharded.key}/state_initialized",
                value=initialized,
                replica_id=sharded.replica_id,
                record_kind="state_marker",
            )
        )
        for key, value in sorted(state.items()):
            logical_key = f"{sharded.key}/{key}"
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                records.append(
                    _record_from_sharded_tensor(
                        component="optimizer",
                        logical_key=logical_key,
                        sharded=sharded,
                        tensor=value,
                        replica_id=sharded.replica_id,
                    )
                )
            else:
                records.append(
                    CanonicalDraftStateRecord.for_scalar(
                        component="optimizer",
                        logical_key=logical_key,
                        value=value,
                        replica_id=sharded.replica_id,
                    )
                )
    return records


def canonical_draft_state_records(
    draft_model: Any, optimizer: Any
) -> list[CanonicalDraftStateRecord]:
    """Build canonical local draft shards without gathering full DP state."""
    from megatron.core.dist_checkpointing.mapping import is_main_replica
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    templates = _model_templates(draft_model)
    records: list[CanonicalDraftStateRecord] = []
    for template in templates:
        sharded = template.value
        if not is_main_replica(sharded.replica_id):
            continue
        records.append(
            _record_from_sharded_tensor(
                component="model",
                logical_key=sharded.key,
                sharded=sharded,
                tensor=sharded.data,
                replica_id=sharded.replica_id,
            )
        )

    draft_parameters = {template.parameter for template in templates}
    records.extend(_optimizer_group_records(optimizer, draft_parameters))
    children = (
        optimizer.chained_optimizers
        if isinstance(optimizer, ChainedOptimizer)
        else [optimizer]
    )
    for child in children:
        if isinstance(child, DistributedOptimizer):
            records.extend(_distributed_optimizer_records(child, templates))
        else:
            records.extend(_regular_optimizer_records(child, templates))
    return records


def _record_payload(record: CanonicalDraftStateRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload.pop("replica_id")
    return payload


def _canonical_unique(
    records: Iterable[CanonicalDraftStateRecord],
) -> list[CanonicalDraftStateRecord]:
    unique: dict[str, CanonicalDraftStateRecord] = {}
    logical_identities: dict[tuple[Any, ...], str] = {}
    for record in records:
        _validate_record(record)
        payload = _record_payload(record)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        identity = (
            record.component,
            record.logical_key,
            record.record_kind,
            record.global_shape,
            record.global_offset,
            record.flattened_range,
        )
        previous = logical_identities.setdefault(identity, encoded)
        if previous != encoded:
            raise RuntimeError(
                f"conflicting canonical records for {record.logical_key}"
            )
        unique.setdefault(encoded, record)
    return [unique[key] for key in sorted(unique)]


def _validate_record(record: CanonicalDraftStateRecord) -> None:
    if record.component not in {"model", "optimizer"} or not record.logical_key:
        raise RuntimeError("invalid canonical draft state record identity")
    replica = (
        (record.replica_id,)
        if isinstance(record.replica_id, int)
        else record.replica_id
    )
    if not replica or any(type(item) is not int or item != 0 for item in replica):
        raise RuntimeError(f"noncanonical replica for {record.logical_key}")
    if record.record_kind in {"tensor", "flattened_tensor"}:
        if (
            record.global_shape is None
            or record.global_offset is None
            or record.base_local_shape is None
            or not record.dtype
            or type(record.num_bytes) is not int
            or record.num_bytes < 0
            or not isinstance(record.tensor_sha256, str)
            or len(record.tensor_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record.tensor_sha256
            )
            or any(type(item) is not int or item <= 0 for item in record.global_shape)
            or any(type(item) is not int or item < 0 for item in record.global_offset)
            or any(
                type(item) is not int or item <= 0 for item in record.base_local_shape
            )
        ):
            raise RuntimeError(
                f"invalid canonical tensor record for {record.logical_key}"
            )
        if record.record_kind == "flattened_tensor":
            interval = record.flattened_range
            if (
                interval is None
                or type(interval[0]) is not int
                or type(interval[1]) is not int
                or interval[0] < 0
                or interval[1] <= interval[0]
            ):
                raise RuntimeError(
                    f"invalid flattened tensor record for {record.logical_key}"
                )
        elif record.flattened_range is not None:
            raise RuntimeError(f"unexpected flattened range for {record.logical_key}")
        return
    if record.record_kind not in {"scalar", "group", "state_marker"}:
        raise RuntimeError(f"invalid canonical record kind for {record.logical_key}")
    if any(
        value is not None
        for value in (
            record.global_shape,
            record.global_offset,
            record.base_local_shape,
            record.flattened_range,
            record.dtype,
            record.num_bytes,
            record.tensor_sha256,
        )
    ):
        raise RuntimeError(
            f"scalar record has tensor metadata for {record.logical_key}"
        )
    if record.record_kind == "state_marker" and type(record.scalar_value) is not bool:
        raise RuntimeError(f"state marker is not boolean for {record.logical_key}")
    _canonical_scalar(record.scalar_value)


def _rectangles_overlap(
    left: CanonicalDraftStateRecord, right: CanonicalDraftStateRecord
) -> bool:
    assert left.global_offset is not None and left.base_local_shape is not None
    assert right.global_offset is not None and right.base_local_shape is not None
    return all(
        max(left_offset, right_offset)
        < min(left_offset + left_size, right_offset + right_size)
        for left_offset, left_size, right_offset, right_size in zip(
            left.global_offset,
            left.base_local_shape,
            right.global_offset,
            right.base_local_shape,
        )
    )


def _validate_rectangular_coverage(
    logical_key: str, records: Sequence[CanonicalDraftStateRecord]
) -> None:
    global_shapes = {record.global_shape for record in records}
    if len(global_shapes) != 1 or None in global_shapes:
        raise RuntimeError(f"inconsistent global shape for {logical_key}")
    global_shape = next(iter(global_shapes))
    assert global_shape is not None
    total = 0
    for index, record in enumerate(records):
        if record.global_offset is None or record.base_local_shape is None:
            raise RuntimeError(f"missing shard geometry for {logical_key}")
        if len(record.global_offset) != len(global_shape) or len(
            record.base_local_shape
        ) != len(global_shape):
            raise RuntimeError(f"invalid shard rank for {logical_key}")
        if any(
            offset < 0 or size < 0 or offset + size > global_size
            for offset, size, global_size in zip(
                record.global_offset, record.base_local_shape, global_shape
            )
        ):
            raise RuntimeError(f"out-of-bounds shard for {logical_key}")
        if any(_rectangles_overlap(record, other) for other in records[index + 1 :]):
            raise RuntimeError(f"overlapping shards for {logical_key}")
        total += math.prod(record.base_local_shape)
    if total != math.prod(global_shape):
        raise RuntimeError(f"gapped rectangular shard coverage for {logical_key}")


def _validate_tensor_coverage(records: Sequence[CanonicalDraftStateRecord]) -> None:
    grouped: dict[tuple[str, str], list[CanonicalDraftStateRecord]] = {}
    for record in records:
        if record.record_kind in {"tensor", "flattened_tensor"}:
            grouped.setdefault((record.component, record.logical_key), []).append(
                record
            )
    for (_, logical_key), shards in grouped.items():
        flattened = [
            record for record in shards if record.record_kind == "flattened_tensor"
        ]
        if not flattened:
            _validate_rectangular_coverage(logical_key, shards)
            continue
        if len(flattened) != len(shards):
            raise RuntimeError(f"mixed flattened and tensor records for {logical_key}")
        base_groups: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
            list[CanonicalDraftStateRecord],
        ] = {}
        for record in flattened:
            assert record.global_shape is not None
            assert record.global_offset is not None
            assert record.base_local_shape is not None
            base_groups.setdefault(
                (record.global_shape, record.global_offset, record.base_local_shape), []
            ).append(record)
        base_records: list[CanonicalDraftStateRecord] = []
        for (_, _, base_shape), slices in base_groups.items():
            intervals = sorted(record.flattened_range for record in slices)
            if any(interval is None for interval in intervals):
                raise RuntimeError(f"missing flattened range for {logical_key}")
            cursor = 0
            for interval in intervals:
                assert interval is not None
                if interval[0] != cursor or interval[1] <= interval[0]:
                    raise RuntimeError(f"gapped flattened coverage for {logical_key}")
                cursor = interval[1]
            if cursor != math.prod(base_shape):
                raise RuntimeError(f"gapped flattened coverage for {logical_key}")
            base_records.append(
                replace(slices[0], record_kind="tensor", flattened_range=None)
            )
        _validate_rectangular_coverage(logical_key, base_records)


def canonical_draft_state_roots(
    records: Iterable[CanonicalDraftStateRecord],
) -> CanonicalDraftStateRoots:
    canonical = _canonical_unique(records)
    _validate_tensor_coverage(canonical)

    def root(component: Literal["model", "optimizer"]) -> str:
        component_payload = [
            _record_payload(record)
            for record in canonical
            if record.component == component
        ]
        if not component_payload:
            raise RuntimeError(f"canonical draft {component} state is empty")
        encoded = json.dumps(
            {"domain": f"nemo-rl-draft-{component}-v1", "records": component_payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    return CanonicalDraftStateRoots(
        model_sha256=root("model"), optimizer_sha256=root("optimizer")
    )


def _validate_receipt_envelopes(
    gathered: Sequence[_ReceiptEnvelope | None], *, world_size: int
) -> list[_ReceiptEnvelope]:
    envelopes: list[_ReceiptEnvelope] = []
    ranks: set[int] = set()
    for item in gathered:
        if (
            item is None
            or type(item.get("rank")) is not int
            or item["rank"] in ranks
            or type(item.get("wrapper_visible")) is not bool
            or not isinstance(item.get("records"), list)
            or not all(
                isinstance(record, CanonicalDraftStateRecord)
                for record in item["records"]
            )
            or (
                item.get("error") is not None and not isinstance(item.get("error"), str)
            )
        ):
            raise RuntimeError("invalid WORLD draft update receipt envelope")
        ranks.add(item["rank"])
        envelopes.append(item)
    if len(envelopes) != world_size or ranks != set(range(world_size)):
        raise RuntimeError("draft update receipt envelopes do not cover WORLD")
    return envelopes


def maybe_capture_draft_update_receipt(
    *,
    capture_draft_update_receipt: bool,
    decision: DraftUpdateDecisionLike | None,
    draft_update_successful: bool,
    shard_factory: Callable[[], list[CanonicalDraftStateRecord]],
    wrapper_visible: bool,
) -> dict[str, Any] | None:
    """WORLD-consensus canonical roots and choose one wrapper-visible publisher."""
    if (
        not capture_draft_update_receipt
        or decision is None
        or not decision.update_requested
        or not draft_update_successful
    ):
        return None

    error: str | None = None
    local_records: list[CanonicalDraftStateRecord] = []
    try:
        local_records = shard_factory()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    envelope: _ReceiptEnvelope = {
        "rank": rank,
        "records": local_records,
        "error": error,
        "wrapper_visible": bool(wrapper_visible),
    }
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered: list[_ReceiptEnvelope | None] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, envelope)
        envelopes = _validate_receipt_envelopes(gathered, world_size=world_size)
    else:
        envelopes = _validate_receipt_envelopes([envelope], world_size=1)

    errors = sorted(
        (int(item["rank"]), str(item["error"]))
        for item in envelopes
        if item["error"] is not None
    )
    if errors:
        details = "; ".join(f"rank {rank}: {message}" for rank, message in errors)
        raise RuntimeError(f"draft update receipt capture failed on WORLD: {details}")

    visible_ranks = sorted(
        int(item["rank"]) for item in envelopes if item["wrapper_visible"]
    )
    if not visible_ranks:
        raise RuntimeError("no wrapper-visible draft update receipt publisher")
    publisher_rank = visible_ranks[0]
    records = [record for item in envelopes for record in item["records"]]
    roots = canonical_draft_state_roots(records)
    return {
        "publisher_rank": publisher_rank,
        "receipt": (
            {
                "successful": True,
                "decision_id": int(decision.decision_id),
                "global_step": int(decision.global_step),
                "draft_model_sha256": roots.model_sha256,
                "draft_optimizer_sha256": roots.optimizer_sha256,
            }
            if rank == publisher_rank
            else None
        ),
    }


def select_published_draft_update_receipt(
    rows: Sequence[Mapping[str, Any]],
    *,
    capture_draft_update_receipt: bool,
    receipt_required: bool,
) -> dict[str, Any] | None:
    receipt_rows = [row for row in rows if row.get("draft_update_receipt") is not None]
    publisher_rows = [
        row
        for row in rows
        if row.get("draft_update_receipt_publisher_rank") is not None
    ]
    if not capture_draft_update_receipt:
        if receipt_rows or publisher_rows:
            raise RuntimeError("disabled receipt capture produced receipt metadata")
        return None
    if not receipt_required:
        if receipt_rows or publisher_rows:
            raise RuntimeError("skipped or failed draft update produced a receipt")
        return None
    publishers = {row.get("draft_update_receipt_publisher_rank") for row in rows}
    publishers.discard(None)
    if len(publishers) != 1:
        raise RuntimeError(
            "workers did not agree on one draft update receipt publisher"
        )
    publisher_rank = next(iter(publishers))
    published = [
        row["draft_update_receipt"]
        for row in rows
        if row.get("world_rank") == publisher_rank
        and row.get("draft_update_receipt") is not None
    ]
    if len(published) != 1:
        raise RuntimeError("expected exactly one wrapper-visible draft update receipt")
    receipt = published[0]
    if (
        not isinstance(receipt, dict)
        or receipt.get("successful") is not True
        or type(receipt.get("decision_id")) is not int
        or type(receipt.get("global_step")) is not int
        or any(
            not isinstance(receipt.get(key), str)
            or len(receipt[key]) != 64
            or any(character not in "0123456789abcdef" for character in receipt[key])
            for key in ("draft_model_sha256", "draft_optimizer_sha256")
        )
    ):
        raise RuntimeError("invalid published draft update receipt schema")
    return receipt
