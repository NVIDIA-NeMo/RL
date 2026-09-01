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

"""Ray-backed references for large routed-expert replay payloads.

The HTTP/Gym/replay-buffer path carries only small JSON tags. Each source-store
actor owns the full ``[tokens, layers, topk]`` arrays produced on its generation
node and returns only the logical ranges needed by a selected policy microbatch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

LOGGER = logging.getLogger(__name__)

ROUTED_EXPERTS_REF_SCHEMA = "nemo_rl.routed_experts_ref.v1"
ROUTED_EXPERTS_REF_KEY = "routed_experts"
ROUTED_EXPERTS_REF_TRANSPORT = "ray"
ROUTED_EXPERTS_REF_DTYPE = "int16"
ROUTED_EXPERTS_RAY_NAMESPACE = "nemo_rl_routed_experts"
# Shared router-replay wire contract: -1 means that no route was captured.
_MISSING_ROUTE_SENTINEL = -1

_REGISTRY_ACTOR_PREFIX = "nrl_routed_experts_registry_"
_STORE_ACTOR_PREFIX = "nrl_routed_experts_store_"
_MATERIALIZER_ACTOR_PREFIX = "nrl_routed_experts_materializer_"
# This cache only bridges the skew between TP/CP/PP siblings requesting the
# same DP-local microbatch. Target-weight-version retirement is the primary
# lifetime boundary; the entry limit is a backstop against unbounded growth.
_MATERIALIZER_CACHE_MAX_ENTRIES = 8


def _validate_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    if not all(character.isalnum() or character in "-_." for character in value):
        raise ValueError(f"{field} contains unsupported characters: {value!r}")
    return value


def registry_actor_name(run_instance_id: str) -> str:
    run_instance_id = _validate_identifier(run_instance_id, field="run_instance_id")
    return f"{_REGISTRY_ACTOR_PREFIX}{run_instance_id}"


def materializer_actor_name(run_instance_id: str, dp_rank: int) -> str:
    run_instance_id = _validate_identifier(run_instance_id, field="run_instance_id")
    if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
        raise ValueError(f"dp_rank must be a non-negative integer, got {dp_rank!r}")
    return f"{_MATERIALIZER_ACTOR_PREFIX}{run_instance_id}_dp_{dp_rank}"


def is_routed_experts_ref(value: Any) -> bool:
    return isinstance(value, dict) and value.get("schema") == ROUTED_EXPERTS_REF_SCHEMA


def routed_experts_ref_lookup_key(ref: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Return the insert-only actor-local identity for one route object."""
    return (
        int(ref["target_weight_version"]),
        int(ref["task_index"]),
        int(ref["rollout_index"]),
        str(ref["request_id"]),
    )


def validate_routed_experts_ref(value: Any) -> dict[str, Any]:
    if not is_routed_experts_ref(value):
        raise ValueError(
            "Expected a routed-experts Ray reference with schema "
            f"{ROUTED_EXPERTS_REF_SCHEMA!r}, got {value!r}"
        )
    ref = cast(dict[str, Any], value)
    if ref.get("key") != ROUTED_EXPERTS_REF_KEY:
        raise ValueError(
            f"routed-experts reference key must be {ROUTED_EXPERTS_REF_KEY!r}"
        )
    if ref.get("dtype") != ROUTED_EXPERTS_REF_DTYPE:
        raise ValueError(
            "routed-experts Ray references must use signed int16, got "
            f"{ref.get('dtype')!r}"
        )

    _validate_identifier(ref.get("store"), field="store")
    _validate_identifier(ref.get("store_instance_id"), field="store_instance_id")
    request_id = ref.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("routed-experts reference request_id must be non-empty")
    routed_experts_ref_lookup_key(ref)

    shape = ref.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        )
    ):
        raise ValueError(
            "routed-experts reference shape must be three non-negative integers, "
            f"got {shape!r}"
        )
    if shape[1] <= 0 or shape[2] <= 0:
        raise ValueError(
            f"routed-experts layer and top-k dimensions must be positive, got {shape!r}"
        )

    offset = ref.get("offset")
    length = ref.get("length")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length < 0
        or offset + length > shape[0]
    ):
        raise ValueError(
            "routed-experts reference slice is outside its source tensor: "
            f"offset={offset!r}, length={length!r}, shape={shape!r}"
        )
    return ref


def slice_routed_experts_ref(
    value: Mapping[str, Any], *, offset: int, length: int
) -> dict[str, Any]:
    """Create a zero-copy logical view tag into the same stored object."""
    ref = validate_routed_experts_ref(dict(value))
    sliced = dict(ref)
    sliced["offset"] = int(offset)
    sliced["length"] = int(length)
    return validate_routed_experts_ref(sliced)


MaterializationKey = str


@dataclass
class _NormalizedRoutedExpertsBatch:
    refs_by_sample: list[list[dict[str, Any]]]
    input_lengths: tuple[int, ...]
    batch_size: int
    padded_length: int
    layer_topk: tuple[int, int]
    target_weight_versions: tuple[int, ...]
    cache_key: MaterializationKey


@dataclass
class _RoutedExpertsRangePlacement:
    ref: dict[str, Any]
    sample_index: int
    destination_offset: int


@dataclass
class _RoutedExpertsRangeReadGroup:
    store_name: str
    store_instance_id: str
    placements: list[_RoutedExpertsRangePlacement]

    @property
    def requested_rows(self) -> int:
        return sum(int(placement.ref["length"]) for placement in self.placements)


_RANGE_READ_STAT_KEYS = (
    "range_read_requests",
    "range_read_store_calls",
    "range_read_source_objects",
    "range_read_segments",
    "range_read_rows",
    "range_read_bytes",
    "full_source_rows_equivalent",
    "full_source_bytes_equivalent",
    "range_read_avoided_bytes",
)


def _normalize_input_lengths(
    input_lengths: torch.Tensor | Sequence[int], *, batch_size: int
) -> tuple[int, ...]:
    if isinstance(input_lengths, torch.Tensor):
        if input_lengths.ndim != 1:
            raise ValueError(
                "input_lengths must be one-dimensional, got "
                f"shape={list(input_lengths.shape)}"
            )
        raw_lengths = input_lengths.detach().to(device="cpu").tolist()
    else:
        raw_lengths = list(input_lengths)
    if len(raw_lengths) != batch_size:
        raise ValueError(
            "input_lengths batch size does not match routed-experts references: "
            f"lengths={len(raw_lengths)}, batch={batch_size}"
        )
    lengths: list[int] = []
    for value in raw_lengths:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"input_lengths values must be integers, got {value!r}")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(
                f"input_lengths values must be non-negative, got {value!r}"
            )
        lengths.append(normalized)
    return tuple(lengths)


def _normalize_routed_experts_batch(
    refs_by_sample: Any,
    *,
    batch_size: int,
    padded_length: int,
    input_lengths: torch.Tensor | Sequence[int],
) -> _NormalizedRoutedExpertsBatch:
    if not isinstance(refs_by_sample, list):
        raise TypeError(
            "Ray-reference routed_experts must be a list with one entry per sample"
        )
    if len(refs_by_sample) != batch_size:
        raise ValueError(
            "routed-experts reference batch size does not match input_ids: "
            f"refs={len(refs_by_sample)}, batch={batch_size}"
        )
    if padded_length < 0:
        raise ValueError(f"padded_length must be non-negative, got {padded_length}")

    lengths = _normalize_input_lengths(input_lengths, batch_size=batch_size)
    normalized: list[list[dict[str, Any]]] = []
    layer_topk: tuple[int, int] | None = None
    target_weight_versions: set[int] = set()
    sample_keys: list[tuple[Any, ...]] = []
    for sample_index, raw_segments in enumerate(refs_by_sample):
        if is_routed_experts_ref(raw_segments):
            raw_segments = [raw_segments]
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(
                f"Sample {sample_index} has no routed-experts reference segments"
            )
        segments = [validate_routed_experts_ref(segment) for segment in raw_segments]
        segment_length = sum(int(segment["length"]) for segment in segments)
        expected_length = lengths[sample_index]
        if segment_length != expected_length:
            raise ValueError(
                "Routed-experts reference segments do not cover the sample's token "
                f"length: sample={sample_index}, segments={segment_length}, "
                f"input_length={expected_length}"
            )
        if expected_length > padded_length:
            raise ValueError(
                "Routed-experts sample length exceeds the padded output length: "
                f"sample={sample_index}, input_length={expected_length}, "
                f"padded_length={padded_length}"
            )

        segment_keys: list[tuple[Any, ...]] = []
        for segment in segments:
            this_layer_topk = (int(segment["shape"][1]), int(segment["shape"][2]))
            if layer_topk is None:
                layer_topk = this_layer_topk
            elif layer_topk != this_layer_topk:
                raise ValueError(
                    "Routed-experts references disagree on layer/top-k shape: "
                    f"expected={layer_topk}, got={this_layer_topk}"
                )
            lookup_key = routed_experts_ref_lookup_key(segment)
            target_weight_versions.add(lookup_key[0])
            segment_keys.append(
                (
                    str(segment["store"]),
                    str(segment["store_instance_id"]),
                    lookup_key,
                    int(segment["offset"]),
                    int(segment["length"]),
                    tuple(int(dim) for dim in segment["shape"]),
                    str(segment["dtype"]),
                )
            )
        normalized.append(segments)
        sample_keys.append(tuple(segment_keys))

    if layer_topk is None:
        raise ValueError("Cannot materialize an empty routed-experts reference batch")
    key_payload = (
        ROUTED_EXPERTS_REF_SCHEMA,
        batch_size,
        padded_length,
        lengths,
        tuple(sample_keys),
    )
    cache_key = hashlib.sha256(
        json.dumps(key_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _NormalizedRoutedExpertsBatch(
        refs_by_sample=normalized,
        input_lengths=lengths,
        batch_size=batch_size,
        padded_length=padded_length,
        layer_topk=layer_topk,
        target_weight_versions=tuple(sorted(target_weight_versions)),
        cache_key=cache_key,
    )


def _plan_routed_experts_range_reads(
    batch: _NormalizedRoutedExpertsBatch,
) -> tuple[list[_RoutedExpertsRangeReadGroup], dict[str, int]]:
    """Group logical route slices by source actor and quantify avoided reads."""
    groups_by_store: OrderedDict[tuple[str, str], _RoutedExpertsRangeReadGroup] = (
        OrderedDict()
    )
    source_shapes: dict[
        tuple[str, str, tuple[int, int, int, str]], tuple[int, int, int]
    ] = {}
    requested_rows = 0
    segment_count = 0

    for sample_index, segments in enumerate(batch.refs_by_sample):
        destination_offset = 0
        for segment in segments:
            length = int(segment["length"])
            if length > 0:
                source_identity = (
                    str(segment["store"]),
                    str(segment["store_instance_id"]),
                    routed_experts_ref_lookup_key(segment),
                )
                source_shape = tuple(int(dim) for dim in segment["shape"])
                existing_shape = source_shapes.setdefault(source_identity, source_shape)
                if existing_shape != source_shape:
                    raise ValueError(
                        "Routed-experts references sharing one source identity "
                        "disagree on shape: "
                        f"expected={existing_shape}, got={source_shape}"
                    )
                store_identity = source_identity[:2]
                group = groups_by_store.get(store_identity)
                if group is None:
                    group = _RoutedExpertsRangeReadGroup(
                        store_name=store_identity[0],
                        store_instance_id=store_identity[1],
                        placements=[],
                    )
                    groups_by_store[store_identity] = group
                group.placements.append(
                    _RoutedExpertsRangePlacement(
                        ref=segment,
                        sample_index=sample_index,
                        destination_offset=destination_offset,
                    )
                )
                requested_rows += length
                segment_count += 1
            destination_offset += length

    expected_rows = sum(batch.input_lengths)
    if requested_rows != expected_rows:
        raise RuntimeError(
            "Internal routed-experts range plan does not cover every valid row: "
            f"planned={requested_rows}, expected={expected_rows}"
        )

    bytes_per_row = (
        np.dtype(np.int16).itemsize * batch.layer_topk[0] * batch.layer_topk[1]
    )
    full_source_rows = sum(shape[0] for shape in source_shapes.values())
    range_read_bytes = requested_rows * bytes_per_row
    full_source_bytes = full_source_rows * bytes_per_row
    stats = {
        "range_read_requests": 1,
        "range_read_store_calls": len(groups_by_store),
        "range_read_source_objects": len(source_shapes),
        "range_read_segments": segment_count,
        "range_read_rows": requested_rows,
        "range_read_bytes": range_read_bytes,
        "full_source_rows_equivalent": full_source_rows,
        "full_source_bytes_equivalent": full_source_bytes,
        "range_read_avoided_bytes": full_source_bytes - range_read_bytes,
    }
    return list(groups_by_store.values()), stats


def routed_experts_materialization_key(
    refs_by_sample: Any,
    *,
    batch_size: int,
    padded_length: int,
    input_lengths: torch.Tensor | Sequence[int],
) -> MaterializationKey:
    """Build the exact DP-local composite identity without resolving payloads."""
    return _normalize_routed_experts_batch(
        refs_by_sample,
        batch_size=batch_size,
        padded_length=padded_length,
        input_lengths=input_lengths,
    ).cache_key


@dataclass
class _StoredRouteValue:
    value: np.ndarray
    nbytes: int
    shape: tuple[int, int, int]
    dtype: str


class RoutedExpertsStoreState:
    """Pure state machine used by the Ray actor and unit tests."""

    def __init__(self, store_instance_id: str):
        self.store_instance_id = _validate_identifier(
            store_instance_id, field="store_instance_id"
        )
        self._retired_through = -1
        self._entries: dict[tuple[int, int, int, str], _StoredRouteValue] = {}
        self._keys_by_target: dict[int, set[tuple[int, int, int, str]]] = {}

    def put(
        self,
        *,
        key: tuple[int, int, int, str],
        value: Any,
    ) -> None:
        target_weight_version = key[0]
        if target_weight_version <= self._retired_through:
            raise RuntimeError(
                "Cannot insert routed experts for a retired target-weight version: "
                f"target={target_weight_version}, "
                f"retired_through={self._retired_through}"
            )
        if key in self._entries:
            raise RuntimeError(
                "Duplicate routed-experts object key; inserts are never overwritten: "
                f"{key!r}"
            )
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(
                "Expected a three-dimensional routed-experts value, got "
                f"shape={array.shape!r}"
            )
        if array.dtype != np.int16:
            raise ValueError(f"Expected int16 routed experts, got {array.dtype!r}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        normalized_shape = tuple(int(dim) for dim in array.shape)
        self._entries[key] = _StoredRouteValue(
            value=array,
            nbytes=int(array.nbytes),
            shape=cast(tuple[int, int, int], normalized_shape),
            dtype=str(array.dtype),
        )
        self._keys_by_target.setdefault(target_weight_version, set()).add(key)

    def get(
        self, *, key: tuple[int, int, int, str], store_instance_id: str
    ) -> _StoredRouteValue:
        if store_instance_id != self.store_instance_id:
            raise RuntimeError(
                "Stale routed-experts store instance: "
                f"expected={self.store_instance_id!r}, got={store_instance_id!r}"
            )
        try:
            return self._entries[key]
        except KeyError as error:
            raise KeyError(f"Missing routed-experts object for key {key!r}") from error

    def get_ranges(self, refs: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Return exact, detached slices in input order from actor-owned arrays."""
        validated = [validate_routed_experts_ref(dict(ref)) for ref in refs]
        if not validated:
            raise ValueError(
                "A routed-experts range read must contain at least one ref"
            )

        layer_topk = (
            int(validated[0]["shape"][1]),
            int(validated[0]["shape"][2]),
        )
        total_rows = 0
        placements_by_key: dict[
            tuple[int, int, int, str], list[tuple[dict[str, Any], int]]
        ] = {}
        for ref in validated:
            this_layer_topk = (int(ref["shape"][1]), int(ref["shape"][2]))
            if this_layer_topk != layer_topk:
                raise ValueError(
                    "One routed-experts range read cannot mix layer/top-k shapes: "
                    f"expected={layer_topk}, got={this_layer_topk}"
                )
            key = routed_experts_ref_lookup_key(ref)
            placements_by_key.setdefault(key, []).append((ref, total_rows))
            total_rows += int(ref["length"])

        packed = np.empty((total_rows, *layer_topk), dtype=np.int16)
        for key, placements in placements_by_key.items():
            first_ref = placements[0][0]
            entry = self.get(
                key=key,
                store_instance_id=str(first_ref["store_instance_id"]),
            )
            if (
                list(entry.shape) != first_ref["shape"]
                or entry.dtype != first_ref["dtype"]
            ):
                raise RuntimeError(
                    "Stored routed-experts value does not match its tag: "
                    f"actual_shape={list(entry.shape)}, "
                    f"expected_shape={first_ref['shape']}, "
                    f"actual_dtype={entry.dtype}, expected_dtype={first_ref['dtype']}"
                )
            for ref, destination_offset in placements:
                if (
                    list(entry.shape) != ref["shape"]
                    or entry.dtype != ref["dtype"]
                    or str(ref["store_instance_id"]) != self.store_instance_id
                ):
                    raise RuntimeError(
                        "Routed-experts slices sharing one lookup key disagree on "
                        "source metadata"
                    )
                source_offset = int(ref["offset"])
                length = int(ref["length"])
                packed[destination_offset : destination_offset + length] = entry.value[
                    source_offset : source_offset + length
                ]
        return packed

    def retire_through(self, target_weight_version: int) -> dict[str, int]:
        target_weight_version = int(target_weight_version)
        self._retired_through = max(self._retired_through, target_weight_version)
        retired_objects = 0
        retired_bytes = 0
        for version in sorted(
            version
            for version in self._keys_by_target
            if version <= self._retired_through
        ):
            keys = self._keys_by_target.pop(version)
            for key in keys:
                entry = self._entries.pop(key, None)
                if entry is None:
                    continue
                retired_objects += 1
                retired_bytes += entry.nbytes
        return {
            "retired_through": self._retired_through,
            "retired_objects": retired_objects,
            "retired_bytes": retired_bytes,
            "remaining_objects": len(self._entries),
        }


@dataclass
class _MaterializedRouteRef:
    object_ref: Any
    nbytes: int
    shape: tuple[int, int, int, int]
    dtype: str
    target_weight_versions: tuple[int, ...]


class RoutedExpertsMaterializerState:
    """Bounded LRU metadata/cache state for one DP materializer actor."""

    def __init__(self, max_entries: int = _MATERIALIZER_CACHE_MAX_ENTRIES):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError(f"max_entries must be an integer, got {max_entries!r}")
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self.max_entries = max_entries
        self._retired_through = -1
        self._entries: OrderedDict[MaterializationKey, _MaterializedRouteRef] = (
            OrderedDict()
        )
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._materializations = 0
        self._materialized_bytes = 0
        self._evictions = 0
        self._range_read_stats = {key: 0 for key in _RANGE_READ_STAT_KEYS}

    def get(self, key: MaterializationKey) -> _MaterializedRouteRef | None:
        self._requests += 1
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        self._entries.move_to_end(key)
        return entry

    def put(
        self,
        *,
        key: MaterializationKey,
        object_ref: Any,
        nbytes: int,
        shape: Sequence[int],
        dtype: str,
        target_weight_versions: Sequence[int],
        range_read_stats: Mapping[str, int] | None = None,
    ) -> _MaterializedRouteRef:
        if key in self._entries:
            raise RuntimeError("Duplicate routed-experts materialization cache key")
        if range_read_stats is not None:
            unexpected_keys = set(range_read_stats) - set(_RANGE_READ_STAT_KEYS)
            if unexpected_keys:
                raise ValueError(
                    "Unexpected routed-experts range-read stats: "
                    f"{sorted(unexpected_keys)}"
                )
        normalized_shape = tuple(int(dim) for dim in shape)
        if len(normalized_shape) != 4:
            raise ValueError(
                f"Expected a four-dimensional materialized shape, got {shape!r}"
            )
        if dtype != ROUTED_EXPERTS_REF_DTYPE:
            raise ValueError(f"Expected int16 routed experts, got {dtype!r}")
        versions = tuple(sorted({int(version) for version in target_weight_versions}))
        if not versions:
            raise ValueError(
                "A routed-experts materialization must have a target version"
            )
        retired_versions = [
            version for version in versions if version <= self._retired_through
        ]
        if retired_versions:
            raise RuntimeError(
                "Cannot cache routed experts containing retired target-weight versions: "
                f"versions={retired_versions}, retired_through={self._retired_through}"
            )
        entry = _MaterializedRouteRef(
            object_ref=object_ref,
            nbytes=int(nbytes),
            shape=cast(tuple[int, int, int, int], normalized_shape),
            dtype=dtype,
            target_weight_versions=versions,
        )
        self._entries[key] = entry
        self._materializations += 1
        self._materialized_bytes += entry.nbytes
        if range_read_stats is not None:
            for stat_key in _RANGE_READ_STAT_KEYS:
                self._range_read_stats[stat_key] += int(
                    range_read_stats.get(stat_key, 0)
                )
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        return entry

    def retire_through(self, target_weight_version: int) -> dict[str, int]:
        target_weight_version = int(target_weight_version)
        self._retired_through = max(self._retired_through, target_weight_version)
        retired_materializations = 0
        retired_materialized_bytes = 0
        for key, entry in list(self._entries.items()):
            if max(entry.target_weight_versions) > self._retired_through:
                continue
            del self._entries[key]
            retired_materializations += 1
            retired_materialized_bytes += entry.nbytes
        return self.stats() | {
            "retired_through": self._retired_through,
            "retired_materializations": retired_materializations,
            "retired_materialized_bytes": retired_materialized_bytes,
        }

    def stats(self) -> dict[str, int]:
        return {
            "remaining_materializations": len(self._entries),
            "materialization_cache_requests": self._requests,
            "materialization_cache_hits": self._hits,
            "materialization_cache_misses": self._misses,
            "materializations": self._materializations,
            "materialized_bytes": self._materialized_bytes,
            "materialization_cache_evictions": self._evictions,
        } | self._range_read_stats


def _materialize_normalized_routed_experts(
    batch: _NormalizedRoutedExpertsBatch,
    *,
    resolver: Callable[[dict[str, Any]], Any],
) -> np.ndarray:
    dense = np.full(
        (
            batch.batch_size,
            batch.padded_length,
            batch.layer_topk[0],
            batch.layer_topk[1],
        ),
        _MISSING_ROUTE_SENTINEL,
        dtype=np.int16,
    )
    resolved: dict[tuple[str, str, tuple[int, int, int, str]], np.ndarray] = {}
    for sample_index, segments in enumerate(batch.refs_by_sample):
        destination_offset = 0
        for segment in segments:
            length = int(segment["length"])
            if length == 0:
                continue
            source_key = (
                str(segment["store"]),
                str(segment["store_instance_id"]),
                routed_experts_ref_lookup_key(segment),
            )
            source = resolved.get(source_key)
            if source is None:
                value = resolver(segment)
                if isinstance(value, torch.Tensor):
                    source = value.detach().to(device="cpu").numpy()
                else:
                    source = np.asarray(value)
                resolved[source_key] = source
            if source.dtype != np.int16 or list(source.shape) != segment["shape"]:
                raise RuntimeError(
                    "Resolved routed-experts object does not match its tag: "
                    f"actual_shape={list(source.shape)}, expected_shape={segment['shape']}, "
                    f"actual_dtype={source.dtype}, expected_dtype=int16"
                )
            source_offset = int(segment["offset"])
            dense[
                sample_index,
                destination_offset : destination_offset + length,
            ] = source[source_offset : source_offset + length]
            destination_offset += length
    return dense


@ray.remote(num_cpus=0)  # pragma: no cover - exercised in distributed jobs
class RoutedExpertsObjectStore:
    """Source-local owner/index actor for full routed-experts arrays."""

    def __init__(self, store_instance_id: str):
        self._state = RoutedExpertsStoreState(store_instance_id)
        self._lock = threading.Lock()

    def put_array(
        self,
        ref: dict[str, Any],
        value: np.ndarray,
    ) -> None:
        ref = validate_routed_experts_ref(ref)
        if ref["store_instance_id"] != self._state.store_instance_id:
            raise RuntimeError("Routed-experts tag targets a different store instance")
        array = np.asarray(value)
        if list(array.shape) != ref["shape"] or str(array.dtype) != ref["dtype"]:
            raise RuntimeError(
                "Inserted routed-experts value does not match its tag: "
                f"actual_shape={list(array.shape)}, expected_shape={ref['shape']}, "
                f"actual_dtype={array.dtype}, expected_dtype={ref['dtype']}"
            )
        with self._lock:
            self._state.put(
                key=routed_experts_ref_lookup_key(ref),
                value=array,
            )

    def get_ranges(self, refs: list[dict[str, Any]]) -> dict[str, Any]:
        """Copy only requested rows into one detached actor result."""
        with self._lock:
            packed = self._state.get_ranges(refs)
        return {
            "values": packed,
            "shape": list(packed.shape),
            "dtype": str(packed.dtype),
            "nbytes": int(packed.nbytes),
        }

    def retire_through(self, target_weight_version: int) -> dict[str, int]:
        with self._lock:
            return self._state.retire_through(target_weight_version)


@ray.remote(num_cpus=0)  # pragma: no cover - exercised in distributed jobs
class RoutedExpertsMaterializer:
    """Single-flight CPU assembler shared by all policy ranks in one DP shard."""

    def __init__(
        self,
        run_instance_id: str,
        dp_rank: int,
        max_entries: int = _MATERIALIZER_CACHE_MAX_ENTRIES,
    ):
        self.run_instance_id = _validate_identifier(
            run_instance_id, field="run_instance_id"
        )
        if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
            raise ValueError(f"dp_rank must be a non-negative integer, got {dp_rank!r}")
        self.dp_rank = dp_rank
        self._state = RoutedExpertsMaterializerState(max_entries=max_entries)

    def materialize(
        self,
        cache_key: MaterializationKey,
        refs_by_sample: Any,
        input_lengths: Sequence[int],
        batch_size: int,
        padded_length: int,
    ) -> dict[str, Any]:
        entry = self._state.get(cache_key)
        cache_hit = entry is not None
        if entry is None:
            batch = _normalize_routed_experts_batch(
                refs_by_sample,
                batch_size=int(batch_size),
                padded_length=int(padded_length),
                input_lengths=input_lengths,
            )
            if batch.cache_key != cache_key:
                raise RuntimeError(
                    "Policy worker and materializer computed different routed-experts "
                    "cache keys"
                )
            dense, range_read_stats = (
                _materialize_normalized_routed_experts_with_ray_ranges(batch)
            )
            # Keep the immutable composite alive in this actor's LRU. Returning
            # the ObjectRef nested in metadata lets every policy process ray.get
            # the same plasma object instead of serializing a fresh actor result.
            object_ref = ray.put(dense)
            entry = self._state.put(
                key=batch.cache_key,
                object_ref=object_ref,
                nbytes=int(dense.nbytes),
                shape=dense.shape,
                dtype=str(dense.dtype),
                target_weight_versions=batch.target_weight_versions,
                range_read_stats=range_read_stats,
            )
        return {
            "object_ref": entry.object_ref,
            "shape": list(entry.shape),
            "dtype": entry.dtype,
            "nbytes": entry.nbytes,
            "target_weight_versions": list(entry.target_weight_versions),
            "cache_hit": cache_hit,
        }

    def retire_through(self, target_weight_version: int) -> dict[str, int]:
        return self._state.retire_through(target_weight_version)

    def get_stats(self) -> dict[str, int]:
        return self._state.stats()


@ray.remote(num_cpus=0)  # pragma: no cover - exercised in distributed jobs
class RoutedExpertsStoreRegistry:
    def __init__(self, run_instance_id: str):
        self.run_instance_id = _validate_identifier(
            run_instance_id, field="run_instance_id"
        )
        self._stores: dict[str, Any] = {}
        self._materializers: dict[str, Any] = {}

    def register_store(self, store_instance_id: str, store: Any) -> None:
        store_instance_id = _validate_identifier(
            store_instance_id, field="store_instance_id"
        )
        existing = self._stores.get(store_instance_id)
        if existing is not None and existing != store:
            raise RuntimeError(
                f"Routed-experts store id {store_instance_id!r} was registered twice"
            )
        self._stores[store_instance_id] = store

    def register_materializer(self, actor_name: str, materializer: Any) -> None:
        actor_name = _validate_identifier(actor_name, field="actor_name")
        expected_prefix = f"{_MATERIALIZER_ACTOR_PREFIX}{self.run_instance_id}_dp_"
        if not actor_name.startswith(expected_prefix):
            raise ValueError(
                "Routed-experts materializer belongs to a different run: "
                f"actor={actor_name!r}, run_instance_id={self.run_instance_id!r}"
            )
        # The deterministic actor name plus get_if_exists makes registration
        # idempotent across all TP/CP/PP siblings. Keep the first equivalent
        # handle rather than depending on private Ray ActorHandle identity APIs.
        self._materializers.setdefault(actor_name, materializer)

    def retire_through(self, target_weight_version: int) -> dict[str, int]:
        store_futures = [
            store.retire_through.remote(target_weight_version)
            for store in self._stores.values()
        ]
        materializer_futures = [
            materializer.retire_through.remote(target_weight_version)
            for materializer in self._materializers.values()
        ]
        store_results = ray.get(store_futures)
        materializer_results = ray.get(materializer_futures)
        summary = {
            "retired_through": int(target_weight_version),
            "stores": len(store_results),
            "retired_objects": sum(
                result["retired_objects"] for result in store_results
            ),
            "retired_bytes": sum(result["retired_bytes"] for result in store_results),
            "remaining_objects": sum(
                result["remaining_objects"] for result in store_results
            ),
            "materializers": len(materializer_results),
            "retired_materializations": sum(
                result["retired_materializations"] for result in materializer_results
            ),
            "retired_materialized_bytes": sum(
                result["retired_materialized_bytes"] for result in materializer_results
            ),
            "remaining_materializations": sum(
                result["remaining_materializations"] for result in materializer_results
            ),
            "materialization_cache_requests": sum(
                result["materialization_cache_requests"]
                for result in materializer_results
            ),
            "materialization_cache_hits": sum(
                result["materialization_cache_hits"] for result in materializer_results
            ),
            "materialization_cache_misses": sum(
                result["materialization_cache_misses"]
                for result in materializer_results
            ),
            "materializations": sum(
                result["materializations"] for result in materializer_results
            ),
            "materialized_bytes": sum(
                result["materialized_bytes"] for result in materializer_results
            ),
            "materialization_cache_evictions": sum(
                result["materialization_cache_evictions"]
                for result in materializer_results
            ),
        }
        summary.update(
            {
                key: sum(result[key] for result in materializer_results)
                for key in _RANGE_READ_STAT_KEYS
            }
        )
        return summary


class RoutedExpertsStoreWriter:
    """Lazy per-vLLM-actor writer for full routed-experts arrays."""

    def __init__(self, run_instance_id: str):
        self.run_instance_id = _validate_identifier(
            run_instance_id, field="run_instance_id"
        )
        self.store_instance_id = uuid.uuid4().hex
        self.store_name = f"{_STORE_ACTOR_PREFIX}{self.store_instance_id}"
        node_id = ray.get_runtime_context().get_node_id()
        self.store = RoutedExpertsObjectStore.options(
            name=self.store_name,
            namespace=ROUTED_EXPERTS_RAY_NAMESPACE,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            ),
        ).remote(self.store_instance_id)
        self.registry = RoutedExpertsStoreRegistry.options(
            name=registry_actor_name(self.run_instance_id),
            namespace=ROUTED_EXPERTS_RAY_NAMESPACE,
            get_if_exists=True,
        ).remote(self.run_instance_id)
        ray.get(self.registry.register_store.remote(self.store_instance_id, self.store))

    def put(
        self,
        routed_experts: torch.Tensor,
        *,
        request_id: str,
        task_index: int,
        rollout_index: int,
        target_weight_version: int,
    ) -> dict[str, Any]:
        array = (
            routed_experts.detach()
            .to(device="cpu", dtype=torch.int16)
            .contiguous()
            .numpy()
        )
        if array.ndim != 3:
            raise ValueError(
                "Stored routed experts must have shape [tokens, layers, topk], "
                f"got {array.shape}"
            )
        ref = validate_routed_experts_ref(
            {
                "schema": ROUTED_EXPERTS_REF_SCHEMA,
                "store": self.store_name,
                "store_instance_id": self.store_instance_id,
                "request_id": str(request_id),
                "key": ROUTED_EXPERTS_REF_KEY,
                "task_index": int(task_index),
                "rollout_index": int(rollout_index),
                "target_weight_version": int(target_weight_version),
                "offset": 0,
                "length": int(array.shape[0]),
                "shape": [int(dim) for dim in array.shape],
                "dtype": ROUTED_EXPERTS_REF_DTYPE,
            }
        )
        # The source actor retains the NumPy value and performs range slicing on
        # its generation node. Ray may use shared memory internally for this
        # actor argument, but no full-payload ObjectRef is exposed to policy.
        ray.get(self.store.put_array.remote(ref, array))
        return ref


_STORE_HANDLE_CACHE: dict[str, Any] = {}
_MATERIALIZER_HANDLE_CACHE: dict[tuple[str, int], Any] = {}


def _get_routed_experts_store(store_name: str) -> Any:
    store = _STORE_HANDLE_CACHE.get(store_name)
    if store is None:
        store = ray.get_actor(store_name, namespace=ROUTED_EXPERTS_RAY_NAMESPACE)
        _STORE_HANDLE_CACHE[store_name] = store
    return store


def _assemble_routed_experts_range_results(
    batch: _NormalizedRoutedExpertsBatch,
    groups: Sequence[_RoutedExpertsRangeReadGroup],
    range_results: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if len(groups) != len(range_results):
        raise RuntimeError(
            "Routed-experts range result count does not match the read plan: "
            f"groups={len(groups)}, results={len(range_results)}"
        )

    dense = np.full(
        (
            batch.batch_size,
            batch.padded_length,
            batch.layer_topk[0],
            batch.layer_topk[1],
        ),
        _MISSING_ROUTE_SENTINEL,
        dtype=np.int16,
    )
    for group, result in zip(groups, range_results):
        values = np.asarray(result.get("values"))
        expected_shape = (
            group.requested_rows,
            batch.layer_topk[0],
            batch.layer_topk[1],
        )
        if (
            tuple(values.shape) != expected_shape
            or values.dtype != np.int16
            or result.get("shape") != list(expected_shape)
            or result.get("dtype") != ROUTED_EXPERTS_REF_DTYPE
            or result.get("nbytes") != int(values.nbytes)
        ):
            returned_metadata = {
                "shape": result.get("shape"),
                "dtype": result.get("dtype"),
                "nbytes": result.get("nbytes"),
            }
            raise RuntimeError(
                "Source actor returned an invalid routed-experts range result: "
                f"actual_shape={list(values.shape)}, "
                f"expected_shape={list(expected_shape)}, "
                f"actual_dtype={values.dtype}, metadata={returned_metadata}"
            )

        source_offset = 0
        for placement in group.placements:
            length = int(placement.ref["length"])
            dense[
                placement.sample_index,
                placement.destination_offset : placement.destination_offset + length,
            ] = values[source_offset : source_offset + length]
            source_offset += length
        if source_offset != group.requested_rows:
            raise RuntimeError(
                "Routed-experts range scatter did not consume its full source result: "
                f"consumed={source_offset}, available={group.requested_rows}"
            )
    return dense


def _materialize_normalized_routed_experts_with_ray_ranges(
    batch: _NormalizedRoutedExpertsBatch,
) -> tuple[np.ndarray, dict[str, int]]:
    groups, range_read_stats = _plan_routed_experts_range_reads(batch)
    futures = []
    for group in groups:
        store = _get_routed_experts_store(group.store_name)
        futures.append(
            store.get_ranges.remote([placement.ref for placement in group.placements])
        )
    range_results = ray.get(futures)
    return (
        _assemble_routed_experts_range_results(batch, groups, range_results),
        range_read_stats,
    )


def materialize_routed_experts_refs(
    refs_by_sample: Any,
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    resolver: Callable[[dict[str, Any]], Any] | None = None,
) -> torch.Tensor:
    """Resolve one selected policy microbatch into a dense CPU tensor.

    Production calls use source-local Ray range reads. Supplying ``resolver``
    explicitly retains the whole-object path for pure tests and diagnostics.
    """
    batch_size, padded_length = input_ids.shape[:2]
    batch = _normalize_routed_experts_batch(
        refs_by_sample,
        batch_size=batch_size,
        padded_length=padded_length,
        input_lengths=input_lengths,
    )
    if resolver is None:
        dense, _ = _materialize_normalized_routed_experts_with_ray_ranges(batch)
    else:
        dense = _materialize_normalized_routed_experts(batch, resolver=resolver)
    return torch.from_numpy(dense)


def _get_or_create_routed_experts_materializer(
    run_instance_id: str, dp_rank: int
) -> Any:
    run_instance_id = _validate_identifier(run_instance_id, field="run_instance_id")
    actor_name = materializer_actor_name(run_instance_id, dp_rank)
    cache_key = (run_instance_id, dp_rank)
    materializer = _MATERIALIZER_HANDLE_CACHE.get(cache_key)
    if materializer is not None:
        return materializer

    node_id = ray.get_runtime_context().get_node_id()
    materializer = RoutedExpertsMaterializer.options(
        name=actor_name,
        namespace=ROUTED_EXPERTS_RAY_NAMESPACE,
        get_if_exists=True,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
    ).remote(run_instance_id, dp_rank)
    registry = RoutedExpertsStoreRegistry.options(
        name=registry_actor_name(run_instance_id),
        namespace=ROUTED_EXPERTS_RAY_NAMESPACE,
        get_if_exists=True,
    ).remote(run_instance_id)
    ray.get(registry.register_materializer.remote(actor_name, materializer))
    _MATERIALIZER_HANDLE_CACHE[cache_key] = materializer
    return materializer


def materialize_routed_experts_refs_once_per_dp(
    refs_by_sample: Any,
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    run_instance_id: str,
    dp_rank: int,
) -> torch.Tensor:
    """Resolve and assemble once, then share the immutable CPU result per DP."""
    batch_size, padded_length = input_ids.shape[:2]
    batch = _normalize_routed_experts_batch(
        refs_by_sample,
        batch_size=batch_size,
        padded_length=padded_length,
        input_lengths=input_lengths,
    )
    materializer = _get_or_create_routed_experts_materializer(run_instance_id, dp_rank)
    metadata = ray.get(
        materializer.materialize.remote(
            batch.cache_key,
            refs_by_sample,
            batch.input_lengths,
            batch_size,
            padded_length,
        )
    )
    object_or_ref = metadata["object_ref"]
    value = (
        ray.get(object_or_ref)
        if isinstance(object_or_ref, ray.ObjectRef)
        else object_or_ref
    )
    array = np.asarray(value)
    if (
        list(array.shape) != metadata["shape"]
        or str(array.dtype) != metadata["dtype"]
        or int(array.nbytes) != metadata["nbytes"]
    ):
        raise RuntimeError(
            "Shared routed-experts materialization does not match its metadata: "
            f"actual_shape={list(array.shape)}, expected_shape={metadata['shape']}, "
            f"actual_dtype={array.dtype}, expected_dtype={metadata['dtype']}, "
            f"actual_nbytes={array.nbytes}, expected_nbytes={metadata['nbytes']}"
        )
    # Ray plasma NumPy views are intentionally immutable. PyTorch warns when a
    # read-only view is wrapped even though this tensor is only read before the
    # existing H2D copy. Suppress that narrowly without making a CPU copy.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        return torch.from_numpy(array)


def retire_routed_experts_through(
    policy_config: Mapping[str, Any], target_weight_version: int
) -> dict[str, int] | None:
    """Retire route objects consumed by a completed optimizer step."""
    router_replay = policy_config.get("router_replay") or {}
    if router_replay.get("transport", "inline") != ROUTED_EXPERTS_REF_TRANSPORT:
        return None
    run_instance_id = router_replay.get("_store_run_instance_id")
    if not isinstance(run_instance_id, str) or not run_instance_id:
        raise RuntimeError(
            "router_replay.transport=ray is missing its store run instance id"
        )
    registry = ray.get_actor(
        registry_actor_name(run_instance_id),
        namespace=ROUTED_EXPERTS_RAY_NAMESPACE,
    )
    result = cast(
        dict[str, int],
        ray.get(registry.retire_through.remote(int(target_weight_version))),
    )
    LOGGER.info("Retired routed-experts Ray objects: %s", result)
    return result
