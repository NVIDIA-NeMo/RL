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

"""Ray-object references for large routed-expert replay payloads.

The HTTP/Gym/replay-buffer path carries only small JSON tags.  The full
``[tokens, layers, topk]`` array stays in Ray's object store until a Megatron
policy worker resolves the references in its already-selected microbatch.
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
class _StoredRouteRef:
    object_ref: Any
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
        self._entries: dict[tuple[int, int, int, str], _StoredRouteRef] = {}
        self._keys_by_target: dict[int, set[tuple[int, int, int, str]]] = {}

    def put(
        self,
        *,
        key: tuple[int, int, int, str],
        object_ref: Any,
        nbytes: int,
        shape: Sequence[int],
        dtype: str,
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
        normalized_shape = tuple(int(dim) for dim in shape)
        if len(normalized_shape) != 3:
            raise ValueError(f"Expected a three-dimensional shape, got {shape!r}")
        if dtype != ROUTED_EXPERTS_REF_DTYPE:
            raise ValueError(f"Expected int16 routed experts, got {dtype!r}")
        self._entries[key] = _StoredRouteRef(
            object_ref=object_ref,
            nbytes=int(nbytes),
            shape=cast(tuple[int, int, int], normalized_shape),
            dtype=dtype,
        )
        self._keys_by_target.setdefault(target_weight_version, set()).add(key)

    def get(
        self, *, key: tuple[int, int, int, str], store_instance_id: str
    ) -> _StoredRouteRef:
        if store_instance_id != self.store_instance_id:
            raise RuntimeError(
                "Stale routed-experts store instance: "
                f"expected={self.store_instance_id!r}, got={store_instance_id!r}"
            )
        try:
            return self._entries[key]
        except KeyError as error:
            raise KeyError(f"Missing routed-experts object for key {key!r}") from error

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
    ) -> _MaterializedRouteRef:
        if key in self._entries:
            raise RuntimeError("Duplicate routed-experts materialization cache key")
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
        }


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
    """Ownership/index actor; payload bytes remain in Ray's object store."""

    def __init__(self, store_instance_id: str):
        self._state = RoutedExpertsStoreState(store_instance_id)
        self._lock = threading.Lock()

    def put_ref(
        self,
        ref: dict[str, Any],
        boxed_object_ref: list[Any],
        nbytes: int,
    ) -> None:
        ref = validate_routed_experts_ref(ref)
        if ref["store_instance_id"] != self._state.store_instance_id:
            raise RuntimeError("Routed-experts tag targets a different store instance")
        if len(boxed_object_ref) != 1:
            raise ValueError("boxed_object_ref must contain exactly one Ray ObjectRef")
        with self._lock:
            self._state.put(
                key=routed_experts_ref_lookup_key(ref),
                object_ref=boxed_object_ref[0],
                nbytes=nbytes,
                shape=ref["shape"],
                dtype=ref["dtype"],
            )

    def get_ref(self, ref: dict[str, Any]) -> dict[str, Any]:
        ref = validate_routed_experts_ref(ref)
        with self._lock:
            entry = self._state.get(
                key=routed_experts_ref_lookup_key(ref),
                store_instance_id=ref["store_instance_id"],
            )
            # Keep the ObjectRef nested so Ray does not resolve the large array
            # in this metadata actor. The requesting policy worker performs the
            # only ray.get() that materializes the payload.
            return {
                "object_ref": entry.object_ref,
                "shape": list(entry.shape),
                "dtype": entry.dtype,
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
            dense = _materialize_normalized_routed_experts(
                batch,
                resolver=_resolve_routed_experts_ref_with_ray,
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
        return {
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
        object_ref = ray.put(array)
        # A nested ObjectRef is passed by reference instead of being eagerly
        # resolved as a top-level Ray task argument.
        ray.get(self.store.put_ref.remote(ref, [object_ref], int(array.nbytes)))
        return ref


_STORE_HANDLE_CACHE: dict[str, Any] = {}
_MATERIALIZER_HANDLE_CACHE: dict[tuple[str, int], Any] = {}


def _resolve_routed_experts_ref_with_ray(ref: dict[str, Any]) -> np.ndarray:
    store_name = str(ref["store"])
    store = _STORE_HANDLE_CACHE.get(store_name)
    if store is None:
        store = ray.get_actor(store_name, namespace=ROUTED_EXPERTS_RAY_NAMESPACE)
        _STORE_HANDLE_CACHE[store_name] = store
    metadata = ray.get(store.get_ref.remote(ref))
    object_or_ref = metadata["object_ref"]
    value = (
        ray.get(object_or_ref)
        if isinstance(object_or_ref, ray.ObjectRef)
        else object_or_ref
    )
    array = np.asarray(value)
    if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
        raise RuntimeError(
            "Routed-experts object metadata changed between insertion and lookup: "
            f"actual_shape={list(array.shape)}, expected_shape={metadata['shape']}, "
            f"actual_dtype={array.dtype}, expected_dtype={metadata['dtype']}"
        )
    return array


def materialize_routed_experts_refs(
    refs_by_sample: Any,
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    resolver: Callable[[dict[str, Any]], Any] = _resolve_routed_experts_ref_with_ray,
) -> torch.Tensor:
    """Resolve one selected policy microbatch into a dense CPU tensor."""
    batch_size, padded_length = input_ids.shape[:2]
    batch = _normalize_routed_experts_batch(
        refs_by_sample,
        batch_size=batch_size,
        padded_length=padded_length,
        input_lengths=input_lengths,
    )
    return torch.from_numpy(
        _materialize_normalized_routed_experts(batch, resolver=resolver)
    )


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
