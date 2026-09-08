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

from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

import numpy as np
import ray.cloudpickle as cloudpickle
import torch
from PIL import Image

from nemo_rl.data.multimodal_utils import (
    MULTIMODAL_CONTENT_TYPES,
    NATIVE_MULTIMODAL_KEYS,
    PackedTensor,
)

_PENDING_PAYLOAD_METRICS: dict[str, int | float] = {}
_PENDING_PAYLOAD_METRICS_LOCK = Lock()


def _tensor_nbytes(value: torch.Tensor | None) -> int:
    if value is None:
        return 0
    return value.numel() * value.element_size()


def _value_nbytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate data bytes, optionally counting shared leaves only once."""
    if value is None:
        return 0
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        object_id = id(value)
        if seen is not None:
            if object_id in seen:
                return 0
            seen.add(object_id)
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        return len(value)
    if torch.is_tensor(value):
        object_id = id(value)
        if seen is not None:
            if object_id in seen:
                return 0
            seen.add(object_id)
        return _tensor_nbytes(value)
    if isinstance(value, np.ndarray):
        object_id = id(value)
        if seen is not None:
            if object_id in seen:
                return 0
            seen.add(object_id)
        return value.nbytes
    if isinstance(value, Image.Image):
        object_id = id(value)
        if seen is not None:
            if object_id in seen:
                return 0
            seen.add(object_id)
        return len(value.tobytes())
    if isinstance(value, Mapping):
        return sum(_value_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_nbytes(item, seen) for item in value)
    return 0


def _value_segment_count(value: Any, seen: set[int] | None = None) -> int:
    """Count media leaves, optionally counting shared objects only once."""
    if value is None:
        return 0
    if isinstance(
        value,
        (str, bytes, bytearray, memoryview, torch.Tensor, np.ndarray, Image.Image),
    ):
        object_id = id(value)
        if seen is not None:
            if object_id in seen:
                return 0
            seen.add(object_id)
        return 1
    if isinstance(value, Mapping):
        return sum(_value_segment_count(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_segment_count(item, seen) for item in value)
    return 0


def _typed_content_media_nbytes(
    value: Any,
    seen: set[int] | None = None,
) -> int:
    """Count media embedded in typed vLLM content without counting prompt text."""
    if isinstance(value, Mapping):
        if value.get("type") in MULTIMODAL_CONTENT_TYPES:
            return sum(
                _value_nbytes(item, seen)
                for key, item in value.items()
                if key != "type"
            )
        return sum(_typed_content_media_nbytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_typed_content_media_nbytes(item, seen) for item in value)
    return 0


def _typed_content_media_segment_count(
    value: Any,
    seen: set[int] | None = None,
) -> int:
    """Count media leaves embedded in typed content without counting text."""
    if isinstance(value, Mapping):
        if value.get("type") in MULTIMODAL_CONTENT_TYPES:
            return sum(
                _value_segment_count(item, seen)
                for key, item in value.items()
                if key != "type"
            )
        return sum(
            _typed_content_media_segment_count(item, seen) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_typed_content_media_segment_count(item, seen) for item in value)
    return 0


def protocol5_serialized_nbytes(value: Any) -> int:
    """Return cloudpickle protocol-5 frame plus out-of-band buffer bytes."""
    buffers = []
    frame = cloudpickle.dumps(value, protocol=5, buffer_callback=buffers.append)
    buffer_bytes = 0
    for buffer in buffers:
        raw = buffer.raw() if hasattr(buffer, "raw") else memoryview(buffer)
        buffer_bytes += raw.nbytes
    return len(frame) + buffer_bytes


def collect_multimodal_payload_metrics(
    data: Any,
    boundary: str,
    *,
    enabled: bool,
) -> dict[str, int | float]:
    """Measure one exact Ray argument without scanning when disabled."""
    if not enabled:
        return {}

    totals = {
        "physical_media_bytes": 0,
        "logical_media_bytes": 0,
        "physical_segments": 0,
        "logical_segments": 0,
    }
    per_key: dict[str, int] = {}
    seen_native_leaves: set[int] = set()
    seen_native_segments: set[int] = set()
    seen_packed_leaves: set[int] = set()
    seen_packed_leaves_by_key: dict[str, set[int]] = {}

    def visit(key: str, value: Any) -> None:
        if isinstance(value, PackedTensor):
            key_seen = seen_packed_leaves_by_key.setdefault(key, set())
            key_physical_segments = 0
            key_physical = 0
            for item in value.tensors:
                if item is None:
                    continue
                object_id = id(item)
                if object_id not in key_seen:
                    key_seen.add(object_id)
                    key_physical_segments += 1
                if object_id not in seen_packed_leaves:
                    seen_packed_leaves.add(object_id)
                    key_physical += _tensor_nbytes(item)

            logical_items = [
                item for item in value.iter_logical_segments() if item is not None
            ]
            key_logical = sum(_tensor_nbytes(item) for item in logical_items)
            totals["physical_media_bytes"] += key_physical
            totals["logical_media_bytes"] += key_logical
            totals["physical_segments"] += key_physical_segments
            totals["logical_segments"] += len(logical_items)
            physical_key = f"payload_counts/{boundary}/{key}/physical_segments"
            logical_key = f"payload_counts/{boundary}/{key}/logical_segments"
            per_key[physical_key] = per_key.get(physical_key, 0) + key_physical_segments
            per_key[logical_key] = per_key.get(logical_key, 0) + len(logical_items)
        elif key == "vllm_content":
            totals["physical_media_bytes"] += _typed_content_media_nbytes(
                value, seen_native_leaves
            )
            totals["logical_media_bytes"] += _typed_content_media_nbytes(value)
            totals["physical_segments"] += _typed_content_media_segment_count(
                value, seen_native_segments
            )
            totals["logical_segments"] += _typed_content_media_segment_count(value)
        elif key in NATIVE_MULTIMODAL_KEYS:
            totals["physical_media_bytes"] += _value_nbytes(value, seen_native_leaves)
            totals["logical_media_bytes"] += _value_nbytes(value)
            totals["physical_segments"] += _value_segment_count(
                value, seen_native_segments
            )
            totals["logical_segments"] += _value_segment_count(value)
        elif (
            isinstance(value, Mapping) and value.get("type") in MULTIMODAL_CONTENT_TYPES
        ):
            totals["physical_media_bytes"] += _typed_content_media_nbytes(
                value, seen_native_leaves
            )
            totals["logical_media_bytes"] += _typed_content_media_nbytes(value)
            totals["physical_segments"] += _typed_content_media_segment_count(
                value, seen_native_segments
            )
            totals["logical_segments"] += _typed_content_media_segment_count(value)
        elif isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                visit(str(nested_key), nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                visit(key, nested_value)

    if isinstance(data, Mapping):
        for key, value in data.items():
            visit(str(key), value)
    else:
        visit("root", data)

    physical_media_bytes = totals["physical_media_bytes"]
    logical_media_bytes = totals["logical_media_bytes"]
    saved_bytes = max(logical_media_bytes - physical_media_bytes, 0)
    ratio = (
        float(physical_media_bytes) / float(logical_media_bytes)
        if logical_media_bytes
        else 1.0
    )
    return {
        f"payload_bytes/{boundary}/serialized": protocol5_serialized_nbytes(data),
        f"payload_bytes/{boundary}/physical_media": physical_media_bytes,
        f"payload_bytes/{boundary}/logical_media": logical_media_bytes,
        f"payload_bytes/{boundary}/estimated_saved": saved_bytes,
        f"payload_counts/{boundary}/physical_segments": totals["physical_segments"],
        f"payload_counts/{boundary}/logical_segments": totals["logical_segments"],
        f"payload_counts/{boundary}/calls": 1,
        f"payload_ratio/{boundary}/physical_to_logical": ratio,
        **per_key,
    }


def collect_sharded_multimodal_payload_metrics(
    shards: Sequence[Mapping[str, Any]],
    boundary: str,
    *,
    enabled: bool,
) -> dict[str, int | float]:
    """Aggregate metrics over the exact unique per-DP-shard Ray arguments."""
    if not enabled:
        return {}

    per_shard = [
        collect_multimodal_payload_metrics(
            shard,
            f"{boundary}/shard_{index}",
            enabled=True,
        )
        for index, shard in enumerate(shards)
    ]
    serialized = [
        int(metrics[f"payload_bytes/{boundary}/shard_{index}/serialized"])
        for index, metrics in enumerate(per_shard)
    ]
    physical = [
        int(metrics[f"payload_bytes/{boundary}/shard_{index}/physical_media"])
        for index, metrics in enumerate(per_shard)
    ]
    logical = [
        int(metrics[f"payload_bytes/{boundary}/shard_{index}/logical_media"])
        for index, metrics in enumerate(per_shard)
    ]
    physical_segments = [
        int(metrics[f"payload_counts/{boundary}/shard_{index}/physical_segments"])
        for index, metrics in enumerate(per_shard)
    ]
    logical_segments = [
        int(metrics[f"payload_counts/{boundary}/shard_{index}/logical_segments"])
        for index, metrics in enumerate(per_shard)
    ]
    total_logical_bytes = sum(logical)
    return {
        f"payload_bytes/{boundary}/serialized_total": sum(serialized),
        f"payload_bytes/{boundary}/serialized_max_shard": max(serialized, default=0),
        f"payload_bytes/{boundary}/physical_media_total": sum(physical),
        f"payload_bytes/{boundary}/logical_media_total": total_logical_bytes,
        f"payload_bytes/{boundary}/estimated_saved_total": max(
            total_logical_bytes - sum(physical), 0
        ),
        f"payload_counts/{boundary}/physical_segments_total": sum(physical_segments),
        f"payload_counts/{boundary}/logical_segments_total": sum(logical_segments),
        f"payload_counts/{boundary}/shards": len(shards),
        f"payload_counts/{boundary}/calls": len(shards),
        f"payload_ratio/{boundary}/physical_to_logical": (
            float(sum(physical)) / float(total_logical_bytes)
            if total_logical_bytes
            else 1.0
        ),
    }


def merge_multimodal_payload_metrics(
    metric_sets: Sequence[Mapping[str, int | float]],
) -> dict[str, int | float]:
    """Aggregate payload measurements collected during one logging interval.

    Byte and segment counts are summed because repeated calls represent distinct
    Ray transfers. Per-call maxima and shard counts retain their maximum value,
    and physical-to-logical ratios are recomputed from the aggregated byte
    totals instead of averaging ratios.
    """
    merged: dict[str, int | float] = {}
    ratio_keys: set[str] = set()
    boundaries: set[str] = set()
    for metrics in metric_sets:
        for key, value in metrics.items():
            if key.startswith("payload_ratio/") and key.endswith(
                "/physical_to_logical"
            ):
                ratio_keys.add(key)
                continue
            if key.startswith("payload_bytes/") and key.endswith(
                "/serialized_mean_per_call"
            ):
                continue
            if key.startswith("payload_counts/") and key.endswith("/calls"):
                boundaries.add(
                    key.removeprefix("payload_counts/").removesuffix("/calls")
                )
            if key.endswith("/serialized_max_shard") or key.endswith("/shards"):
                merged[key] = max(merged.get(key, 0), value)
            else:
                merged[key] = merged.get(key, 0) + value

    for ratio_key in ratio_keys:
        boundary = ratio_key.removeprefix("payload_ratio/").removesuffix(
            "/physical_to_logical"
        )
        physical_key = f"payload_bytes/{boundary}/physical_media"
        logical_key = f"payload_bytes/{boundary}/logical_media"
        if physical_key not in merged and logical_key not in merged:
            physical_key += "_total"
            logical_key += "_total"
        physical = merged.get(physical_key, 0)
        logical = merged.get(logical_key, 0)
        merged[ratio_key] = float(physical) / float(logical) if logical else 1.0

    for boundary in boundaries:
        calls = merged[f"payload_counts/{boundary}/calls"]
        serialized_key = f"payload_bytes/{boundary}/serialized"
        if serialized_key not in merged:
            serialized_key += "_total"
        if serialized_key in merged:
            merged[f"payload_bytes/{boundary}/serialized_mean_per_call"] = (
                float(merged[serialized_key]) / float(calls) if calls else 0.0
            )

    return merged


def drain_multimodal_payload_metrics() -> dict[str, int | float]:
    """Return and clear payload measurements recorded in this process."""
    with _PENDING_PAYLOAD_METRICS_LOCK:
        pending = dict(_PENDING_PAYLOAD_METRICS)
        _PENDING_PAYLOAD_METRICS.clear()
    return merge_multimodal_payload_metrics([pending])


def print_multimodal_payload_metrics(
    metrics: Mapping[str, int | float],
) -> None:
    """Record metrics for the logger and print a stable, scrapeable line."""
    if not metrics:
        return
    with _PENDING_PAYLOAD_METRICS_LOCK:
        merged = merge_multimodal_payload_metrics([_PENDING_PAYLOAD_METRICS, metrics])
        _PENDING_PAYLOAD_METRICS.clear()
        _PENDING_PAYLOAD_METRICS.update(merged)
    values = []
    for key, value in sorted(metrics.items()):
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        values.append(f"{key}={rendered}")
    print("▶ [PAYLOAD] " + ", ".join(values), flush=True)
