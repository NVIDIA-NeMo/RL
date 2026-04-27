# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections.abc import Mapping
from typing import Any, Optional

import torch

from nemo_rl.data.multimodal_utils import PackedTensor

_VLLM_MM_KEYS = frozenset({"vllm_content", "vllm_images", "vllm_videos"})
_ADDITIONAL_MM_KEYS = frozenset({"token_type_ids", "imgs_sizes"})


def _compact_payload_bytes(compact: dict) -> int:
    """Estimate bytes for a vllm_mm_compact_payload dict (unique strings only)."""
    total = 0
    for s in compact.get("unique_contents", []):
        total += len(s.encode("utf-8")) if isinstance(s, str) else 0
    for s in compact.get("unique_images", []):
        total += len(s.encode("utf-8")) if isinstance(s, str) else 0
    return total


def _estimate_non_tensor_bytes(value: Any) -> int:
    """Estimate payload bytes for non-tensor multimodal fields."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(_estimate_non_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        # Count both keys and values to approximate serialized payload size.
        return sum(
            _estimate_non_tensor_bytes(k) + _estimate_non_tensor_bytes(v)
            for k, v in value.items()
        )
    return 0


def _tensor_nbytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.numel()


def _infer_logical_rows(data: Mapping[str, Any]) -> int:
    if "input_ids" in data and torch.is_tensor(data["input_ids"]):
        return int(data["input_ids"].shape[0])
    if not data:
        return 0
    first_key = next(iter(data))
    first_val = data[first_key]
    if torch.is_tensor(first_val):
        return int(first_val.shape[0])
    if isinstance(first_val, PackedTensor):
        return len(first_val)
    if isinstance(first_val, list):
        return len(first_val)
    return 0


def infer_unique_prompt_count(
    data: Mapping[str, Any], default_rows: Optional[int] = None
) -> int:
    """Infer unique prompt count using prompt-tracking keys when available."""
    if "_dedup_prompt_idx" in data:
        dedup_prompt_idx = data["_dedup_prompt_idx"]
        if torch.is_tensor(dedup_prompt_idx):
            return int(torch.unique(dedup_prompt_idx).numel())
        if isinstance(dedup_prompt_idx, list):
            return len(set(dedup_prompt_idx))

    if "idx" in data:
        idx = data["idx"]
        if torch.is_tensor(idx):
            return int(torch.unique(idx).numel())
        if isinstance(idx, list):
            return len(set(idx))

    if default_rows is not None:
        return int(default_rows)
    return _infer_logical_rows(data)


def collect_multimodal_payload_metrics(
    data: Mapping[str, Any],
    boundary: str,
    *,
    logical_rows: Optional[int] = None,
    unique_prompts: Optional[int] = None,
    total_input_tokens: Optional[int] = None,
) -> dict[str, float | int]:
    """Collect consistent multimodal payload metrics for a boundary."""
    tensor_mm_bytes = 0
    non_tensor_mm_bytes = 0
    unique_mm_items = 0

    for key, value in data.items():
        if isinstance(value, PackedTensor):
            tensor_mm_bytes += sum(_tensor_nbytes(t) for t in value.tensors)
            unique_mm_items += len(value.tensors)
            continue

        if key in _ADDITIONAL_MM_KEYS:
            if torch.is_tensor(value):
                tensor_mm_bytes += _tensor_nbytes(value)
            else:
                non_tensor_mm_bytes += _estimate_non_tensor_bytes(value)
            continue

        if key in _VLLM_MM_KEYS:
            non_tensor_mm_bytes += _estimate_non_tensor_bytes(value)

        if key == "vllm_mm_compact_payload" and isinstance(value, dict):
            non_tensor_mm_bytes += _compact_payload_bytes(value)

    rows = _infer_logical_rows(data) if logical_rows is None else int(logical_rows)
    prompts = (
        infer_unique_prompt_count(data, default_rows=rows)
        if unique_prompts is None
        else int(unique_prompts)
    )
    prompts = max(prompts, 1) if rows > 0 else 0
    total_mm = tensor_mm_bytes + non_tensor_mm_bytes
    ratio = float(rows) / float(prompts) if prompts > 0 else 0.0

    metrics: dict[str, float | int] = {
        f"payload_bytes/{boundary}/tensor_mm": int(tensor_mm_bytes),
        f"payload_bytes/{boundary}/non_tensor_mm": int(non_tensor_mm_bytes),
        f"payload_bytes/{boundary}/total_mm": int(total_mm),
        f"payload_counts/{boundary}/logical_rows": int(rows),
        f"payload_counts/{boundary}/unique_prompts": int(prompts),
        f"payload_counts/{boundary}/unique_mm_items": int(unique_mm_items),
        f"payload_ratio/{boundary}/logical_to_unique": ratio,
    }
    if total_input_tokens is not None:
        metrics[f"payload_counts/{boundary}/total_input_tokens"] = int(total_input_tokens)
    return metrics


def print_multimodal_payload_metrics(
    metrics: Mapping[str, float | int], enabled: bool = True
) -> None:
    """Print multimodal payload metrics in a stable key=value format."""
    if not enabled:
        return
    formatted_items = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_items.append(f"{key}={value:.6f}")
        else:
            formatted_items.append(f"{key}={value}")
    print("▶ [PAYLOAD] " + ", ".join(formatted_items), flush=True)
