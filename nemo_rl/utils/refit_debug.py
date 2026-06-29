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

import hashlib
import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


REFIT_DEBUG_PREFIX = "[REFIT_DEBUG]"
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def refit_debug_enabled() -> bool:
    """Return whether opt-in refit diagnostics are enabled for this process."""
    return os.environ.get("NRL_REFIT_DEBUG", "").strip().lower() in _TRUTHY_VALUES


def refit_debug_rank() -> str:
    """Return the distributed rank without requiring process-group setup."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return str(torch.distributed.get_rank())
    return os.environ.get("RANK", "unknown")


def _matches_embedding(name: str) -> bool:
    return name.endswith(
        ("embed_tokens.weight", "tok_embeddings.weight", "word_embeddings.weight")
    )


def _matches_lm_head(name: str) -> bool:
    return name.endswith("lm_head.weight")


def _matches_attention(name: str) -> bool:
    return name.endswith(
        (
            "q_proj.weight",
            "qkv_proj.weight",
            "q_a_proj.weight",
            "kv_a_proj_with_mqa.weight",
            "indexer.weight",
        )
    ) or ".indexer." in name


def _matches_router_gate(name: str) -> bool:
    return name.endswith(".gate.weight")


def _matches_routing_bias(name: str) -> bool:
    return "e_score_correction_bias" in name


def _matches_expert(name: str) -> bool:
    return ".experts." in name and name.endswith("weight")


_REPRESENTATIVE_MATCHERS = (
    ("embedding", _matches_embedding),
    ("lm_head", _matches_lm_head),
    ("attention", _matches_attention),
    ("router_gate", _matches_router_gate),
    ("routing_bias", _matches_routing_bias),
    ("expert", _matches_expert),
)


def select_refit_debug_names(names: Iterable[str]) -> dict[str, str]:
    """Select at most one deterministic parameter name for each debug category."""
    sorted_names = sorted(set(names))
    selected: dict[str, str] = {}
    for category, matcher in _REPRESENTATIVE_MATCHERS:
        match = next((name for name in sorted_names if matcher(name)), None)
        if match is not None:
            selected[category] = match
    return selected


def tensor_fingerprint(tensor: torch.Tensor, max_samples: int = 16) -> str:
    """Return a bounded fingerprint without copying the complete tensor to CPU."""
    try:
        flat = tensor.detach().reshape(-1)
        numel = flat.numel()
        if numel == 0:
            return "samples=0 digest=empty finite=0 nonfinite=0"
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")

        sample_count = min(numel, max_samples)
        if sample_count == 1:
            indices = torch.zeros(1, dtype=torch.long, device=flat.device)
        else:
            indices = torch.arange(
                sample_count, dtype=torch.long, device=flat.device
            )
            indices = indices * (numel - 1) // (sample_count - 1)

        sample = flat.index_select(0, indices).to(
            device="cpu", dtype=torch.float32
        )
        digest = hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:16]
        finite_mask = torch.isfinite(sample)
        finite_count = int(finite_mask.sum().item())
        nonfinite_count = sample_count - finite_count
        finite_values = sample[finite_mask]
        if finite_count:
            sample_sum = float(finite_values.sum(dtype=torch.float64).item())
            sample_absmax = float(finite_values.abs().max().item())
        else:
            sample_sum = float("nan")
            sample_absmax = float("nan")
        return (
            f"samples={sample_count} digest={digest} finite={finite_count} "
            f"nonfinite={nonfinite_count} sample_sum={sample_sum:.9g} "
            f"sample_absmax={sample_absmax:.9g}"
        )
    except Exception as exc:
        return f"fingerprint_error={type(exc).__name__}:{exc}"


@dataclass
class RefitDebugStats:
    """Aggregate counts without retaining refit tensor references."""

    parameter_count: int = 0
    total_bytes: int = 0
    dtype_counts: Counter[str] = field(default_factory=Counter)
    dtype_bytes: Counter[str] = field(default_factory=Counter)
    loaded_names: set[str] = field(default_factory=set)
    load_result_seen: bool = False

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_names)

    def _observe(self, dtype: torch.dtype, size_bytes: int) -> None:
        dtype_name = str(dtype)
        self.parameter_count += 1
        self.total_bytes += size_bytes
        self.dtype_counts[dtype_name] += 1
        self.dtype_bytes[dtype_name] += size_bytes

    def observe_tensor(self, name: str, tensor: torch.Tensor) -> None:
        del name
        self._observe(tensor.dtype, tensor.numel() * tensor.element_size())

    def observe_metadata(
        self, name: str, shape: Iterable[int], dtype: torch.dtype
    ) -> None:
        del name
        self._observe(dtype, math.prod(shape) * dtype.itemsize)

    def observe_loaded(self, names: Iterable[str] | None) -> None:
        if names is None:
            return
        self.load_result_seen = True
        self.loaded_names.update(names)

    def format(self) -> str:
        dtype_summary = ";".join(
            f"{dtype}:count={self.dtype_counts[dtype]},bytes={self.dtype_bytes[dtype]}"
            for dtype in sorted(self.dtype_counts)
        )
        if not dtype_summary:
            dtype_summary = "none"
        loaded = str(self.loaded_count) if self.load_result_seen else "unavailable"
        return (
            f"parameters={self.parameter_count} total_bytes={self.total_bytes} "
            f"dtypes={dtype_summary} loaded={loaded}"
        )


def debug_refit_tensors(
    iterator: Iterator[tuple[str, torch.Tensor]],
    *,
    phase: str,
    selected_names: dict[str, str],
    rank: str,
    stats: RefitDebugStats,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Observe and log tensors while preserving generator values and order."""
    name_to_category = {name: category for category, name in selected_names.items()}
    enabled = refit_debug_enabled()
    for name, tensor in iterator:
        stats.observe_tensor(name, tensor)
        category = name_to_category.get(name)
        if enabled and category is not None:
            print(
                f"{REFIT_DEBUG_PREFIX} phase={phase} rank={rank} "
                f"category={category} name={name} shape={list(tensor.shape)} "
                f"dtype={tensor.dtype} device={tensor.device} "
                f"{tensor_fingerprint(tensor)}",
                flush=True,
            )
        yield name, tensor


def log_refit_destinations(
    model: nn.Module,
    selected_names: dict[str, str],
    *,
    rank: str,
) -> None:
    """Log exact destination parameters and label packed/mapped names explicitly."""
    if not refit_debug_enabled():
        return
    destination_parameters: dict[str, Any] = dict(model.named_parameters())
    for category, name in selected_names.items():
        tensor = destination_parameters.get(name)
        if tensor is None:
            print(
                f"{REFIT_DEBUG_PREFIX} phase=vllm_destination rank={rank} "
                f"category={category} source_name={name} "
                "status=mapped_or_unresolved",
                flush=True,
            )
            continue
        print(
            f"{REFIT_DEBUG_PREFIX} phase=vllm_destination rank={rank} "
            f"category={category} source_name={name} destination_name={name} "
            f"status=exact shape={list(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} {tensor_fingerprint(tensor)}",
            flush=True,
        )
