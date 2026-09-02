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
"""Strict, vLLM-free configuration for per-token NVFP4 rollout."""

import re
from collections.abc import Iterable
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

NVFP4_PERTOKEN_ZMQ_TIMEOUT_MS = 600_000
# Keep driver-side boundary normalization aligned with the defaults on MCore's
# TransformerConfig.  The trainer later reads the effective values directly
# from its instantiated model config.
MCORE_DEFAULT_NUM_LAYERS_AT_START_IN_BF16 = 1
MCORE_DEFAULT_NUM_LAYERS_AT_END_IN_BF16 = 1
# Ordinary linears are BF16 by construction in NvFp4PerTokenConfig. Only
# semantic BF16 decoder-layer boundaries belong in ModelOpt's ignore list.
DEFAULT_NVFP4_IGNORE: list[str] = []

_FULL_EXPERT_LAYER_IGNORE_RE = re.compile(r"^\*\.layers\.(\d+)\.mlp\.experts\*$")
# vLLM module prefixes for a decoder layer's routed experts always carry the
# layer index as ``layers.<index>``; the container name below it differs per
# model family (``mlp.experts``, ``block_sparse_moe.experts``, ...).
_MODULE_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def boundary_layer_indices(patterns: Iterable[str]) -> set[int]:
    """Recover the decoder-layer indices encoded in boundary ignore patterns."""
    indices: set[int] = set()
    for pattern in patterns:
        match = _FULL_EXPERT_LAYER_IGNORE_RE.fullmatch(pattern)
        if match is None:
            raise ValueError(
                "NVFP4 boundary verification requires complete expert-layer "
                f"ignore patterns; cannot read a layer index from {pattern!r}"
            )
        indices.add(int(match.group(1)))
    return indices


def module_layer_index(module_name: str) -> int:
    """Return the decoder-layer index for a routed-expert module path."""
    match = _MODULE_LAYER_INDEX_RE.search(module_name)
    if match is None:
        raise ValueError(
            "NVFP4 boundary verification requires a 'layers.<index>' segment in "
            f"the routed-expert module path, got {module_name!r}"
        )
    return int(match.group(1))


def _require_full_expert_layer_ignores(value: list[str]) -> list[str]:
    invalid = [
        pattern
        for pattern in value
        if _FULL_EXPERT_LAYER_IGNORE_RE.fullmatch(pattern) is None
    ]
    if invalid:
        raise ValueError(
            "additional_ignore may only exclude complete expert layers using "
            "'*.layers.<index>.mlp.experts*'; invalid patterns: "
            f"{invalid}"
        )
    return value


def resolve_boundary_ignore_patterns(
    *,
    num_hidden_layers: int,
    first_last_layers_bf16: bool,
    num_layers_at_start_in_bf16: int,
    num_layers_at_end_in_bf16: int,
    expected_additional_ignore: list[str] | None = None,
) -> list[str]:
    """Resolve semantic Megatron BF16 boundaries to rollout layer patterns.

    The Megatron boundary is the only source of truth for rollout precision.
    When ``expected_additional_ignore`` is supplied it is cross-checked against
    the derived set rather than establishing a boundary of its own, so a config
    that never reached driver-side normalization fails here instead of running
    with a rollout precision boundary the trainer does not share.
    """
    counts = (num_layers_at_start_in_bf16, num_layers_at_end_in_bf16)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError(
            "NVFP4 BF16 boundary layer counts must be non-negative integers"
        )
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError(
            "NVFP4 BF16 boundary resolution requires num_hidden_layers > 0"
        )
    if not first_last_layers_bf16:
        # MCore keeps non-zero count defaults even while the feature is off.
        # They become effective only when first_last_layers_bf16 is enabled.
        indices: set[int] = set()
    else:
        if sum(counts) > num_hidden_layers:
            raise ValueError(
                "NVFP4 BF16 boundary counts exceed the model's decoder layer count"
            )
        indices = set(range(num_layers_at_start_in_bf16))
        indices.update(
            range(num_hidden_layers - num_layers_at_end_in_bf16, num_hidden_layers)
        )
    resolved = [f"*.layers.{index}.mlp.experts*" for index in sorted(indices)]

    if expected_additional_ignore is not None:
        _require_full_expert_layer_ignores(expected_additional_ignore)
        if set(expected_additional_ignore) != set(resolved):
            raise ValueError(
                "generation.nvfp4_pertoken_rollout.additional_ignore does not "
                "match the effective Megatron first/last BF16 layer boundary. "
                f"Expected {resolved}, got {expected_additional_ignore}. An "
                "empty value usually means normalize_nvfp4_pertoken_policy_config "
                "did not run for this entry point."
            )
    return resolved


class NvFp4PerTokenRolloutConfig(BaseModel, extra="forbid"):
    """User configuration for the constrained per-token NVFP4 rollout."""

    enabled: bool = False
    additional_ignore: Annotated[
        list[str], AfterValidator(_require_full_expert_layer_ignores)
    ] = Field(default_factory=list)

    def resolved_ignore(self) -> list[str]:
        return [*DEFAULT_NVFP4_IGNORE, *self.additional_ignore]
