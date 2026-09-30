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
"""Shared route-plan executor for direct and deferred router replay.

One implementation turns a verified :class:`RouteAssemblyPlan` plus fetched
:class:`RouteFragment` payloads into a training route tensor. The finalizer
runs it eagerly in direct mode (a failure rejects the rollout before
publication as ``route_assembly:<reason>``); the policy worker runs it after
canonical publication in deferred mode (the same reason maps to the existing
counted sentinel fallback). The executor owns per-span extras-digest
verification against Gym's receipt-bound commitments, span classification via
Gym's decision table, full/tail slicing, sentinel fill, and shape checks — it
returns a tensor or a failure reason and never decides policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from nemo_rl.data_plane.schema import (
    ROUTE_ENCODING_ENVELOPE,
    ROUTE_ENCODING_LIST,
    ROUTED_EXPERTS_FIELD,
)
from nemo_rl.experience.route_plan import RouteAssemblyPlan

# Gym's router-replay missing-route wire sentinel (== nemo_gym
# MISSING_ROUTE_SENTINEL, restated so this module imports without the
# optional nemo_gym extra).
ROUTE_MISSING_SENTINEL = -1

# The shared failure-code vocabulary. Direct mode surfaces these as
# pre-publication ``route_assembly:<reason>`` rejections; deferred mode maps
# them onto the existing counted-fallback metrics.
ROUTE_FAILURE_CANONICAL_LENGTH = "canonical_length_mismatch"
ROUTE_FAILURE_MISSING_FRAGMENT = "missing_fragment"
ROUTE_FAILURE_INTEGRITY = "fragment_integrity"
ROUTE_FAILURE_RANK = "fragment_rank"
ROUTE_FAILURE_LENGTH = "fragment_length"
ROUTE_FAILURE_MODEL_SHAPE = "fragment_model_shape"
ROUTE_FAILURE_ASSEMBLED_LENGTH = "assembled_length_mismatch"


@dataclass(frozen=True)
class RouteLayout:
    """Which columns of a staged route fragment belong to the trained MoE layers.

    A fragment's layer axis comes in one of two layouts. MInf stages one
    column per MoE layer (``num_moe_layers`` columns). vLLM stages one column
    per transformer layer (``total_num_layers`` columns), so for a hybrid
    Mamba/attention/MoE stack the executor must pick out the MoE columns by
    ``moe_layer_numbers``. Both counts coincide for a stack where every layer
    is MoE, in which case no selection happens.
    """

    moe_layer_numbers: tuple[int, ...]
    """1-based global layer numbers of the MoE layers, ascending."""
    total_num_layers: int
    top_k: int

    def __post_init__(self) -> None:
        if not self.moe_layer_numbers or self.top_k <= 0:
            raise ValueError(
                "route layout requires at least one MoE layer and top_k > 0, got "
                f"moe_layer_numbers={self.moe_layer_numbers}, top_k={self.top_k}"
            )
        if self.total_num_layers < len(self.moe_layer_numbers) or any(
            not 1 <= n <= self.total_num_layers for n in self.moe_layer_numbers
        ):
            raise ValueError(
                f"MoE layer numbers {self.moe_layer_numbers} do not fit in "
                f"{self.total_num_layers} layers"
            )

    @classmethod
    def compressed(cls, *, num_moe_layers: int, top_k: int) -> "RouteLayout":
        """A MoE-only layout for callers that know only ``(num_moe_layers, top_k)``.

        Direct mode learns these from the fetched fragments themselves and has
        no model config, so it can never need column selection.
        """
        return cls(
            moe_layer_numbers=tuple(range(1, num_moe_layers + 1)),
            total_num_layers=num_moe_layers,
            top_k=top_k,
        )

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_numbers)

    @property
    def dims(self) -> tuple[int, int]:
        """The published ``(num_moe_layers, top_k)`` route tensor dims."""
        return self.num_moe_layers, self.top_k

    def moe_column_indices(self, num_payload_layers: int) -> Optional[tuple[int, ...]]:
        """Payload column for each MoE layer, in ``moe_layer_numbers`` order.

        This is the one rule both replay paths share. A MoE-only payload
        (MInf) maps the i-th MoE layer to column i; a full-layer payload
        (vLLM) maps layer number ``n`` to column ``n - 1``. Any other width
        is unsupported and returns None.
        """
        if num_payload_layers == self.num_moe_layers:
            return tuple(range(self.num_moe_layers))
        if num_payload_layers == self.total_num_layers:
            return tuple(n - 1 for n in self.moe_layer_numbers)
        return None

    def select_moe_columns(self, routes: torch.Tensor) -> Optional[torch.Tensor]:
        """Return ``routes`` as ``[T, num_moe_layers, top_k]`` or None if it fits neither layout."""
        if int(routes.shape[2]) != self.top_k:
            return None
        columns = self.moe_column_indices(int(routes.shape[1]))
        if columns is None:
            return None
        if len(columns) == int(routes.shape[1]):
            return routes
        return routes[:, list(columns), :]


@dataclass(frozen=True)
class RouteFragment:
    """One staged route payload plus the metadata its extras digest binds.

    ``routes`` is the staged ``[staged_len, num_moe_layers, topk]`` tensor,
    ``encoding`` the ``ROUTE_ENCODING_*`` wire code the digest was committed
    over, and ``extras_metadata_json`` the canonical non-route extras JSON
    staged beside it.
    """

    routes: torch.Tensor
    encoding: int
    extras_metadata_json: bytes


def verify_route_fragment_integrity(
    fragment: RouteFragment,
    *,
    extras_digest_version: int,
    expected_extras_digest: str,
) -> bool:
    """Rebuild the staged extras envelope and verify the receipt-bound digest."""
    # Deferred: nemo_gym is an optional extra absent in non-gym runs.
    from nemo_gym.token_id_capture.staging.digest import (
        EXTRAS_DIGEST_VERSION,
        compute_extras_digest,
    )

    from nemo_rl.utils.routed_experts_codec import encode_routed_experts

    if extras_digest_version != EXTRAS_DIGEST_VERSION:
        return False
    try:
        decoded = json.loads(fragment.extras_metadata_json.decode("utf-8"))
        if decoded is None:
            extras: dict[str, Any] = {}
        elif isinstance(decoded, dict):
            extras = decoded
        else:
            return False
        if fragment.encoding == ROUTE_ENCODING_ENVELOPE:
            extras[ROUTED_EXPERTS_FIELD] = encode_routed_experts(fragment.routes)
        elif fragment.encoding == ROUTE_ENCODING_LIST:
            extras[ROUTED_EXPERTS_FIELD] = fragment.routes.tolist()
        else:
            return False
        return compute_extras_digest(extras) == expected_extras_digest
    except (TypeError, ValueError):
        return False


def execute_route_plan(
    plan: RouteAssemblyPlan,
    fragments: Mapping[str, RouteFragment],
    *,
    layout: RouteLayout,
    canonical_len: int,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Assemble one canonical route tensor from staged fragments.

    Args:
        plan: The verified assembly plan built by the finalizer.
        fragments: Fetched fragments keyed by staging key. Sentinel spans
            need no entry.
        layout: Which fragment columns are the trained MoE layers. The
            policy worker supplies the real model layout (the authoritative
            shape check, accepting both MoE-only and full-layer fragments);
            the direct-mode finalizer supplies a compressed layout learned
            from the fetched fragments.
        canonical_len: The published row's token length.

    Returns:
        ``(tensor, None)`` on success — ``[canonical_len, num_moe_layers,
        topk]`` int16, sentinel-filled wherever no fragment contributed — or
        ``(None, reason)`` with a reason from the shared vocabulary.
    """
    # Deferred: nemo_gym is an optional extra absent in non-gym runs.
    from nemo_gym.token_id_capture.staging.routes import classify_route_span

    if canonical_len != plan.expected_token_length:
        return None, ROUTE_FAILURE_CANONICAL_LENGTH
    num_moe_layers, top_k = layout.dims
    routed = torch.full(
        (canonical_len, num_moe_layers, top_k),
        ROUTE_MISSING_SENTINEL,
        dtype=torch.int16,
    )
    position = 0
    for span in plan.spans:
        contribution = span.carry_len + span.generation_len
        mode = classify_route_span(
            carry_len=span.carry_len,
            generation_len=span.generation_len,
            staged_route_len=span.staged_route_len,
        )
        if mode != "sentinel":
            fragment = fragments.get(span.staging_key)
            if fragment is None:
                return None, ROUTE_FAILURE_MISSING_FRAGMENT
            if not verify_route_fragment_integrity(
                fragment,
                extras_digest_version=span.extras_digest_version,
                expected_extras_digest=span.extras_digest,
            ):
                return None, ROUTE_FAILURE_INTEGRITY
            routes = fragment.routes
            if routes.dim() != 3:
                return None, ROUTE_FAILURE_RANK
            if int(routes.shape[0]) != span.staged_route_len:
                return None, ROUTE_FAILURE_LENGTH
            routes = layout.select_moe_columns(routes)
            if routes is None:
                return None, ROUTE_FAILURE_MODEL_SHAPE
            if mode == "full":
                routed[position : position + contribution] = routes.to(torch.int16)
            else:
                tail_start = position + span.carry_len
                routed[tail_start : position + contribution] = routes[
                    -span.generation_len :
                ].to(torch.int16)
        position += contribution
    if plan.spans and position != canonical_len:
        return None, ROUTE_FAILURE_ASSEMBLED_LENGTH
    return routed, None
