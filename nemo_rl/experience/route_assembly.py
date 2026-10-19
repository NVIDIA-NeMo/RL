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
also applies authenticated newest conditioning routes while preserving the
routes that predicted each call's generated targets. It returns a tensor or
a failure reason and never decides policy.
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
    ROUTED_EXPERTS_PREFIX_LEN_FIELD,
)
from nemo_rl.experience.route_plan import RouteAssemblyPlan
from nemo_rl.utils.routed_experts_codec import decode_routed_experts

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
ROUTE_FAILURE_PREFIX_BOUNDARY = "prefix_boundary_mismatch"
ROUTE_FAILURE_PREFIX_REPLACEMENT = "prefix_replacement_mismatch"


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


@dataclass(frozen=True)
class CallRouteView:
    """An authenticated prefix before any later call replaces its routes.

    The prefix includes all conditioning tokens. Only targets in
    ``[generation_start, generation_end)`` belong to this call; earlier
    generated tokens remain context and belong to their own views.
    """

    staging_key: str
    generation_start: int
    generation_end: int
    routed_experts: torch.Tensor


@dataclass(frozen=True)
class RoutePlanExecution:
    """Canonical predictor routes and optional independent original call prefixes."""

    routed_experts: torch.Tensor
    call_views: tuple[CallRouteView, ...]


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
    dims: tuple[int, int],
    canonical_len: int,
    backpatch_predecessor_tail: bool = False,
) -> tuple[Optional[torch.Tensor], Optional[str]]:
    """Assemble one canonical route tensor from staged fragments.

    Args:
        plan: The verified assembly plan built by the finalizer.
        fragments: Fetched fragments keyed by staging key. Sentinel spans
            need no entry.
        dims: Model-owned ``(num_moe_layers, topk)``. The policy worker
            supplies real model dims (the authoritative shape check); the
            direct-mode finalizer supplies dims learned from the fetched
            fragments.
        canonical_len: The published row's token length.
        backpatch_predecessor_tail: CC requires complete routes and the
            committed continuation route for the preceding final token. Each
            plan is one segment, so no patch can cross a compaction root.

    Returns:
        ``(tensor, None)`` on success — ``[canonical_len, num_moe_layers,
        topk]`` int16, sentinel-filled wherever no fragment contributed — or
        ``(None, reason)`` with a reason from the shared vocabulary.
    """
    execution, failure = _execute_route_plan(
        plan,
        fragments,
        dims=dims,
        canonical_len=canonical_len,
        backpatch_predecessor_tail=backpatch_predecessor_tail,
        capture_call_views=False,
    )
    return (execution.routed_experts if execution is not None else None), failure


def execute_route_plan_with_call_views(
    plan: RouteAssemblyPlan,
    fragments: Mapping[str, RouteFragment],
    *,
    dims: tuple[int, int],
    canonical_len: int,
    backpatch_predecessor_tail: bool,
) -> tuple[Optional[RoutePlanExecution], Optional[str]]:
    """Keep each call's routes before subsequent prefix/tail replacement.

    Verification and canonical assembly are identical to ``execute_route_plan``.
    All fragments must verify before any views are returned. Each view owns a
    copy, so later continuation backpatches cannot change earlier evidence.
    This does not change training rows, masks, normalization, or model routing.
    """
    return _execute_route_plan(
        plan,
        fragments,
        dims=dims,
        canonical_len=canonical_len,
        backpatch_predecessor_tail=backpatch_predecessor_tail,
        capture_call_views=True,
    )


def _execute_route_plan(
    plan: RouteAssemblyPlan,
    fragments: Mapping[str, RouteFragment],
    *,
    dims: tuple[int, int],
    canonical_len: int,
    backpatch_predecessor_tail: bool,
    capture_call_views: bool,
) -> tuple[Optional[RoutePlanExecution], Optional[str]]:
    # Deferred: nemo_gym is an optional extra absent in non-gym runs.
    from nemo_gym.token_id_capture.staging.routes import classify_route_span

    if canonical_len != plan.expected_token_length:
        return None, ROUTE_FAILURE_CANONICAL_LENGTH
    num_moe_layers, top_k = dims
    routed = torch.full(
        (canonical_len, num_moe_layers, top_k),
        ROUTE_MISSING_SENTINEL,
        dtype=torch.int16,
    )
    protected_predictors = torch.zeros(canonical_len, dtype=torch.bool)
    # Diagnostic views must retain each actual prefill, independently of the
    # canonical tensor's custody of earlier generation predictors.
    call_context = routed.clone() if capture_call_views else None
    position = 0
    call_views: list[CallRouteView] = []
    for span in plan.spans:
        contribution = span.carry_len + span.generation_len
        # A digest-covered prefix marker extends Gym's legacy full/tail span
        # modes: the fragment contains [newest prefix, current delta]. We need
        # its metadata before classifying its route length.
        fragment = fragments.get(span.staging_key)
        metadata: dict[str, Any] = {}
        prefix_len: Any = None
        if fragment is not None:
            if not verify_route_fragment_integrity(
                fragment,
                extras_digest_version=span.extras_digest_version,
                expected_extras_digest=span.extras_digest,
            ):
                return None, ROUTE_FAILURE_INTEGRITY
            try:
                decoded_metadata = json.loads(
                    fragment.extras_metadata_json.decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None, ROUTE_FAILURE_INTEGRITY
            if decoded_metadata is not None:
                if not isinstance(decoded_metadata, dict):
                    return None, ROUTE_FAILURE_INTEGRITY
                metadata = decoded_metadata
            prefix_len = metadata.get(ROUTED_EXPERTS_PREFIX_LEN_FIELD)
        prefix_replacement = prefix_len is not None
        if prefix_replacement:
            if (
                type(prefix_len) is not int
                or prefix_len != position
                or span.staged_route_len != prefix_len + contribution
            ):
                return None, ROUTE_FAILURE_PREFIX_REPLACEMENT
            mode = "full"
        else:
            mode = classify_route_span(
                carry_len=span.carry_len,
                generation_len=span.generation_len,
                staged_route_len=span.staged_route_len,
            )
        if backpatch_predecessor_tail and mode != "full":
            return None, ROUTE_FAILURE_MISSING_FRAGMENT
        if mode != "sentinel":
            if fragment is None:
                return None, ROUTE_FAILURE_MISSING_FRAGMENT
            routes = fragment.routes
            if routes.dim() != 3:
                return None, ROUTE_FAILURE_RANK
            if int(routes.shape[0]) != span.staged_route_len:
                return None, ROUTE_FAILURE_LENGTH
            if tuple(routes.shape[1:]) != (num_moe_layers, top_k):
                return None, ROUTE_FAILURE_MODEL_SHAPE
            if prefix_replacement:
                mutable_prefix = ~protected_predictors[:position]
                routed[:position][mutable_prefix] = routes[:position][
                    mutable_prefix
                ].to(torch.int16)
                if call_context is not None:
                    call_context[:position] = routes[:position].to(torch.int16)
            boundary = metadata.get("routed_experts_prefix_boundary")
            if boundary is not None:
                if position == 0:
                    return None, ROUTE_FAILURE_PREFIX_BOUNDARY
                try:
                    previous_route = decode_routed_experts(boundary, torch.int16)
                except (TypeError, ValueError):
                    return None, ROUTE_FAILURE_PREFIX_BOUNDARY
                if previous_route.shape != (1, num_moe_layers, top_k):
                    return None, ROUTE_FAILURE_PREFIX_BOUNDARY
                routed[position - 1] = previous_route[0]
                if call_context is not None:
                    call_context[position - 1] = previous_route[0]
            if backpatch_predecessor_tail:
                # Integrity was checked above over this exact metadata too.
                tail = metadata.get("predecessor_tail_route")
                if position == 0:
                    if tail is not None:
                        return None, "unexpected_predecessor_tail"
                else:
                    if (
                        not isinstance(tail, list)
                        or len(tail) != num_moe_layers
                        or any(
                            not isinstance(layer, list)
                            or len(layer) != top_k
                            or any(
                                type(expert) is not int or not 0 <= expert <= 32767
                                for expert in layer
                            )
                            for layer in tail
                        )
                    ):
                        return None, "invalid_predecessor_tail"
                    routed[position - 1] = torch.tensor(tail, dtype=torch.int16)
                    if call_context is not None:
                        call_context[position - 1] = routed[position - 1]
            if prefix_replacement:
                routed[position : position + contribution] = routes[prefix_len:].to(
                    torch.int16
                )
            elif mode == "full":
                routed[position : position + contribution] = routes.to(torch.int16)
            else:
                tail_start = position + span.carry_len
                routed[tail_start : position + contribution] = routes[
                    -span.generation_len :
                ].to(torch.int16)
            if span.generation_len:
                # Targets [start, end) use predictor routes [start-1, end-1).
                # The final generated token's route is only padding until a
                # continuation captures it, so never protect that final row.
                # Legacy tail-only fragments do not supply the first predictor;
                # do not freeze their missing prompt route against later repair.
                supplied_start = 0 if prefix_replacement else position
                if mode == "tail":
                    supplied_start = position + span.carry_len
                elif not prefix_replacement and (
                    boundary is not None or backpatch_predecessor_tail
                ):
                    supplied_start = position - 1
                predictor_start = max(0, supplied_start, position + span.carry_len - 1)
                predictor_end = position + contribution - 1
                if predictor_start < predictor_end:
                    protected_predictors[predictor_start:predictor_end] = True
        if call_context is not None:
            call_context[position : position + contribution] = routed[
                position : position + contribution
            ]
            call_views.append(
                CallRouteView(
                    staging_key=span.staging_key,
                    generation_start=position + span.carry_len,
                    generation_end=position + contribution,
                    routed_experts=call_context[: position + contribution].clone(),
                )
            )
        position += contribution
    if plan.spans and position != canonical_len:
        return None, ROUTE_FAILURE_ASSEMBLED_LENGTH
    return RoutePlanExecution(routed, tuple(call_views)), None
