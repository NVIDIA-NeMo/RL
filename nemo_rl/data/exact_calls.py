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

"""Construct route-aware exact-call trees without importing a training algorithm."""

from dataclasses import dataclass
from typing import Any, cast

import torch

from nemo_rl.data.packed_rollouts import TreeAttentionLayout
from nemo_rl.models.generation.interfaces import ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL


@dataclass(frozen=True)
class _ExactCallGeneratedSpan:
    """One independently sampled token span within an exact model call."""

    start: int
    end: int
    message: dict[str, Any]


@dataclass(frozen=True)
class _ExactCallSequence:
    """Tensor view of one captured model invocation."""

    index: int
    messages: list[dict[str, Any]]
    length: int
    token_parts: list[torch.Tensor]
    routed_expert_parts: list[torch.Tensor] | None
    has_incomplete_routes: bool
    generated_spans: list[_ExactCallGeneratedSpan]
    generation_replica_id: str | None
    generation_weight_version: int | None
    generation_weight_version_end: int | None
    kv_cache_scheduler_block_size: int | None
    kv_cache_hash_block_size: int | None
    kv_cache_num_cached_tokens: int | None


@dataclass
class _ExactCallPath:
    """A materialized trie leaf and the sampled spans assigned to it."""

    call: _ExactCallSequence
    generated_spans: list[_ExactCallGeneratedSpan]


def _flatten_exact_call(
    call: list[dict[str, Any]], call_index: int
) -> _ExactCallSequence:
    token_parts: list[torch.Tensor] = []
    route_parts: list[torch.Tensor] = []
    route_presence: list[bool] = []
    generated_spans: list[_ExactCallGeneratedSpan] = []
    offset = 0

    def call_metadata(field: str, expected_type: type) -> Any:
        values = {message[field] for message in call if message.get(field) is not None}
        if len(values) > 1:
            raise ValueError(
                f"NeMo-Gym exact call has inconsistent {field}: {sorted(values)!r}"
            )
        if not values:
            return None
        value = next(iter(values))
        if not isinstance(value, expected_type):
            raise ValueError(
                f"NeMo-Gym exact call {field} must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        return value

    for message in call:
        token_ids = message.get("token_ids")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError(
                "NeMo-Gym exact calls require one-dimensional token_ids tensors"
            )
        message_length = int(token_ids.shape[0])
        token_parts.append(token_ids)

        routed_experts = message.get("routed_experts")
        if routed_experts is not None:
            if not isinstance(routed_experts, torch.Tensor):
                raise ValueError("routed_experts must be a tensor when present")
            if routed_experts.shape[0] != message_length:
                raise ValueError(
                    "routed_experts must have one row per exact-call token: "
                    f"routes={routed_experts.shape[0]}, tokens={message_length}"
                )
            route_parts.append(routed_experts)
            route_presence.append(True)
        elif message_length:
            route_presence.append(False)

        generation_logprobs = message.get("generation_logprobs")
        if message.get("role") == "assistant" and generation_logprobs is not None:
            if (
                not isinstance(generation_logprobs, torch.Tensor)
                or generation_logprobs.ndim != 1
                or generation_logprobs.shape[0] != message_length
            ):
                raise ValueError(
                    "generation_logprobs must have one value per generated token"
                )
            if message_length:
                generated_spans.append(
                    _ExactCallGeneratedSpan(
                        start=offset,
                        end=offset + message_length,
                        message=message,
                    )
                )
        offset += message_length

    if offset <= 0:
        raise ValueError("NeMo-Gym produced an empty exact training call")

    has_any_routes = any(route_presence)
    has_incomplete_routes = has_any_routes and not all(route_presence)
    return _ExactCallSequence(
        index=call_index,
        messages=call,
        length=offset,
        token_parts=token_parts,
        routed_expert_parts=(
            route_parts if has_any_routes and not has_incomplete_routes else None
        ),
        has_incomplete_routes=has_incomplete_routes,
        generated_spans=generated_spans,
        generation_replica_id=call_metadata("ng_generation_replica_id", str),
        generation_weight_version=call_metadata("ng_generation_weight_version", int),
        generation_weight_version_end=call_metadata(
            "ng_generation_weight_version_end", int
        ),
        kv_cache_scheduler_block_size=call_metadata(
            "ng_kv_cache_scheduler_block_size", int
        ),
        kv_cache_hash_block_size=call_metadata("ng_kv_cache_hash_block_size", int),
        kv_cache_num_cached_tokens=call_metadata("ng_kv_cache_num_cached_tokens", int),
    )


def _tensor_parts_equal_prefix(
    prefix_parts: list[torch.Tensor],
    descendant_parts: list[torch.Tensor],
    *,
    prefix_length: int | None = None,
) -> bool:
    """Compare tensor sequences without concatenating their token dimension."""
    total_prefix_length = sum(int(part.shape[0]) for part in prefix_parts)
    if prefix_length is None:
        prefix_length = total_prefix_length
    if prefix_length < 0 or prefix_length > total_prefix_length:
        raise ValueError(
            "prefix_length must lie within the flattened prefix tensor sequence"
        )

    prefix_index = 0
    descendant_index = 0
    prefix_offset = 0
    descendant_offset = 0
    compared = 0
    while compared < prefix_length:
        if descendant_index >= len(descendant_parts):
            return False
        prefix_part = prefix_parts[prefix_index]
        descendant_part = descendant_parts[descendant_index]
        length = min(
            int(prefix_part.shape[0]) - prefix_offset,
            int(descendant_part.shape[0]) - descendant_offset,
            prefix_length - compared,
        )
        if not torch.equal(
            prefix_part[prefix_offset : prefix_offset + length],
            descendant_part[descendant_offset : descendant_offset + length],
        ):
            return False
        compared += length
        prefix_offset += length
        descendant_offset += length
        if prefix_offset == int(prefix_part.shape[0]):
            prefix_index += 1
            prefix_offset = 0
        if descendant_offset == int(descendant_part.shape[0]):
            descendant_index += 1
            descendant_offset = 0
    return True


def _tensor_parts_common_prefix_length(
    left_parts: list[torch.Tensor],
    right_parts: list[torch.Tensor],
    *,
    limit: int | None = None,
) -> int:
    """Return the number of equal token-axis rows without concatenating parts."""
    left_parts = [part for part in left_parts if part.shape[0]]
    right_parts = [part for part in right_parts if part.shape[0]]
    left_length = sum(int(part.shape[0]) for part in left_parts)
    right_length = sum(int(part.shape[0]) for part in right_parts)
    comparison_length = min(left_length, right_length)
    if limit is not None:
        if limit < 0:
            raise ValueError("common-prefix limit must be non-negative")
        comparison_length = min(comparison_length, limit)

    left_index = 0
    right_index = 0
    left_offset = 0
    right_offset = 0
    compared = 0
    while compared < comparison_length:
        left = left_parts[left_index]
        right = right_parts[right_index]
        length = min(
            int(left.shape[0]) - left_offset,
            int(right.shape[0]) - right_offset,
            comparison_length - compared,
        )
        left_rows = left[left_offset : left_offset + length]
        right_rows = right[right_offset : right_offset + length]
        equal_rows = left_rows.eq(right_rows).reshape(length, -1).all(dim=1)
        if not bool(equal_rows.all().item()):
            first_mismatch = int((~equal_rows).nonzero(as_tuple=False)[0].item())
            return compared + first_mismatch
        compared += length
        left_offset += length
        right_offset += length
        if left_offset == int(left.shape[0]):
            left_index += 1
            left_offset = 0
        if right_offset == int(right.shape[0]):
            right_index += 1
            right_offset = 0
    return compared


def _tensor_parts_row(parts: list[torch.Tensor], index: int) -> torch.Tensor:
    """Read one token-axis row from a tensor-part sequence."""
    if index < 0:
        raise IndexError("tensor-part row index must be non-negative")
    for part in parts:
        part_length = int(part.shape[0])
        if index < part_length:
            return part[index]
        index -= part_length
    raise IndexError("tensor-part row index is out of range")


def _tensor_parts_first_row_containing(
    parts: list[torch.Tensor], *, prefix_length: int, value: int
) -> int | None:
    """Find the first token-axis row containing ``value`` in a prefix."""
    remaining = prefix_length
    offset = 0
    for part in parts:
        if remaining <= 0:
            break
        length = min(int(part.shape[0]), remaining)
        matching_rows = part[:length].eq(value).reshape(length, -1).any(dim=1)
        if bool(matching_rows.any().item()):
            return offset + int(matching_rows.nonzero(as_tuple=False)[0].item())
        remaining -= length
        offset += length
    if remaining:
        raise ValueError("prefix_length exceeds the flattened tensor sequence")
    return None


def _matching_execution_metadata(
    left: _ExactCallSequence, right: _ExactCallSequence
) -> bool:
    """Return whether calls can refer to the same runtime KV-cache domain."""
    metadata = (
        (left.generation_replica_id, right.generation_replica_id),
        (
            left.kv_cache_scheduler_block_size,
            right.kv_cache_scheduler_block_size,
        ),
        (left.kv_cache_hash_block_size, right.kv_cache_hash_block_size),
    )
    return all(a is not None and a == b for a, b in metadata)


def _shared_execution_prefix_length(
    left: _ExactCallSequence, right: _ExactCallSequence
) -> int:
    """Return the longest token prefix safe to represent by one tree path.

    Route disagreement inside a cache page means the page was recomputed under
    a materially different execution. In that case sharing stops at the last
    fully matched page. Without cache metadata, exact equal token-and-route rows
    are still shareable. The final token of either request has no captured route;
    it may be shared when the other request later executes it.
    """
    token_prefix = _tensor_parts_common_prefix_length(
        left.token_parts, right.token_parts
    )
    if token_prefix == 0:
        return 0
    has_execution_metadata = (
        left.generation_replica_id is not None
        or right.generation_replica_id is not None
        or left.kv_cache_scheduler_block_size is not None
        or right.kv_cache_scheduler_block_size is not None
    )
    if has_execution_metadata and not _matching_execution_metadata(left, right):
        return 0

    share_limit = token_prefix
    if left.generation_weight_version != right.generation_weight_version:
        if (
            left.generation_weight_version is None
            or right.generation_weight_version is None
            or not _matching_execution_metadata(left, right)
            or right.kv_cache_num_cached_tokens is None
            or right.kv_cache_num_cached_tokens <= 0
        ):
            return 0
        share_limit = min(share_limit, right.kv_cache_num_cached_tokens)
        block_size = cast(int, right.kv_cache_scheduler_block_size)
        share_limit = (share_limit // block_size) * block_size
        if share_limit <= 0:
            return 0
    if left.has_incomplete_routes or right.has_incomplete_routes:
        return 0
    if (left.routed_expert_parts is None) != (right.routed_expert_parts is None):
        return 0
    if left.routed_expert_parts is None:
        return share_limit

    left_routes = left.routed_expert_parts
    right_routes = cast(list[torch.Tensor], right.routed_expert_parts)
    executed_prefix = min(share_limit, left.length - 1, right.length - 1)
    route_prefix = _tensor_parts_common_prefix_length(
        left_routes,
        right_routes,
        limit=executed_prefix,
    )
    missing_rows = [
        _tensor_parts_first_row_containing(
            routes,
            prefix_length=route_prefix,
            value=ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
        )
        for routes in (left_routes, right_routes)
    ]
    route_prefix = min(
        [route_prefix] + [row for row in missing_rows if row is not None]
    )

    if route_prefix < executed_prefix:
        if _matching_execution_metadata(left, right):
            block_size = cast(int, left.kv_cache_scheduler_block_size)
            return (route_prefix // block_size) * block_size
        return route_prefix
    return share_limit


def _tensor_parts_prefix_contains_value(
    parts: list[torch.Tensor], *, prefix_length: int, value: int
) -> bool:
    """Return whether a value occurs in the first ``prefix_length`` token rows."""
    remaining = prefix_length
    for part in parts:
        if remaining <= 0:
            break
        length = min(int(part.shape[0]), remaining)
        if length and bool(part[:length].eq(value).any().item()):
            return True
        remaining -= length
    if remaining:
        raise ValueError("prefix_length exceeds the flattened tensor sequence")
    return False


def _is_exact_execution_prefix(
    prefix: _ExactCallSequence, descendant: _ExactCallSequence
) -> bool:
    """Return whether ``prefix`` is a strict token-and-route prefix."""
    if prefix.length >= descendant.length:
        return False
    if not _tensor_parts_equal_prefix(prefix.token_parts, descendant.token_parts):
        return False

    # Partial route capture cannot prove execution-prefix equivalence. Calls with
    # no routes are the router-replay-off case and may still share token prefixes.
    if prefix.has_incomplete_routes or descendant.has_incomplete_routes:
        return False
    if (prefix.routed_expert_parts is None) != (descendant.routed_expert_parts is None):
        return False
    if prefix.routed_expert_parts is None:
        return True

    # vLLM has not forwarded the final sampled token, so its route row is a
    # placeholder. Compare only routes that were actually executed. A missing
    # route before that terminal position is different: execution happened but
    # capture was incomplete, so equivalence cannot be established.
    route_comparison_length = prefix.length - 1
    descendant_routes = cast(list[torch.Tensor], descendant.routed_expert_parts)
    if _tensor_parts_prefix_contains_value(
        prefix.routed_expert_parts,
        prefix_length=route_comparison_length,
        value=ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
    ) or _tensor_parts_prefix_contains_value(
        descendant_routes,
        prefix_length=route_comparison_length,
        value=ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
    ):
        return False
    return _tensor_parts_equal_prefix(
        prefix.routed_expert_parts,
        descendant_routes,
        prefix_length=route_comparison_length,
    )


def _page_aligned_execution_prefix_length(
    prefix: _ExactCallSequence, descendant: _ExactCallSequence
) -> int | None:
    """Return the fully reusable cache-page prefix, or ``None`` if unproven.

    The final sampled token has not been forwarded through the model, so it is
    not part of a reusable page even when the call length lands on a boundary.
    Routes in the trailing partial page may differ because vLLM recomputes that
    page. This classifies the fork; it does not merge the two training paths.
    """
    if prefix.length >= descendant.length:
        return None
    if not _tensor_parts_equal_prefix(prefix.token_parts, descendant.token_parts):
        return None
    if (
        prefix.generation_replica_id is None
        or prefix.generation_replica_id != descendant.generation_replica_id
    ):
        return None
    block_size = prefix.kv_cache_scheduler_block_size
    hash_block_size = prefix.kv_cache_hash_block_size
    if (
        block_size is None
        or block_size <= 0
        or block_size != descendant.kv_cache_scheduler_block_size
        or hash_block_size is None
        or hash_block_size <= 0
        or hash_block_size != descendant.kv_cache_hash_block_size
    ):
        return None
    if prefix.has_incomplete_routes or descendant.has_incomplete_routes:
        return None
    if prefix.routed_expert_parts is None or descendant.routed_expert_parts is None:
        return None

    if descendant.kv_cache_num_cached_tokens is None:
        return None
    reusable_length = (
        min(prefix.length - 1, descendant.kv_cache_num_cached_tokens) // block_size
    ) * block_size
    if reusable_length <= 0:
        return None
    descendant_routes = cast(list[torch.Tensor], descendant.routed_expert_parts)
    if _tensor_parts_prefix_contains_value(
        prefix.routed_expert_parts,
        prefix_length=reusable_length,
        value=ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
    ) or _tensor_parts_prefix_contains_value(
        descendant_routes,
        prefix_length=reusable_length,
        value=ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
    ):
        return None
    if not _tensor_parts_equal_prefix(
        prefix.routed_expert_parts,
        descendant_routes,
        prefix_length=reusable_length,
    ):
        return None
    if _tensor_parts_equal_prefix(
        prefix.routed_expert_parts,
        descendant_routes,
        prefix_length=prefix.length - 1,
    ):
        return None
    return reusable_length


def _exact_call_trie_diagnostics(
    calls: list[_ExactCallSequence],
) -> tuple[int, int, set[str]]:
    """Count conservative page forks and replicas used by one rollout."""
    page_forks = 0
    page_shared_tokens = 0
    for prefix in calls:
        for descendant in calls:
            reusable_length = _page_aligned_execution_prefix_length(prefix, descendant)
            if reusable_length is not None:
                page_forks += 1
                page_shared_tokens += reusable_length
    replicas = {
        call.generation_replica_id
        for call in calls
        if call.generation_replica_id is not None
    }
    return page_forks, page_shared_tokens, replicas


def _generated_spans_overlap(
    left: _ExactCallGeneratedSpan, right: _ExactCallGeneratedSpan
) -> bool:
    return left.start < right.end and right.start < left.end


def _slice_exact_call_message(
    message: dict[str, Any], start: int, end: int
) -> dict[str, Any]:
    """Slice token-aligned fields while preserving message-level metadata."""
    message_length = int(cast(torch.Tensor, message["token_ids"]).shape[0])
    result: dict[str, Any] = {}
    for key, value in message.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == message_length
        ):
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def _materialize_exact_call_path(path: _ExactCallPath) -> list[dict[str, Any]]:
    """Overlay assigned sampled spans onto one exact leaf execution path."""
    base_messages: list[tuple[int, int, dict[str, Any]]] = []
    boundaries = {0, path.call.length}
    offset = 0
    for message in path.call.messages:
        end = offset + int(cast(torch.Tensor, message["token_ids"]).shape[0])
        if end > offset:
            base_messages.append((offset, end, message))
            boundaries.update((offset, end))
        offset = end

    generated_spans = sorted(path.generated_spans, key=lambda span: span.start)
    for previous, current in zip(generated_spans, generated_spans[1:]):
        if _generated_spans_overlap(previous, current):
            raise ValueError(
                "Exact-call compaction assigned multiple sampled occurrences to "
                "the same token positions"
            )
    for span in generated_spans:
        boundaries.update((span.start, span.end))

    materialized: list[dict[str, Any]] = []
    sorted_boundaries = sorted(boundaries)
    base_index = 0
    span_index = 0
    for start, end in zip(sorted_boundaries, sorted_boundaries[1:]):
        while base_messages[base_index][1] <= start:
            base_index += 1
        base_start, _, base_message = base_messages[base_index]

        while (
            span_index < len(generated_spans)
            and generated_spans[span_index].end <= start
        ):
            span_index += 1
        owner = (
            generated_spans[span_index]
            if span_index < len(generated_spans)
            and generated_spans[span_index].start <= start
            and end <= generated_spans[span_index].end
            else None
        )
        if owner is not None:
            source = owner.message
            source_start = start - owner.start
            source_end = end - owner.start
        else:
            source = base_message
            source_start = start - base_start
            source_end = end - base_start
        materialized_message = _slice_exact_call_message(
            source, source_start, source_end
        )

        # Sampling ownership (role, logprobs, penalty flags) comes from the call
        # that generated this span, but routes describe the maximal leaf's exact
        # execution. In particular, do not let an ancestor's synthetic terminal
        # route overwrite the real route supplied by a descendant prefill.
        base_routes = base_message.get("routed_experts")
        if isinstance(base_routes, torch.Tensor):
            materialized_message["routed_experts"] = base_routes[
                start - base_start : end - base_start
            ]
        else:
            materialized_message.pop("routed_experts", None)
        materialized.append(materialized_message)

    return materialized


@dataclass(frozen=True)
class _ExactCallAttachment:
    """How one captured call attaches to an earlier execution path."""

    parent_call_index: int
    shared_length: int
    raw_segment_index: int | None


@dataclass(frozen=True)
class _ExactCallRawSegment:
    """New token suffix introduced by one captured call."""

    call_index: int
    call_start: int
    length: int
    old_start: int
    parent_old_node: int


@dataclass(frozen=True)
class _ExactCallTreePiece:
    """A non-branching slice of a raw suffix."""

    raw_segment_index: int
    local_start: int
    local_end: int

    @property
    def length(self) -> int:
        return self.local_end - self.local_start


@dataclass(frozen=True)
class _ExactCallTreeResult:
    """Materialized unique nodes and sampled-edge metadata for one rollout."""

    unique_message_log: list[dict[str, Any]]
    edge_message_log: list[dict[str, Any]]
    layout: TreeAttentionLayout


def _slice_exact_call_range(
    call: _ExactCallSequence, start: int, end: int
) -> list[dict[str, Any]]:
    """Slice a flattened call range back into token-aligned message pieces."""
    if start < 0 or end < start or end > call.length:
        raise ValueError(
            f"invalid exact-call range [{start}, {end}) for length {call.length}"
        )
    output: list[dict[str, Any]] = []
    message_start = 0
    for message in call.messages:
        message_length = int(cast(torch.Tensor, message["token_ids"]).shape[0])
        message_end = message_start + message_length
        overlap_start = max(start, message_start)
        overlap_end = min(end, message_end)
        if overlap_start < overlap_end:
            output.append(
                _slice_exact_call_message(
                    message,
                    overlap_start - message_start,
                    overlap_end - message_start,
                )
            )
        message_start = message_end
    if sum(int(cast(torch.Tensor, msg["token_ids"]).shape[0]) for msg in output) != (
        end - start
    ):
        raise RuntimeError(
            "exact-call range slicing did not cover the requested tokens"
        )
    return output


def _replace_message_log_route_row(
    messages: list[dict[str, Any]], offset: int, route: torch.Tensor
) -> None:
    """Replace one route row in a materialized message list."""
    for message in messages:
        token_count = int(cast(torch.Tensor, message["token_ids"]).shape[0])
        if offset < token_count:
            routes = message.get("routed_experts")
            if not isinstance(routes, torch.Tensor):
                raise ValueError(
                    "route override requires routed_experts on the message"
                )
            routes = routes.clone()
            routes[offset] = route.to(device=routes.device, dtype=routes.dtype)
            message["routed_experts"] = routes
            return
        offset -= token_count
    raise IndexError("route override offset is outside the message list")


def _build_exact_call_tree(
    rollout_calls: list[list[dict[str, Any]]],
) -> _ExactCallTreeResult:
    """Build a route-aware compressed execution tree for one rollout."""
    calls = [
        _flatten_exact_call(call, call_index)
        for call_index, call in enumerate(rollout_calls)
    ]
    attachments: list[_ExactCallAttachment] = []
    raw_segments: list[_ExactCallRawSegment] = []
    route_overrides: dict[int, torch.Tensor] = {}
    next_old_node = 0

    def resolve_old_node(call_index: int, token_position: int) -> int:
        if token_position < 0 or token_position >= calls[call_index].length:
            raise IndexError("exact-call token position is out of range")
        attachment = attachments[call_index]
        if (
            attachment.raw_segment_index is not None
            and token_position >= attachment.shared_length
        ):
            raw = raw_segments[attachment.raw_segment_index]
            return raw.old_start + token_position - attachment.shared_length
        if attachment.parent_call_index < 0:
            raise RuntimeError("root exact call did not materialize its token")
        return resolve_old_node(attachment.parent_call_index, token_position)

    def materialized_route(old_node: int) -> torch.Tensor:
        if old_node in route_overrides:
            return route_overrides[old_node]
        for raw in raw_segments:
            if raw.old_start <= old_node < raw.old_start + raw.length:
                routes = calls[raw.call_index].routed_expert_parts
                if routes is None:
                    raise ValueError("shared routed node has no captured routes")
                return _tensor_parts_row(
                    routes, raw.call_start + old_node - raw.old_start
                )
        raise IndexError("shared routed node is outside the materialized tree")

    for call_index, call in enumerate(calls):
        parent_call_index = -1
        shared_length = 0
        for candidate_index in range(call_index):
            candidate_shared = _shared_execution_prefix_length(
                calls[candidate_index], call
            )
            candidate = calls[candidate_index]
            if (
                candidate_shared == candidate.length
                and candidate.routed_expert_parts is not None
                and call.routed_expert_parts is not None
            ):
                # A terminal placeholder is a wildcard only until that physical
                # node has been executed. Short aliases must respect the route
                # already selected by another descendant of the same node.
                terminal_position = candidate.length - 1
                existing = materialized_route(
                    resolve_old_node(candidate_index, terminal_position)
                )
                incoming = _tensor_parts_row(
                    call.routed_expert_parts, terminal_position
                )
                if (
                    not bool(existing.eq(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL).any())
                    and not bool(
                        incoming.eq(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL).any()
                    )
                    and not torch.equal(existing, incoming)
                ):
                    candidate_shared = terminal_position
                    if _matching_execution_metadata(candidate, call):
                        block_size = cast(int, candidate.kv_cache_scheduler_block_size)
                        candidate_shared = (candidate_shared // block_size) * block_size
            if candidate_shared > shared_length or (
                candidate_shared == shared_length
                and candidate_shared > 0
                and candidate_index > parent_call_index
            ):
                parent_call_index = candidate_index
                shared_length = candidate_shared

        raw_segment_index = None
        if shared_length < call.length:
            parent_old_node = (
                -1
                if shared_length == 0
                else resolve_old_node(parent_call_index, shared_length - 1)
            )
            raw_segment_index = len(raw_segments)
            raw_segments.append(
                _ExactCallRawSegment(
                    call_index=call_index,
                    call_start=shared_length,
                    length=call.length - shared_length,
                    old_start=next_old_node,
                    parent_old_node=parent_old_node,
                )
            )
            next_old_node += call.length - shared_length
        elif parent_call_index < 0:
            raise RuntimeError(
                "non-empty exact call produced neither a root nor a parent"
            )
        attachments.append(
            _ExactCallAttachment(
                parent_call_index=parent_call_index,
                shared_length=shared_length,
                raw_segment_index=raw_segment_index,
            )
        )

        if parent_call_index >= 0 and call.routed_expert_parts is not None:
            parent = calls[parent_call_index]
            if shared_length == parent.length:
                terminal_position = parent.length - 1
                incoming = _tensor_parts_row(
                    call.routed_expert_parts, terminal_position
                )
                if not bool(incoming.eq(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL).any()):
                    old_node = resolve_old_node(parent_call_index, terminal_position)
                    if bool(
                        materialized_route(old_node)
                        .eq(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL)
                        .any()
                    ):
                        route_overrides[old_node] = incoming

    def raw_and_local_for_old_node(old_node: int) -> tuple[int, int]:
        for raw_index, raw in enumerate(raw_segments):
            if raw.old_start <= old_node < raw.old_start + raw.length:
                return raw_index, old_node - raw.old_start
        raise IndexError("tree node does not belong to a raw segment")

    breakpoints = [{0, raw.length} for raw in raw_segments]
    for raw in raw_segments:
        if raw.parent_old_node < 0:
            continue
        parent_raw_index, parent_local = raw_and_local_for_old_node(raw.parent_old_node)
        breakpoints[parent_raw_index].add(parent_local + 1)

    pieces: list[_ExactCallTreePiece] = []
    raw_piece_indices: list[list[int]] = []
    for raw_index, raw_breakpoints in enumerate(breakpoints):
        piece_indices = []
        ordered = sorted(raw_breakpoints)
        for start, end in zip(ordered, ordered[1:]):
            piece_indices.append(len(pieces))
            pieces.append(
                _ExactCallTreePiece(
                    raw_segment_index=raw_index,
                    local_start=start,
                    local_end=end,
                )
            )
        raw_piece_indices.append(piece_indices)

    def piece_for_old_node(old_node: int) -> int:
        raw_index, local = raw_and_local_for_old_node(old_node)
        for piece_index in raw_piece_indices[raw_index]:
            piece = pieces[piece_index]
            if piece.local_start <= local < piece.local_end:
                return piece_index
        raise RuntimeError("tree node was not covered by a split segment")

    piece_parents: list[int] = [-1] * len(pieces)
    for raw_index, piece_indices in enumerate(raw_piece_indices):
        raw = raw_segments[raw_index]
        for index_in_raw, piece_index in enumerate(piece_indices):
            if index_in_raw:
                piece_parents[piece_index] = piece_indices[index_in_raw - 1]
            elif raw.parent_old_node >= 0:
                piece_parents[piece_index] = piece_for_old_node(raw.parent_old_node)

    children: list[list[int]] = [[] for _ in pieces]
    roots: list[int] = []
    for piece_index, parent in enumerate(piece_parents):
        if parent < 0:
            roots.append(piece_index)
        else:
            children[parent].append(piece_index)

    dfs_piece_indices: list[int] = []

    def visit(piece_index: int) -> None:
        dfs_piece_indices.append(piece_index)
        for child in children[piece_index]:
            visit(child)

    for root in roots:
        visit(root)
    if len(dfs_piece_indices) != len(pieces):
        raise RuntimeError("exact-call tree contains an unreachable or cyclic segment")

    dfs_index_by_piece = {
        piece_index: dfs_index
        for dfs_index, piece_index in enumerate(dfs_piece_indices)
    }
    new_starts: list[int] = []
    total_unique_tokens = 0
    for piece_index in dfs_piece_indices:
        new_starts.append(total_unique_tokens)
        total_unique_tokens += pieces[piece_index].length

    def new_node_for_old_node(old_node: int) -> int:
        raw_index, local = raw_and_local_for_old_node(old_node)
        piece_index = piece_for_old_node(old_node)
        piece = pieces[piece_index]
        if piece.raw_segment_index != raw_index:
            raise RuntimeError("tree piece/raw mapping is inconsistent")
        dfs_index = dfs_index_by_piece[piece_index]
        return new_starts[dfs_index] + local - piece.local_start

    unique_message_log: list[dict[str, Any]] = []
    for piece_index in dfs_piece_indices:
        piece = pieces[piece_index]
        raw = raw_segments[piece.raw_segment_index]
        call = calls[raw.call_index]
        messages = _slice_exact_call_range(
            call,
            raw.call_start + piece.local_start,
            raw.call_start + piece.local_end,
        )
        for old_node, route in route_overrides.items():
            if (
                raw.old_start + piece.local_start
                <= old_node
                < (raw.old_start + piece.local_end)
            ):
                _replace_message_log_route_row(
                    messages,
                    old_node - raw.old_start - piece.local_start,
                    route,
                )
        unique_message_log.extend(messages)

    if not unique_message_log:
        raise RuntimeError("exact-call tree materialized no unique tokens")
    first_token = cast(torch.Tensor, unique_message_log[0]["token_ids"])[0:1]
    edge_message_log: list[dict[str, Any]] = [
        {"role": "user", "token_ids": first_token.clone()}
    ]
    edge_source_indices: list[int] = []
    for call_index, call in enumerate(calls):
        for span in call.generated_spans:
            edge_message = _slice_exact_call_message(
                span.message,
                0,
                span.end - span.start,
            )
            edge_message.pop("routed_experts", None)
            edge_message_log.append(edge_message)
            for token_position in range(span.start, span.end):
                if token_position == 0:
                    raise ValueError(
                        "generated exact-call token has no autoregressive predecessor"
                    )
                source_old_node = resolve_old_node(call_index, token_position - 1)
                edge_source_indices.append(new_node_for_old_node(source_old_node))

    # Collapse unary chains so the final metadata contains maximal
    # non-branching segments rather than preserving arbitrary request boundaries.
    segment_lengths_list: list[int] = []
    segment_parents_list: list[int] = []
    segment_by_piece: dict[int, int] = {}
    for piece_index in dfs_piece_indices:
        parent_piece = piece_parents[piece_index]
        if parent_piece >= 0 and len(children[parent_piece]) == 1:
            segment_index = segment_by_piece[parent_piece]
            segment_lengths_list[segment_index] += pieces[piece_index].length
        else:
            segment_index = len(segment_lengths_list)
            segment_lengths_list.append(pieces[piece_index].length)
            segment_parents_list.append(
                -1 if parent_piece < 0 else segment_by_piece[parent_piece]
            )
        segment_by_piece[piece_index] = segment_index

    segment_lengths = tuple(segment_lengths_list)
    segment_parents = tuple(segment_parents_list)
    segment_depths_list: list[int] = []
    for segment_index, parent in enumerate(segment_parents):
        segment_depths_list.append(
            0 if parent < 0 else segment_depths_list[parent] + segment_lengths[parent]
        )
    layout = TreeAttentionLayout(
        segment_lengths=segment_lengths,
        segment_parents=segment_parents,
        segment_depths=tuple(segment_depths_list),
        edge_source_indices=tuple(edge_source_indices),
        original_token_count=sum(call.length for call in calls),
    )
    layout.validate()
    return _ExactCallTreeResult(
        unique_message_log=unique_message_log,
        edge_message_log=edge_message_log,
        layout=layout,
    )


def _compact_exact_call_sequences(
    rollout_calls: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Materialize maximal paths through the exact token-and-route prefix trie.

    The trie is represented implicitly by strict-prefix comparisons so building it
    does not allocate one Python object per captured token. Calls are visited from
    leaves toward the root. An internal call's sampled spans are assigned to one
    compatible descendant; calls that retokenize, route differently, or would
    duplicate an already-owned sampled span remain separate paths.
    """
    calls = [
        _flatten_exact_call(call, call_index)
        for call_index, call in enumerate(rollout_calls)
    ]
    paths: list[_ExactCallPath] = []
    for call in reversed(calls):
        assigned = False
        for path in sorted(paths, key=lambda candidate: candidate.call.index):
            if not _is_exact_execution_prefix(call, path.call):
                continue
            if any(
                _generated_spans_overlap(span, owned)
                for span in call.generated_spans
                for owned in path.generated_spans
            ):
                continue
            path.generated_spans.extend(call.generated_spans)
            assigned = True
            break
        if not assigned:
            paths.append(
                _ExactCallPath(
                    call=call,
                    generated_spans=list(call.generated_spans),
                )
            )

    paths.sort(key=lambda path: path.call.index)
    materialized_paths = [_materialize_exact_call_path(path) for path in paths]
    return (
        [message for path in materialized_paths for message in path],
        [
            sum(len(cast(torch.Tensor, message["token_ids"])) for message in path)
            for path in materialized_paths
        ],
    )
