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

"""Select captured calls for training, independently of row reconstruction.

A selection policy is a synchronous callable ``(records, result)`` returning
``(selected_records, terminal_response_id)``. It must return unique, unchanged
records from this attempt and identify the scored response within its selection.
Order determines root/path priority, not storage ancestry. Policies choose
membership; shared capture validation and reconstruction remain authoritative.
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import CallRecord

type SelectionPolicy = Callable[
    [list[CallRecord], dict[str, Any]], tuple[list[CallRecord], str | None]
]


def _output_items(
    result: dict[str, Any], *, captured_response_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[int, str | None]]:
    """Normalize ordinary output, retaining envelope boundaries and identities."""
    # Gym is optional; resolving policies must also work in non-Gym config tools.
    from nemo_gym.responses_converter import _RESPONSE_OUTPUT_BOUNDARY_TYPES

    responses = result.get("responses") or [result.get("response") or {}]
    sessions = {(r.get("metadata") or {}).get("session_id") for r in responses}
    if len(sessions) > 1:
        raise ValueError("Multiple sessions require qualified training membership")
    items = []
    terminals = {}
    for response in responses:
        response_items = []
        output = response.get("output")
        if not isinstance(output, list):
            raise ValueError("Expected ordinary response output")
        if response.get("contains_transitions"):
            seen = {}
            for snapshot in output:
                if not isinstance(snapshot, list):
                    raise ValueError("Malformed transition snapshot")
                for item in snapshot:
                    if not isinstance(item, dict):
                        raise ValueError("Malformed ordinary output item")
                    identity = item.get("id")
                    if identity and identity in seen:
                        if item != seen[identity]:
                            raise ValueError(
                                "Snapshot reused an item ID with different content"
                            )
                        continue
                    if identity:
                        seen[identity] = item
                    response_items.append(item)
        else:
            response_items.extend(output)
        if any(not isinstance(item, dict) for item in response_items):
            raise ValueError("Malformed ordinary output item")
        authored = []
        for position, item in enumerate(response_items):
            item_type = item.get("type")
            if (
                item.get("role") == "assistant"
                or item_type in _RESPONSE_OUTPUT_BOUNDARY_TYPES
            ):
                # Gym's generation boundary includes types that capture cannot
                # yet fingerprint. Never silently discard these from history.
                if item_type not in (None, "message", "reasoning", "function_call"):
                    raise ValueError(
                        "Unsupported generated output item: "
                        f"type={item_type!r}, item_id={item.get('id')!r}, "
                        f"output_index={position}"
                    )
                authored.append(item)
        native_id = response.get("id")
        if native_id in captured_response_ids and not authored:
            raise ValueError("No accepted captured generations in response output")
        items.extend(authored)
        if authored:
            terminals[len(items)] = (
                native_id if native_id in captured_response_ids else None
            )
    return items, terminals


def select_captured_calls(
    records: list[CallRecord], result: dict[str, Any]
) -> tuple[list[CallRecord], str | None]:
    """Resolve references and validate any accompanying ordinary output.

    Stable IDs constrain content matching rather than overriding contradictions.
    Content-only matches must be unique locally; this is deliberately not an
    exponential search over possible transcript interpretations.
    """
    # Defer the optional Gym dependency until the policy is executed.
    from nemo_gym.token_id_capture.completion import output_item_fingerprint

    by_response: dict[str, CallRecord] = {r.response_id: r for r in records}
    by_call: dict[str, CallRecord] = {r.model_call_id: r for r in records}
    if len(by_response) != len(records) or len(by_call) != len(records):
        raise ValueError("Capture contains reused call or response identities")
    terminal = by_response.get((result.get("response") or {}).get("id") or "")
    selected = []
    referenced: list[CallRecord] | None = None
    turns = (result.get("ng_trajectory") or {}).get("turns") or []
    if any(turn.get("model_calls") for turn in turns):
        referenced = []
        for turn_index, turn in enumerate(turns):
            refs = turn.get("model_calls") or []
            if not refs:
                raise ValueError(
                    "Only part of the returned trajectory has call references: "
                    f"turn={turn_index}"
                )
            for ref_index, ref in enumerate(refs):
                location = (
                    f"turn={turn_index}, reference={ref_index}, "
                    f"response_id={ref.get('response_id')!r}, "
                    f"model_call_id={ref.get('model_call_id')!r}"
                )
                witnesses = []
                for key, index in (
                    ("response_id", by_response),
                    ("model_call_id", by_call),
                ):
                    if ref.get(key):
                        if ref[key] not in index:
                            raise ValueError(
                                "Trajectory references a call outside this capture: "
                                + location
                            )
                        witnesses.append(index[ref[key]])
                if not witnesses or any(r != witnesses[0] for r in witnesses):
                    raise ValueError(
                        "Missing or contradictory trajectory call identity: " + location
                    )
                referenced.append(witnesses[0])
    if referenced is None or "response" in result or "responses" in result:
        items, envelope_ids = _output_items(
            result, captured_response_ids=set(by_response)
        )
        terminals = {
            end: by_response[identity]
            for end, identity in envelope_ids.items()
            if identity is not None
        }
        fingerprints = [output_item_fingerprint(item) for item in items]
        by_item_id: dict[str, set[str]] = defaultdict(set)
        by_first: dict[str, list[CallRecord]] = defaultdict(list)
        for record in records:
            if record.output_items:
                by_first[record.output_items[0].fingerprint].append(record)
                for item in record.output_items:
                    if item.id:
                        by_item_id[item.id].add(record.model_call_id)
        # Exact later identities can disambiguate an earlier retry through a
        # verified parent. They never add unreturned actions to the selection.
        required = set()
        identified = {terminal.model_call_id} if terminal is not None else set()
        identified.update(record.model_call_id for record in terminals.values())
        for item in items:
            matches = by_item_id.get(item.get("id") or "", set())
            if len(matches) > 1:
                raise ValueError("Served item identity was reused by multiple calls")
            identified.update(matches)
        for identity in identified:
            visited = set()
            ancestor: str | None = identity
            while ancestor is not None:
                if ancestor in visited or ancestor not in by_call:
                    raise ValueError("Broken captured parent chain")
                visited.add(ancestor)
                required.add(ancestor)
                ancestor = by_call.get(ancestor).parent_call_id
        position = 0
        envelope_selected_start = 0
        while position < len(items):
            matches = []
            for candidate in by_first.get(fingerprints[position], []):
                # References constrain the same whole-call content check. They
                # cannot override edited, missing, or synthetic output items.
                if referenced is not None and (
                    len(selected) >= len(referenced)
                    or candidate != referenced[len(selected)]
                ):
                    continue
                evidence = candidate.output_items or []
                end = position + len(evidence)
                # Whole calls cannot cross response envelopes. A native ID must
                # occur inside its own envelope, but need not be its last call.
                if any(position < stop < end for stop in envelope_ids):
                    continue
                if (
                    end in terminals
                    and terminals[end] != candidate
                    and terminals[end] not in selected[envelope_selected_start:]
                ):
                    continue
                if [item.fingerprint for item in evidence] != fingerprints[
                    position:end
                ]:
                    continue
                if any(
                    by_item_id.get(item.get("id") or "", {candidate.model_call_id})
                    != {candidate.model_call_id}
                    for item in items[position:end]
                ):
                    continue
                matches.append(candidate)
            corroborated = [r for r in matches if r.model_call_id in required]
            if len(matches) > 1 and len(corroborated) == 1:
                matches = corroborated
            if len(matches) != 1:
                expected = (
                    referenced[len(selected)]
                    if referenced is not None and len(selected) < len(referenced)
                    else None
                )
                reason = (
                    "multiple captured calls match"
                    if matches
                    else "no whole captured call matches content and identity"
                )
                raise ValueError(
                    "Missing or ambiguous captured generation in ordinary output: "
                    f"authored_output_index={position}, "
                    f"item_id={items[position].get('id')!r}, reason={reason}; "
                    f"expected_call={expected.model_call_id if expected else None!r}, "
                    f"item_identity_calls={sorted(by_item_id.get(items[position].get('id') or '', set()))}, "
                    f"matching_calls={[r.model_call_id for r in matches]}"
                )
            selected.append(matches[0])
            position += len(matches[0].output_items or [])
            if position in envelope_ids:
                envelope_selected_start = len(selected)
        if referenced is not None and len(selected) != len(referenced):
            raise ValueError(
                "Returned output is missing referenced generations: "
                f"authored_output_index={len(items)}, "
                f"remaining_calls={[r.model_call_id for r in referenced[len(selected) :]]}"
            )
    else:
        selected = referenced
    if not selected:
        raise ValueError("No accepted captured generations")
    if len({record.model_call_id for record in selected}) != len(selected):
        raise ValueError("Accepted generations are duplicated")
    if terminal is not None and terminal not in selected:
        raise ValueError("Scored response is absent from accepted history")
    declared = result.get("terminal_response_id")
    if declared:
        if declared not in {record.response_id for record in selected}:
            raise ValueError("Declared terminal is absent from accepted history")
        if terminal is not None and declared != terminal.response_id:
            raise ValueError("Declared terminal differs from the scored response")
    terminal_response_id = declared or (terminal.response_id if terminal else None)
    if terminal_response_id is None:
        # Returned history is ordered; preserve the existing last accepted call
        # convention when envelopes use harness-local rather than served IDs.
        terminal_response_id = selected[-1].response_id
    return selected, terminal_response_id


class TerminalNotIdentified(ValueError):
    """No captured call can be named as the scored terminal; the owner is masked."""


# Gym attribution methods, in RolloutReceipt.terminal_selection terms.
_RECEIPT_METHOD = {
    "explicit": "declared",
    "declared": "declared",
    "response_id": "response_id",
    "content": "content",
    "item_id": "content",
    "content_output": "content",
    "content_cumulative": "content",
    "content_tool_ids": "content",
}


def resolve_scored_terminal(
    records: list[CallRecord], result: dict[str, Any]
) -> tuple[CallRecord, str, str]:
    """Name the scored call: witness attribution, else the parent-link heuristic.

    Returns ``(record, terminal_selection, attribution_reason)``. A declared
    terminal that no witness confirms never falls back to the heuristic.
    Raises ``TerminalNotIdentified`` when neither stage names a call.
    """
    # Defer the optional Gym dependency until the policy is executed.
    from nemo_gym.token_id_capture.staging import resolve_terminal, select_terminal_call

    declared = result.get("terminal_response_id")
    attribution = resolve_terminal(
        records, result.get("response"), declared_response_id=declared
    )
    by_call = {record.model_call_id: record for record in records}
    if attribution.attributed:
        return (
            by_call[attribution.model_call_id],
            _RECEIPT_METHOD[attribution.method],
            attribution.reason,
        )
    if declared:
        raise TerminalNotIdentified(
            "declared terminal is not attributable: " + attribution.reason
        )
    heuristic = select_terminal_call(records)
    if heuristic.terminal_model_call_id is None:
        raise TerminalNotIdentified(
            f"no scored terminal: {attribution.reason}; heuristic: {heuristic.reason}"
        )
    return by_call[heuristic.terminal_model_call_id], "heuristic", attribution.reason


def select_all_captured(
    records: list[CallRecord], result: dict[str, Any]
) -> tuple[list[CallRecord], str | None]:
    """Include all calls except unusable, unscored leaves, retaining attribution."""
    terminal, _, _ = resolve_scored_terminal(records, result)
    terminal_response_id = terminal.response_id
    # Unusable parents or the scored call still invalidate the owner downstream.
    parents = {record.parent_call_id for record in records}
    selected = []
    for record in records:
        unusable = record.response_status in ("failed", "unknown") or (
            record.response_status == "incomplete"
            and bool(record.finish_reason)
            and record.finish_reason
            not in ("length", "max_tokens", "max_output_tokens")
        )
        if (
            not unusable
            or record.model_call_id == terminal.model_call_id
            or record.model_call_id in parents
        ):
            selected.append(record)
    return selected, terminal_response_id


_LENGTH_FINISH_REASONS = ("length", "max_tokens", "max_output_tokens")


def _trainable(record: CallRecord) -> bool:
    """Completed, or stopped by a length limit; anything else ends its chain."""
    return record.response_status == "completed" or (
        record.response_status == "incomplete"
        and record.finish_reason in _LENGTH_FINISH_REASONS
    )


def select_continued_chains(
    records: list[CallRecord], result: dict[str, Any]
) -> tuple[list[CallRecord], str | None]:
    """Train the scored chain and one continued chain per other root.

    Every independent session (for example, a subagent) becomes its own
    chain. The scored chain comes first and holds only the scored call's
    ancestors; the scored call is named by ``resolve_scored_terminal``. Every
    other root, in
    admission order, descends to a unique leaf, preferring children that were
    continued; an ambiguous fork drops that root. Calls that did not finish
    trainably count as absent, so they end a chain instead of poisoning the
    owner. A leaf whose cumulative hash repeats an earlier chain is a
    byte-identical retry and is dropped. Retries with identical prompts need
    token evidence and are handled at finalization
    (``token_capture.mask_retried_roots``).
    """
    terminal, _, _ = resolve_scored_terminal(records, result)
    by_call = {record.model_call_id: record for record in records}
    scored_chain = [terminal]
    while scored_chain[-1].parent_call_id is not None:
        parent = by_call.get(scored_chain[-1].parent_call_id)
        if parent is None or parent in scored_chain:
            raise ValueError("Scored chain has a missing or cyclic ancestor")
        scored_chain.append(parent)
    scored_chain.reverse()

    position = {record.model_call_id: index for index, record in enumerate(records)}
    children: dict[str, list[CallRecord]] = defaultdict(list)
    for record in records:
        if record.parent_call_id is not None and _trainable(record):
            children[record.parent_call_id].append(record)
    roots = sorted(
        (
            record
            for record in records
            if record.parent_call_id is None
            and record is not scored_chain[0]
            and _trainable(record)
        ),
        key=lambda record: (
            record.admitted_at is None,
            record.admitted_at or 0.0,
            position[record.model_call_id],
        ),
    )
    selected = list(scored_chain)
    seen_leaves = {scored_chain[-1].cumulative_hash}
    for root in roots:
        path = [root]
        while candidates := children[path[-1].model_call_id]:
            continued = [c for c in candidates if children[c.model_call_id]]
            pool = continued or candidates
            if len(pool) != 1:
                break
            path.append(pool[0])
        else:
            if path[-1].cumulative_hash not in seen_leaves:
                seen_leaves.add(path[-1].cumulative_hash)
                selected.extend(path)
    return selected, scored_chain[-1].response_id


def cap_selected_roots(
    selected: list[CallRecord], terminal_response_id: str, max_roots: int
) -> list[CallRecord]:
    """Keep the scored call's root and the first ``max_roots - 1`` other roots.

    Roots are counted in selection order. Under ``continued_chains`` each root
    yields one training row, so this caps rows per rollout.
    """
    by_call = {record.model_call_id: record for record in selected}

    def root_of(record: CallRecord) -> str:
        while record.parent_call_id is not None and record.parent_call_id in by_call:
            record = by_call[record.parent_call_id]
        return record.model_call_id

    roots = list(dict.fromkeys(root_of(record) for record in selected))
    scored = next(r for r in selected if r.response_id == terminal_response_id)
    scored_root = root_of(scored)
    kept = {scored_root, *[r for r in roots if r != scored_root][: max_roots - 1]}
    return [record for record in selected if root_of(record) in kept]


SELECTION_POLICIES: dict[str, SelectionPolicy] = {
    "returned_history": select_captured_calls,
    "all_captured": select_all_captured,
    "continued_chains": select_continued_chains,
}


def resolve_selection_policy(name: str) -> SelectionPolicy:
    """Resolve a built-in name or an importable ``package.module:function``."""
    if name in SELECTION_POLICIES:
        return SELECTION_POLICIES[name]
    module_name, separator, function_name = name.partition(":")
    if not separator or not module_name or not function_name.isidentifier():
        raise ValueError(
            f"Unknown selection_policy {name!r}: expected a built-in "
            f"({', '.join(SELECTION_POLICIES)}) or 'package.module:function'"
        )
    try:
        policy = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load selection_policy {name!r}: {error}") from error
    if not callable(policy):
        raise ValueError(f"selection_policy {name!r} is not callable")
    return cast(SelectionPolicy, policy)
