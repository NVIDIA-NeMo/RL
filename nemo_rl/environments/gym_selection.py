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

"""Join existing Gym rollout evidence to whole captured generations.

This optional Gym integration selects calls, not segments. Captured storage
parents and the existing finalizer continue to determine physical rows.
"""

from collections import defaultdict
from typing import Any

from nemo_gym.responses_converter import _RESPONSE_OUTPUT_BOUNDARY_TYPES
from nemo_gym.token_id_capture.completion import output_item_fingerprint
from nemo_gym.token_id_capture.staging.records import CallRecord


def _output_items(
    result: dict[str, Any], *, captured_response_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Normalize ordinary output, retaining envelope terminals by item offset."""
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
        if native_id in captured_response_ids:
            terminals[len(items)] = native_id
    return items, terminals


def select_captured_calls(
    records: list[CallRecord], result: dict[str, Any]
) -> list[CallRecord]:
    """Resolve references and validate any accompanying ordinary output.

    Stable IDs constrain content matching rather than overriding contradictions.
    Content-only matches must be unique locally; this is deliberately not an
    exponential search over possible transcript interpretations.
    """
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
            end: by_response[identity] for end, identity in envelope_ids.items()
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
                # A native envelope identifies its final whole call, including
                # when its output contains a cumulative conversation. Synthetic
                # exporter IDs are not capture identities and impose no bound.
                if any(position < stop < end for stop in terminals):
                    continue
                if end in terminals and terminals[end] != candidate:
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
    positions = {r.model_call_id: i for i, r in enumerate(records)}
    indexes = [positions[r.model_call_id] for r in selected]
    if indexes != sorted(set(indexes)):
        raise ValueError("Accepted generations are duplicated or out of capture order")
    if terminal is not None and terminal != selected[-1]:
        raise ValueError("Selected terminal differs from the scored response")
    declared = result.get("terminal_response_id")
    if declared and declared != selected[-1].response_id:
        raise ValueError("Declared terminal differs from accepted history")
    return selected
