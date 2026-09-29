# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental ordinary-output join for same-evidence tests, not runtime routing.

The input must be the complete accepted output, excluding seed prompt actions.
The base Gym response schema alone does not guarantee that property. This probe
uses existing content fingerprints and capture ancestry, never producer segments
or LogicalCCResult. It deliberately refuses ambiguous call partitions.
"""

from collections import defaultdict
from typing import Any

from nemo_gym.token_id_capture.fingerprint import (
    FINGERPRINT_VERSION,
    _is_assistant_authored,
    assistant_fingerprint,
)
from nemo_gym.token_id_capture.staging.records import CallRecord

from nemo_rl.environments.nemo_gym import (
    _detect_invalid_tool_call_and_malformed_thinking,
)
from nemo_rl.experience.rollout_reassembler import ActionOutputFlags, RolloutSelection


def selected_response_ids(
    records: list[CallRecord], response: dict[str, Any]
) -> tuple[str, ...]:
    """Join complete accepted output to a unique ordered set of captured calls.

    A call can contain multiple adjacent assistant items; adjacent calls need
    not have an observation between them. Explore both partitions and retain
    ambiguity rather than greedily taking the first content match. Capture
    parents constrain possible selections but never automatically add loss.

    This bounded CPU probe is not a long-context production search algorithm.
    Reasoning-only captures are unsupported even if they may be unselected.
    """
    if not records or any(
        record.fingerprint_version != FINGERPRINT_VERSION
        or not record.output_fingerprint
        for record in records
    ):
        raise ValueError("Unsupported missing/versioned/reasoning-only fingerprint")
    if len({record.response_id for record in records}) != len(records):
        raise ValueError("Ambiguous captured response identity")

    # Attach standalone reasoning to the following visible item. Reasoning is
    # excluded by Gym's existing fingerprint; it cannot supply identity here.
    atoms: list[tuple[int, list[dict]]] = []
    pending: list[dict] = []
    block = 0
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Expected ordinary response.output")
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Malformed ordinary output item")
        if item.get("type") == "reasoning":
            pending.append(item)
        elif _is_assistant_authored(item):
            atoms.append((block, [*pending, item]))
            pending = []
        else:
            if pending:
                raise ValueError("Unsupported reasoning-only returned action")
            block += 1
    if pending:
        raise ValueError("Unsupported reasoning-only returned action")
    if not atoms or len(atoms) > 256 or len(records) > 256:
        raise ValueError("Empty output or comparison probe size limit exceeded")

    by_fingerprint: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_fingerprint[record.output_fingerprint].append(index)
    edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for start in range(len(atoms)):
        items = []
        for end in range(start, len(atoms)):
            if atoms[end][0] != atoms[start][0]:
                break
            items.extend(atoms[end][1])
            for index in by_fingerprint.get(assistant_fingerprint(items), []):
                edges[start].append((end + 1, index))

    # At most two distinct paths per (output position, last capture). A third
    # cannot restore uniqueness: all paths at that state have identical futures.
    states: list[dict[int, set[tuple[int, ...]]]] = [
        defaultdict(set) for _ in range(len(atoms) + 1)
    ]
    states[0][-1].add(())
    for position in range(len(atoms)):
        for previous, paths in states[position].items():
            for end, index in edges[position]:
                parent = records[index].parent_call_id
                if index <= previous or (
                    parent is not None
                    and (previous < 0 or parent != records[previous].model_call_id)
                ):
                    continue
                target = states[end][index]
                for path in paths:
                    if len(target) < 2:
                        target.add((*path, index))

    native_terminal = next(
        (i for i, r in enumerate(records) if r.response_id == response.get("id")),
        None,
    )
    solutions = {
        path
        for index, paths in states[-1].items()
        if native_terminal is None or index == native_terminal
        for path in paths
    }
    if len(solutions) != 1:
        raise ValueError(
            "Ambiguous accepted call membership"
            if solutions
            else "No complete capture join"
        )
    return tuple(records[i].response_id for i in solutions.pop())


def selection_with_observed_metadata(
    records: list[CallRecord],
    response: dict[str, Any],
    observed_responses: dict[str, dict[str, Any]],
    *,
    output_check_config: dict[str, Any] | None = None,
) -> RolloutSelection:
    """Use existing raw response observations for per-call finish/output flags.

    These are captured model responses, including rejected calls, NOT an agent's
    accepted-call list. Observability is optional today; missing observations
    must fail this comparison rather than manufacture normal-stop metadata.
    """
    ids = selected_response_ids(records, response)
    flags, truncated = [], False
    for response_id in ids:
        observed = observed_responses.get(response_id)
        if observed is None or observed.get("status") not in (
            "completed",
            "incomplete",
        ):
            raise ValueError("Missing per-call completion metadata")
        reason = (observed.get("incomplete_details") or {}).get("reason")
        if observed["status"] == "incomplete" and reason is None:
            raise ValueError("Missing per-call completion metadata")
        truncated |= reason in ("length", "max_output_tokens")
        items = observed.get("output") or []
        flags.append(
            ActionOutputFlags(
                *_detect_invalid_tool_call_and_malformed_thinking(
                    items[-1] if items else {}, **(output_check_config or {})
                )
            )
        )
    return RolloutSelection(ids, tuple(flags), truncated=truncated)
