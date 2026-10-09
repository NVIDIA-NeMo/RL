# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bind a harness's OpenCode session report to the selected capture chains.

The AgenticVBench agent returns ``session_traces``: for every OpenCode session
of an episode (the parent, then depth-one children in order of first
appearance) the served response ids it owns, its role, its parent session and
the parent's ``task`` tool-call id that spawned it. This module joins that
report to the response ids NeMo-RL selected from the capture ledger and emits
one trace tag per session in the shape ``session_traces.py`` validates, so a
subagent rollout trains as one scored logical owner whose segments are the
parent's context and each child's context.

No Gym or training runtime is imported here; the helpers are pure.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from nemo_rl.environments.session_traces import (
    TRAINING_MODES,
    TrainingMode,
    validate_session_trace_metadata,
)


def session_trace_metadata(
    session_traces: Any,
    *,
    selected_response_ids: Sequence[str],
    training_mode: TrainingMode,
    compaction_summaries: Any = None,
    rollout_id: str = "",
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    """Return ``(trace_tags, session_id_per_selected, role_per_selected)``.

    Every selected response must be owned by exactly one reported session.
    Sessions that own no selected response (for example a child whose calls
    were all retry dead branches) are dropped; the parent must keep at least
    one. Plain and subagent modes forbid OpenCode compaction; outside subagent
    mode a rollout must be a single session.
    """
    where = f"Rollout {rollout_id!r}: " if rollout_id else ""
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"{where}unknown training_mode {training_mode!r}")
    if not isinstance(session_traces, list) or not session_traces:
        raise ValueError(f"{where}{training_mode} training requires session traces")
    if training_mode != "compaction" and compaction_summaries not in (None, 0):
        raise ValueError(
            f"{where}{training_mode} training forbids OpenCode compaction "
            f"(summaries={compaction_summaries!r})"
        )
    owner_of: dict[str, int] = {}
    for index, trace in enumerate(session_traces):
        if not isinstance(trace, Mapping):
            raise ValueError(f"{where}malformed session trace")
        for response_id in trace.get("response_ids") or ():
            if not isinstance(response_id, str) or not response_id:
                raise ValueError(f"{where}malformed session response id")
            if response_id in owner_of:
                raise ValueError(
                    f"{where}response {response_id!r} is owned by two sessions"
                )
            owner_of[response_id] = index
    selected_set = set(selected_response_ids)
    if len(selected_set) != len(selected_response_ids):
        raise ValueError(f"{where}duplicate selected response ids")
    unowned = [rid for rid in selected_response_ids if rid not in owner_of]
    if unowned:
        raise ValueError(
            f"{where}selected responses without a session: {unowned[:3]!r}"
        )
    parent_index = next(
        (i for i, t in enumerate(session_traces) if t.get("session_role") == "parent"),
        None,
    )
    if parent_index is None:
        raise ValueError(f"{where}session traces have no parent session")
    root = session_traces[parent_index].get("session_id")
    tags: list[dict[str, Any]] = []
    for index, trace in enumerate(session_traces):
        owned = [
            rid for rid in trace.get("response_ids") or () if rid in selected_set
        ]
        if not owned:
            if index == parent_index:
                raise ValueError(f"{where}parent session owns no selected response")
            continue
        role = trace.get("session_role")
        if training_mode != "subagent" and role != "parent":
            raise ValueError(
                f"{where}{training_mode} training admits only the main session"
            )
        tags.append(
            {
                "training_mode": training_mode,
                "session_id": trace.get("session_id"),
                "root_session_id": root,
                "session_role": role,
                "parent_session_id": trace.get("parent_session_id"),
                "spawn_call_id": trace.get("spawn_call_id"),
                "segment_index": 0,
                "segment_boundary_reason": None,
                "is_compaction_summary": False,
                "request_ids": owned,
                "trace_index": len(tags),
            }
        )
    try:
        validate_session_trace_metadata(tags, training_mode=training_mode)
    except ValueError as error:
        raise ValueError(f"{where}{error}") from error
    by_session = {tag["session_id"]: tag for tag in tags}
    session_ids = tuple(
        session_traces[owner_of[rid]]["session_id"] for rid in selected_response_ids
    )
    roles = tuple(by_session[sid]["session_role"] for sid in session_ids)
    return tags, session_ids, roles


def session_training_overrides(training_mode: TrainingMode | None) -> dict[str, Any]:
    """OpenCode agent knobs that a NeMo-Gym ``training_mode`` implies.

    These are the ``responses_api_agents.agentic_vbench_agent`` overrides a
    recipe must carry so ``validate_session_training_config`` can see the
    harness contract next to the trainer's mode.
    """
    if training_mode is None:
        return {}
    knobs: dict[str, Any] = {
        "opencode_training_mode": training_mode,
        "opencode_subagents_enabled": training_mode == "subagent",
    }
    if training_mode == "subagent":
        knobs.update(
            opencode_subagent_max_depth=1,
            opencode_subagent_max_active=None,
            opencode_compaction_auto=False,
            opencode_compaction_prune=False,
        )
    return knobs
