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

"""Validate session ownership without importing the training or Gym runtime."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

TrainingMode = Literal["plain", "compaction", "subagent"]
TRAINING_MODES = ("plain", "compaction", "subagent")
_AGENT_SEMANTIC_KEYS = (
    "opencode_training_mode",
    "opencode_subagents_enabled",
    "opencode_subagent_max_depth",
    "opencode_subagent_max_active",
)


def session_training_semantics(env: Mapping[str, Any]) -> dict[str, Any]:
    """Select mode/limit controls for resume, excluding operational output paths."""
    gym = env.get("nemo_gym")
    if not isinstance(gym, Mapping):
        return {}
    agents = {}

    def collect(node: Mapping[str, Any], path: tuple[str, ...]) -> None:
        for name, value in node.items():
            if not isinstance(value, Mapping):
                continue
            if name == "responses_api_agents":
                for agent_name, agent in value.items():
                    if not isinstance(agent, Mapping):
                        continue
                    semantic = {
                        key: item
                        for key, item in agent.items()
                        if key in _AGENT_SEMANTIC_KEYS
                        or key.startswith("opencode_compaction")
                        or key.startswith("opencode_prun")
                    }
                    if semantic:
                        agents[".".join((*path, name, agent_name))] = semantic
            else:
                collect(value, (*path, name))

    collect(gym, ())
    return {"training_mode": gym.get("training_mode"), "agents": agents}


def validate_session_training_config(env: Mapping[str, Any]) -> TrainingMode | None:
    """Require an explicit, consistent opt-in for subagent-only training."""
    semantics = session_training_semantics(env)
    mode = semantics.get("training_mode")
    if mode is not None and mode not in TRAINING_MODES:
        raise ValueError(f"Unknown NeMo-Gym training_mode: {mode!r}")
    agents = semantics.get("agents", {})
    if mode == "subagent" and not agents:
        raise ValueError("Subagent training requires a configured OpenCode agent")
    for agent in agents.values():
        declared = agent.get("opencode_training_mode")
        enabled = agent.get("opencode_subagents_enabled")
        if declared is not None and declared not in TRAINING_MODES:
            raise ValueError(f"Unknown OpenCode training mode: {declared!r}")
        if declared is not None and declared != mode:
            raise ValueError("OpenCode and NeMo-Gym training modes disagree")
        if mode == "subagent":
            if declared != mode or enabled is not True:
                raise ValueError("Subagent mode requires explicit OpenCode subagents")
            depth = agent.get("opencode_subagent_max_depth")
            if type(depth) is not int or depth != 1:
                raise ValueError(
                    "Subagent training requires opencode_subagent_max_depth=1"
                )
            concurrency_key = "opencode_subagent_max_active"
            if concurrency_key not in agent or (
                agent[concurrency_key] is not None
                and (
                    type(agent[concurrency_key]) is not int
                    or agent[concurrency_key] != 1
                )
            ):
                raise ValueError(
                    "Subagent training requires explicit native parallel concurrency "
                    "(null) or the historical single-child limit (1)"
                )
            for key, value in agent.items():
                if key in (
                    "opencode_compaction_auto",
                    "opencode_compaction_prune",
                    "opencode_pruning_enabled",
                ):
                    if value is not False:
                        raise ValueError("Subagent training forbids compaction/pruning")
        elif enabled:
            raise ValueError("Subagents require explicit NeMo-Gym subagent mode")
    return mode


def validate_session_trace_metadata(
    metadata: Sequence[Mapping[str, Any]], *, training_mode: TrainingMode
) -> None:
    """Reject orphan sessions, mixed modes and duplicated model-call ownership.

    Subagent/plain histories are append-only: each session owns exactly one
    segment zero. Parallel depth-one children retain independent session and
    request identities. Empty sessions may be retained for evidence.
    """
    if training_mode not in TRAINING_MODES or not metadata:
        raise ValueError("Session traces require a known mode and nonempty metadata")
    sessions: set[str] = set()
    root_ids: set[str] = set()
    parent_ids: set[str] = set()
    request_ids: set[str] = set()
    spawn_ids: set[str] = set()
    for tag in metadata:
        if tag.get("training_mode") != training_mode:
            raise ValueError("Session traces changed the configured training mode")
        session = tag.get("session_id")
        root = tag.get("root_session_id")
        if (
            not isinstance(session, str)
            or not session
            or not isinstance(root, str)
            or not root
        ):
            raise ValueError("Session traces require explicit session/root identities")
        root_ids.add(root)
        if training_mode != "compaction" and session in sessions:
            raise ValueError("Append-only sessions must have exactly one trace")
        sessions.add(session)
        role = tag.get("session_role")
        parent = tag.get("parent_session_id")
        spawn = tag.get("spawn_call_id")
        if role == "parent":
            if parent not in (None, "") or spawn not in (None, "") or session != root:
                raise ValueError("Parent trace has inconsistent ancestry")
            parent_ids.add(session)
        elif role == "child" and training_mode == "subagent":
            if parent != root or session == root:
                raise ValueError("Child trace must be a direct descendant of the root")
            if not isinstance(spawn, str) or not spawn or spawn in spawn_ids:
                raise ValueError("Each child needs a unique spawn_call_id")
            spawn_ids.add(spawn)
        else:
            raise ValueError("Child traces are only valid in subagent mode")
        if training_mode != "compaction":
            segment = tag.get("segment_index")
            if type(segment) not in (int, str) or segment not in (0, "0"):
                raise ValueError("Append-only sessions require segment zero")
            if tag.get("segment_boundary_reason") not in (None, "") or tag.get(
                "is_compaction_summary"
            ):
                raise ValueError("Compaction is forbidden in plain/subagent mode")
        owned = tag.get("request_ids")
        if isinstance(owned, str):
            owned = json.loads(owned)
        if not isinstance(owned, list) or not owned:
            raise ValueError("Session trace is missing model-call request ownership")
        for request in owned:
            if not isinstance(request, str) or not request or request in request_ids:
                raise ValueError("Duplicated or invalid model-call request ownership")
            request_ids.add(request)
    if len(root_ids) != 1 or parent_ids != root_ids:
        raise ValueError("Session traces must contain exactly one common parent/root")


def validate_tagged_session_rollout(metadata: Sequence[Mapping[str, Any]]) -> None:
    """Validate new session tags on publication/restore; retain legacy traces."""
    modes = {tag.get("training_mode") for tag in metadata}
    if modes == {None}:
        return
    if len(modes) != 1 or next(iter(modes)) not in TRAINING_MODES:
        raise ValueError("Logical rollout contains mixed or unknown training modes")
    validate_session_trace_metadata(metadata, training_mode=next(iter(modes)))


def bind_session_generation_spans(
    metadata: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Bind actual echoed target spans to immutable, independently captured calls.

    Zero-token calls retain request ownership but do not create a training span.
    Output ordering is checked against the ordered raw-call manifest; duplicated
    copies of a child's return cannot masquerade as another generated output.
    """
    calls = metadata.get("generation_calls")
    requests = metadata.get("request_ids")
    if isinstance(calls, str):
        calls = json.loads(calls)
    if isinstance(requests, str):
        requests = json.loads(requests)
    if not isinstance(calls, list) or not calls:
        raise ValueError(
            "Subagent trace requires an immutable generation-call manifest"
        )
    if any(not isinstance(call, Mapping) for call in calls):
        raise ValueError("Malformed generation-call manifest")
    if [call.get("request_id") for call in calls] != requests:
        raise ValueError("Generation-call manifest disagrees with request ownership")
    for call in calls:
        if any(
            type(call.get(key)) is not int or call[key] < 0
            for key in ("prompt_token_count", "generation_token_count")
        ):
            raise ValueError("Generation-call manifest has invalid token counts")
    generated_calls = [call for call in calls if call["generation_token_count"]]
    if len(generated_calls) != len(outputs):
        raise ValueError("Session echo is missing or duplicates generated model calls")
    spans = []
    for output_index, (call, output) in enumerate(
        zip(generated_calls, outputs, strict=True)
    ):
        start = len(output["prompt_token_ids"])
        tokens = output["generation_token_ids"]
        digest = hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            start != call["prompt_token_count"]
            or len(tokens) != call["generation_token_count"]
            or digest != call.get("generation_token_ids_sha256")
        ):
            raise ValueError("Session echo differs from its captured generation call")
        spans.append(
            {
                "output_index": output_index,
                "request_id": call["request_id"],
                "target_start": start,
                "target_end": start + len(tokens),
                "generation_token_ids_sha256": digest,
            }
        )
    return spans
