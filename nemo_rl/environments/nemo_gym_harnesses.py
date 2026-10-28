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

"""Route typed Gym tasks to session harnesses without changing prompt groups."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemo_rl.experience.failures import GymTransportError, RolloutDataFailure
from nemo_rl.experience.interfaces import (
    NEMO_GYM_GROUP_ID_KEY,
    NEMO_GYM_ROLLOUT_INDEX_KEY,
)


@dataclass(frozen=True)
class HarnessRoute:
    """One resolved Gym environment and the agent/resources pair it owns."""

    environment_name: str
    agent_name: str
    resources_name: str


def parse_harness_pool(value: object) -> list[str] | None:
    """Validate the optional ordered list of Gym environment names."""
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(name, str) or not name.strip() for name in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            "env.nemo_gym.harness_pool must be a nonempty list of distinct environment names"
        )
    return list(value)


def resolve_harness_pool(
    names: list[str], global_config: Mapping[str, Any]
) -> tuple[HarnessRoute, ...]:
    """Resolve a pool of single-agent environments sharing one resources server.

    Args:
        names: Ordered, distinct environment instance names.
        global_config: Gym's resolved global configuration.

    Returns:
        Validated routes in the configured order.
    """
    parse_harness_pool(names)
    routes = []
    for name in names:
        entry = global_config.get(name, {})
        servers = entry.get("environment_servers", {})
        if set(servers) != {"single_agent_turn"}:
            raise ValueError(
                f"Harness environment {name!r} must use typed single_agent_turn"
            )
        config = servers["single_agent_turn"]
        if config.get("agent_server_pool") is not None:
            raise ValueError(
                f"Harness environment {name!r} must bind exactly one agent"
            )
        agent = config.get("agent_server") or {}
        resources = config.get("resources_server") or {}
        if (
            agent.get("type") != "responses_api_agents"
            or resources.get("type") != "resources_servers"
        ):
            raise ValueError(
                f"Harness environment {name!r} has invalid agent/resources references"
            )
        agent_name, resources_name = agent.get("name"), resources.get("name")
        if not isinstance(agent_name, str) or not isinstance(resources_name, str):
            raise ValueError(
                f"Harness environment {name!r} must name its agent and resources server"
            )
        agent_servers = global_config.get(agent_name, {}).get("responses_api_agents")
        if not agent_servers:
            raise ValueError(
                f"Harness environment {name!r} references missing agent {agent_name!r}"
            )
        if len(agent_servers) != 1 or next(iter(agent_servers.values())).get(
            "model_server"
        ) != {
            "type": "responses_api_models",
            "name": "policy_model",
        }:
            raise ValueError(
                f"Harness agent {agent_name!r} must have exactly one configuration "
                "using model_server responses_api_models:policy_model for training capture"
            )
        if not global_config.get(resources_name, {}).get("resources_servers"):
            raise ValueError(
                f"Harness environment {name!r} references missing resources {resources_name!r}"
            )
        routes.append(HarnessRoute(name, agent_name, resources_name))
    if len({route.resources_name for route in routes}) != 1:
        raise ValueError(
            "Harness environments must share one resources server and verifier"
        )
    if len({route.agent_name for route in routes}) != len(routes):
        raise ValueError("Harness environments must bind distinct agent instances")
    return tuple(routes)


def responses_create_params(row: dict[str, Any]) -> dict[str, Any]:
    """Return mutable model parameters from a flat row or typed Gym task."""
    container = (
        row["task_input"]
        if isinstance(row.get("task_id"), dict) and "task_input" in row
        else row
    )
    params = (
        container.get("responses_create_params")
        if isinstance(container, dict)
        else None
    )
    if not isinstance(params, dict):
        raise ValueError("Each Gym task must contain a responses_create_params object")
    return params


def route_harness_rows(
    rows: list[dict[str, Any]], routes: tuple[HarnessRoute, ...]
) -> None:
    """Stamp per-rollout destinations using durable identity, never physical attempts.

    The group ID distinguishes repeated visits to a task. Its generation index
    survives sibling retries, while the capture ID can change on each attempt.
    Pool order must remain fixed when resuming a training run.
    """
    if not routes:
        raise ValueError("Harness routing requires a nonempty resolved pool")
    for row in rows:
        task = row.get("task_id")
        if not isinstance(task, dict) or not isinstance(row.get("task_input"), dict):
            raise ValueError("harness_pool requires typed task_id and task_input rows")
        taskset, task_id = task.get("taskset"), task.get("task_id")
        group = row.get(NEMO_GYM_GROUP_ID_KEY)
        index = row.get(NEMO_GYM_ROLLOUT_INDEX_KEY)
        if any(
            not isinstance(value, str) or not value
            for value in (taskset, task_id, group)
        ):
            raise ValueError(
                "Harness routing requires nonempty taskset, task_id, and durable group ID"
            )
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(
                "Harness routing requires a nonnegative logical rollout index"
            )
        capture_id = row.get("_ng_rollout_id")
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError(
                "Harness routing requires the controller's token-capture rollout ID"
            )
        responses_create_params(row)
        key = ["nemo-rl-harness-v1", taskset, task_id, group, index]
        digest = hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
        ).digest()
        route = routes[int.from_bytes(digest, "big") % len(routes)]
        source = row.get("task_source")
        if source is not None and source != route.resources_name:
            raise ValueError(
                f"Task source {source!r} does not match harness resources {route.resources_name!r}"
            )
        row["_ng_environment_server"] = route.environment_name
        row["agent_ref"] = {"type": "responses_api_agents", "name": route.agent_name}
        row["task_source"] = route.resources_name


def unwrap_harness_result(
    row: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Validate episode identity and expose its scored result to RL capture.

    A typed failure is an exception, not an unscored result with reward zero.
    """
    # Gym is an optional dependency outside the Gym rollout path.
    from nemo_gym.episode_types import EpisodeId, TaskId
    from nemo_gym.single_agent_turn_types import SingleAgentTurnResponse

    episode = SingleAgentTurnResponse.model_validate(result)
    if episode.episode_id != EpisodeId.from_capture_key(row["_ng_rollout_id"]):
        raise ValueError("Harness result has a different capture/episode identity")
    if episode.task_id != TaskId.model_validate(row["task_id"]):
        raise ValueError("Harness result has a different task identity")
    if episode.failure is not None:
        failure = episode.failure
        message = f"Harness environment {row['_ng_environment_server']!r} failed: {failure.failure_reason}"
        if failure.terminal:
            raise RolloutDataFailure(message)
        raise GymTransportError(message)
    assert episode.result is not None
    scored = episode.result.model_dump(mode="json")
    agent_name = row["agent_ref"]["name"]
    if scored.get("ng_agent_server") not in (None, agent_name):
        raise ValueError("Harness result disagrees with the selected agent")
    scored["ng_agent_server"] = agent_name
    scored["ng_environment_server"] = row["_ng_environment_server"]
    return scored
