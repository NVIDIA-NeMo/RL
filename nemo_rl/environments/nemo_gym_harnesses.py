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

from nemo_rl.experience.failures import RolloutDataFailure
from nemo_rl.experience.interfaces import (
    NEMO_GYM_GROUP_ID_KEY,
    NEMO_GYM_ROLLOUT_INDEX_KEY,
)


@dataclass(frozen=True)
class HarnessRoute:
    """One resolved Gym environment, its agent, and its task source."""

    environment_name: str
    agent_name: str
    resources_name: str | None
    harbor_tasksets: tuple[str, ...] = ()


type HarnessPool = list[str] | dict[str, list[str]]
type ResolvedHarnessPool = (
    tuple[HarnessRoute, ...] | dict[str, tuple[HarnessRoute, ...]]
)


def parse_harness_pool(value: object) -> HarnessPool | None:
    """Validate ordered environment lists, optionally keyed by taskset."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        if not value or any(
            not isinstance(taskset, str) or not taskset.strip() for taskset in value
        ):
            raise ValueError(
                "env.nemo_gym.harness_pool must map nonempty taskset names to "
                "nonempty lists of distinct environment names"
            )
        return {
            taskset: _parse_harness_names(names) for taskset, names in value.items()
        }
    return _parse_harness_names(value)


def _parse_harness_names(value: object) -> list[str]:
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
    names: HarnessPool, global_config: Mapping[str, Any]
) -> ResolvedHarnessPool:
    """Resolve each pool against a shared verifier or Harbor taskset.

    Args:
        names: Ordered environment names, optionally keyed by taskset.
        global_config: Gym's resolved global configuration.

    Returns:
        Validated routes in configured order, preserving taskset keys.
    """
    parsed = parse_harness_pool(names)
    if isinstance(parsed, dict):
        return {
            taskset: _resolve_harness_routes(pool, global_config, taskset=taskset)
            for taskset, pool in parsed.items()
        }
    if parsed is None:
        raise ValueError("Harness resolution requires a configured pool")
    return _resolve_harness_routes(parsed, global_config, taskset=None)


def _resolve_harness_routes(
    names: list[str],
    global_config: Mapping[str, Any],
    *,
    taskset: str | None,
) -> tuple[HarnessRoute, ...]:
    routes = []
    harbor_tasksets = []
    for name in names:
        entry = global_config.get(name, {})
        servers = entry.get("environment_servers", {})
        if set(servers) not in ({"single_agent_turn"}, {"harbor"}):
            raise ValueError(
                f"Harness environment {name!r} must use typed single_agent_turn or harbor"
            )
        kind = next(iter(servers))
        config = servers[kind]
        if config.get("agent_server_pool") is not None:
            raise ValueError(
                f"Harness environment {name!r} must bind exactly one agent"
            )
        agent = config.get("agent_server") or {}
        resources = config.get("resources_server") or {}
        if (
            agent.get("type") != "responses_api_agents"
            or kind == "single_agent_turn"
            and resources.get("type") != "resources_servers"
        ):
            raise ValueError(
                f"Harness environment {name!r} has invalid agent/resources references"
            )
        agent_name, resources_name = agent.get("name"), resources.get("name")
        if (
            not isinstance(agent_name, str)
            or not agent_name
            or (
                kind == "single_agent_turn"
                and (not isinstance(resources_name, str) or not resources_name)
            )
        ):
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
        if kind == "single_agent_turn":
            if not global_config.get(resources_name, {}).get("resources_servers"):
                raise ValueError(
                    f"Harness environment {name!r} references missing resources {resources_name!r}"
                )
            routes.append(HarnessRoute(name, agent_name, resources_name))
        else:
            tasksets = config.get("tasksets")
            if (
                not isinstance(tasksets, Mapping)
                or not tasksets
                or any(not isinstance(key, str) or not key for key in tasksets)
            ):
                raise ValueError(
                    f"Harbor harness environment {name!r} must configure tasksets"
                )
            if taskset is not None and taskset not in tasksets:
                raise ValueError(
                    f"Harbor harness environment {name!r} does not serve taskset {taskset!r}"
                )
            harbor_tasksets.append(tasksets if taskset is None else tasksets[taskset])
            routes.append(HarnessRoute(name, agent_name, None, tuple(tasksets)))
    if len({route.resources_name for route in routes}) != 1:
        raise ValueError(
            "Harness environments in each pool must share one resources server and "
            "verifier, or all use Harbor"
        )
    if harbor_tasksets and any(
        value != harbor_tasksets[0] for value in harbor_tasksets[1:]
    ):
        raise ValueError("Harbor harness environments must serve the same taskset data")
    if len({route.agent_name for route in routes}) != len(routes):
        raise ValueError("Harness environments must bind distinct agent instances")
    return tuple(routes)


def route_harness_rows(rows: list[dict[str, Any]], routes: ResolvedHarnessPool) -> None:
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
        pool = routes.get(taskset) if isinstance(routes, dict) else routes
        if not pool:
            raise ValueError(f"No harness_pool is configured for taskset {taskset!r}")
        key = ["nemo-rl-harness-v1", taskset, task_id, group, index]
        digest = hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
        ).digest()
        route = pool[int.from_bytes(digest, "big") % len(pool)]
        if route.resources_name is None and taskset not in route.harbor_tasksets:
            raise ValueError(
                f"Harbor harness environment {route.environment_name!r} "
                f"does not serve taskset {taskset!r}"
            )
        source = row.get("task_source")
        if (
            route.resources_name is not None
            and source is not None
            and source != route.resources_name
        ):
            raise ValueError(
                f"Task source {source!r} does not match harness resources {route.resources_name!r}"
            )
        row["_ng_environment_server"] = route.environment_name
        row["agent_ref"] = {"type": "responses_api_agents", "name": route.agent_name}
        if route.resources_name is not None:
            row["task_source"] = route.resources_name
        elif source is None:
            row["task_source"] = taskset


def attribute_harness_result(
    row: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Attach the selected harness to an already validated episode result."""
    agent_name = row["agent_ref"]["name"]
    if result.get("ng_agent_server") not in (None, agent_name):
        raise RolloutDataFailure("Harness result disagrees with the selected agent")
    return {
        **result,
        "ng_agent_server": agent_name,
        "ng_environment_server": row["_ng_environment_server"],
    }
