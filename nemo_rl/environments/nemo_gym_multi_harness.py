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

"""Run each prompt group of a multi-harness taskset on one weighted-random harness.

GRPO compares a prompt group's generations with the group's mean reward, so
every generation in a group must use the same harness: a group that mixed
harnesses would train on the differences between harnesses.

A taskset is routed by exactly one of two settings. Gym's
``environment_server_routes`` maps it to one Environment Server.
``env.nemo_gym.multi_harness.routes`` maps it to several, one per harness, and
``multi_harness.weights`` weights the harnesses for the whole run by the type
of agent each server runs.

At startup the driver reads which agent each listed server runs from Gym's
config, resolves the routes, and sends them to every Gym actor. A group's
server is a hash of its durable group ID mapped onto the weights. The driver
and the actors therefore pick the same server, groups spread over the
harnesses in proportion to the weights, and a retried or restored group keeps
its server without any stored state.

This module is imported on the driver, where Gym is not installed.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MULTI_HARNESS_CONFIG_KEY = "multi_harness"
HARNESS_RESULT_KEY = "_ng_harness"

# Versions the hash so a future selection rule cannot silently reassign
# groups restored from an older run.
_SELECTION_DOMAIN = b"nemo-rl-multi-harness-v1\0"
_HASH_SPAN = float(2**64)
_CONFIG_KEYS = frozenset({"routes", "weights"})


@dataclass(frozen=True)
class MultiHarnessConfig:
    """``env.nemo_gym.multi_harness`` as written, before servers are matched to agents.

    Attributes:
        routes: ``(taskset, (environment_server, ...))`` pairs, sorted.
        weights: ``(agent type, weight)`` pairs, sorted, or None to weight
            every agent 1.
    """

    routes: tuple[tuple[str, tuple[str, ...]], ...]
    weights: tuple[tuple[str, float], ...] | None

    @property
    def tasksets(self) -> frozenset[str]:
        """Return the tasksets these routes cover."""
        return frozenset(taskset for taskset, _ in self.routes)

    @property
    def servers(self) -> frozenset[str]:
        """Return every server the routes list."""
        return frozenset(server for _, servers in self.routes for server in servers)


@dataclass(frozen=True)
class EnvironmentServerInfo:
    """What routing needs to know about one Environment Server from Gym's config.

    Attributes:
        server_type: The Environment Server type, such as ``single_agent_turn``.
        agent: The type of the agent its ``agent_server`` names, such as
            ``opencode_agent``, or None when it names no single agent.
        resources_server: The resources server that scores its tasks, if any.
        tasksets: The tasksets a Harbor server serves; None for other types.
    """

    server_type: str
    agent: str | None
    resources_server: str | None
    tasksets: tuple[str, ...] | None


@dataclass(frozen=True)
class HarnessChoice:
    """The harness selected for one prompt group and the server that runs it."""

    harness: str
    environment_server: str


@dataclass(frozen=True)
class MultiHarnessRoutes:
    """Each multi-harness taskset's servers, the agent each runs, and its weight.

    Attributes:
        candidates: ``(taskset, ((agent, environment_server, weight), ...))``
            pairs, sorted. Agents with weight 0 are left out.
    """

    candidates: tuple[tuple[str, tuple[tuple[str, str, float], ...]], ...]

    def covers(self, taskset: str) -> bool:
        """Return whether these routes choose the server for this taskset."""
        return any(name == taskset for name, _ in self.candidates)

    def servers_for(self, taskset: str) -> tuple[str, ...]:
        """Return the servers a taskset's groups can be sent to."""
        return tuple(server for _, server, _ in self._candidates(taskset))

    @property
    def servers(self) -> frozenset[str]:
        """Return every server some group can be sent to."""
        return frozenset(
            server for _, candidates in self.candidates for _, server, _ in candidates
        )

    def select(self, taskset: str, group_id: str) -> HarnessChoice:
        """Choose the harness for one prompt group.

        Args:
            taskset: The group's ``task_id.taskset``.
            group_id: The durable prompt-group ID shared by all its generations.

        Returns:
            The selected agent type and the Environment Server that runs it.
        """
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(
                f"Taskset {taskset!r} is in env.nemo_gym.multi_harness.routes, which "
                "needs each row's prompt-group ID to choose its harness"
            )
        candidates = self._candidates(taskset)
        digest = hashlib.sha256(_SELECTION_DOMAIN + group_id.encode()).digest()
        point = (
            int.from_bytes(digest[:8], "big")
            / _HASH_SPAN
            * sum(weight for _, _, weight in candidates)
        )
        for agent, server, weight in candidates:
            point -= weight
            if point < 0:
                return HarnessChoice(agent, server)
        # Float rounding can leave a point at the very top of the range.
        agent, server, _ = candidates[-1]
        return HarnessChoice(agent, server)

    def _candidates(self, taskset: str) -> tuple[tuple[str, str, float], ...]:
        for name, candidates in self.candidates:
            if name == taskset:
                return candidates
        raise ValueError(
            f"env.nemo_gym.multi_harness.routes has no taskset {taskset!r}"
        )


def parse_multi_harness(value: object) -> MultiHarnessConfig | None:
    """Validate the shape of ``env.nemo_gym.multi_harness``.

    ``routes`` maps each taskset to a list of distinct Environment Server
    names. ``weights`` is optional and maps agent types to finite numbers of
    at least 0. Which agent each server runs is only known once Gym has
    started; :func:`resolve_multi_harness` checks the rest then.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            "env.nemo_gym.multi_harness must be a mapping with routes and "
            "optional weights"
        )
    unknown = sorted(str(key) for key in value if key not in _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"env.nemo_gym.multi_harness has unknown keys {unknown}")

    raw_routes = value.get("routes")
    if not isinstance(raw_routes, Mapping) or not raw_routes:
        raise ValueError(
            "env.nemo_gym.multi_harness.routes must map each taskset to a list "
            "of Environment Servers"
        )
    routes: list[tuple[str, tuple[str, ...]]] = []
    for taskset, servers in raw_routes.items():
        if (
            not _is_name(taskset)
            or isinstance(servers, (str, Mapping))
            or not isinstance(servers, Sequence)
            or not servers
            or not all(_is_name(server) for server in servers)
            or len(set(servers)) != len(servers)
        ):
            raise ValueError(
                f"env.nemo_gym.multi_harness.routes.{taskset} must be a nonempty "
                "list of distinct Environment Server names"
            )
        routes.append((taskset, tuple(sorted(servers))))

    raw_weights = value.get("weights")
    weights: tuple[tuple[str, float], ...] | None = None
    if raw_weights is not None:
        if not isinstance(raw_weights, Mapping) or not raw_weights:
            raise ValueError(
                "env.nemo_gym.multi_harness.weights must map agent types to weights"
            )
        parsed: dict[str, float] = {}
        for agent, weight in raw_weights.items():
            if (
                not _is_name(agent)
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(
                    "env.nemo_gym.multi_harness.weights must map agent types to "
                    f"finite numbers >= 0; got {agent!r}: {weight!r}"
                )
            parsed[agent] = float(weight)
        weights = tuple(sorted(parsed.items()))
    return MultiHarnessConfig(routes=tuple(sorted(routes)), weights=weights)


def describe_environment_server(
    name: str, global_config: Mapping[str, Any]
) -> EnvironmentServerInfo | None:
    """Read what routing needs about one Environment Server from Gym's config.

    Args:
        name: The Environment Server entry name.
        global_config: Gym's resolved global configuration.

    Returns:
        The server's description, or None when ``name`` is not an Environment
        Server entry.
    """
    entry = global_config.get(name)
    servers = entry.get("environment_servers") if isinstance(entry, Mapping) else None
    if not isinstance(servers, Mapping) or len(servers) != 1:
        return None
    server_type, server_config = next(iter(servers.items()))
    if not isinstance(server_config, Mapping):
        return None
    agent_ref = server_config.get("agent_server")
    agent_name = agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
    agent_entry = (
        global_config.get(agent_name)
        if isinstance(agent_name, str) and agent_name
        else None
    )
    agent_types = (
        agent_entry.get("responses_api_agents")
        if isinstance(agent_entry, Mapping)
        else None
    )
    resources_ref = server_config.get("resources_server")
    resources_name = (
        resources_ref.get("name") if isinstance(resources_ref, Mapping) else None
    )
    tasksets = server_config.get("tasksets")
    return EnvironmentServerInfo(
        server_type=str(server_type),
        agent=str(next(iter(agent_types)))
        if isinstance(agent_types, Mapping) and len(agent_types) == 1
        else None,
        resources_server=resources_name if _is_name(resources_name) else None,
        tasksets=tuple(str(taskset) for taskset in tasksets)
        if server_type == "harbor" and isinstance(tasksets, Mapping)
        else None,
    )


def resolve_multi_harness(
    config: MultiHarnessConfig, servers: Mapping[str, EnvironmentServerInfo]
) -> MultiHarnessRoutes:
    """Match each listed server to the agent it runs and check the routes.

    The servers of one taskset must run different agents and score the
    taskset the same way: one Environment Server type, the same resources
    server, and for Harbor a server that serves the taskset. With weights,
    every agent the routes use needs a weight and every weight must name one
    of them. Each taskset needs an agent with a positive weight.

    Args:
        config: The parsed ``multi_harness`` block.
        servers: Each listed server's description, reported by the Gym actor
            that runs it.

    Returns:
        The routes that the driver and every Gym actor select from.

    Raises:
        ValueError: The routes list a server that is not an Environment Server
            running one agent, or that would run or score its taskset
            differently from the taskset's other servers.
    """
    agents_by_taskset: dict[str, list[tuple[str, str]]] = {}
    for taskset, names in config.routes:
        agents: list[tuple[str, str]] = []
        types: set[str] = set()
        verifiers: set[str | None] = set()
        for server in names:
            info = servers.get(server)
            if info is None or info.agent is None:
                raise ValueError(
                    f"env.nemo_gym.multi_harness.routes.{taskset} lists {server!r}, "
                    "which is not an Environment Server that runs one agent"
                )
            if info.server_type == "harbor" and taskset not in (info.tasksets or ()):
                raise ValueError(
                    f"Harbor Environment Server {server!r} does not serve taskset "
                    f"{taskset!r}"
                )
            types.add(info.server_type)
            verifiers.add(info.resources_server)
            agents.append((info.agent, server))
        repeated = sorted(
            {agent for agent, _ in agents if [a for a, _ in agents].count(agent) > 1}
        )
        if repeated:
            raise ValueError(
                f"Taskset {taskset!r} lists several servers that run agents "
                f"{repeated}; each server of a taskset must run a different agent"
            )
        if len(types) != 1:
            raise ValueError(
                f"Taskset {taskset!r} mixes Environment Server types {sorted(types)}; "
                "its servers must share one type"
            )
        if len(verifiers) != 1:
            raise ValueError(
                f"Taskset {taskset!r} is scored by different resources servers "
                f"{sorted(map(str, verifiers))}; its servers must use the same verifier"
            )
        agents_by_taskset[taskset] = agents

    used = {agent for agents in agents_by_taskset.values() for agent, _ in agents}
    if config.weights is None:
        weights = dict.fromkeys(used, 1.0)
    else:
        weights = dict(config.weights)
        unweighted = sorted(used - set(weights))
        if unweighted:
            raise ValueError(
                f"env.nemo_gym.multi_harness.weights has no weight for agents "
                f"{unweighted}"
            )
        unused = sorted(set(weights) - used)
        if unused:
            raise ValueError(
                f"env.nemo_gym.multi_harness.weights names agents {unused} that no "
                "server in routes runs"
            )

    candidates: list[tuple[str, tuple[tuple[str, str, float], ...]]] = []
    for taskset, agents in sorted(agents_by_taskset.items()):
        enabled = tuple(
            sorted(
                (agent, server, weights[agent])
                for agent, server in agents
                if weights[agent] > 0
            )
        )
        if not enabled:
            raise ValueError(
                f"Taskset {taskset!r} has no agent with a positive weight in "
                "env.nemo_gym.multi_harness.weights"
            )
        candidates.append((taskset, enabled))
    return MultiHarnessRoutes(candidates=tuple(candidates))


def harness_metric_namespace(
    attribution_key: str, full_result: Mapping[str, Any] | None
) -> str:
    """Namespace a result's metrics by taskset and, for multi-harness, by agent."""
    harness = (full_result or {}).get(HARNESS_RESULT_KEY)
    if isinstance(harness, str) and harness:
        return f"{attribution_key}/{harness}"
    return attribution_key


def _is_name(value: object) -> bool:
    return isinstance(value, str) and bool(value)
