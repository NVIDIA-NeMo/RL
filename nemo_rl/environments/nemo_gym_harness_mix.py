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

"""Assign one weighted-random harness to each prompt group of a taskset.

GRPO compares a prompt group's generations against the group mean, so every
generation in a group must use the same harness: a group that mixes harnesses
would train on harness differences. ``env.nemo_gym.harness_mix`` names, for
each taskset, the Environment Server that runs it with each harness, plus one
set of harness weights for the run. A group's harness is a hash of its durable
group ID mapped onto those weights. Groups therefore spread over the harnesses
in proportion to the weights, and a retried or restored group keeps its
harness without any stored state.

This module is imported on the driver, where Gym is not installed.
"""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

HARNESS_MIX_CONFIG_KEY = "harness_mix"
HARNESS_RESULT_KEY = "_ng_harness"

# Versions the hash so a future selection rule cannot silently reassign
# groups restored from an older run.
_SELECTION_DOMAIN = b"nemo-rl-harness-mix-v1\0"
_HASH_SPAN = float(2**64)
_CONFIG_KEYS = frozenset({"weights", "environment_servers"})


@dataclass(frozen=True)
class HarnessChoice:
    """The harness selected for one prompt group and the server that runs it."""

    harness: str
    environment_server: str


@dataclass(frozen=True)
class HarnessMix:
    """Run-wide harness weights and each taskset's harness-to-server table.

    Both fields are sorted tuples, so the mix is hashable and two configs that
    differ only in key order select identically.

    Attributes:
        weights: ``(harness, weight)`` pairs. A weight of zero disables the harness.
        environment_servers: ``(taskset, ((harness, environment_server), ...))``
            pairs.
    """

    weights: tuple[tuple[str, float], ...]
    environment_servers: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]

    def covers(self, taskset: str) -> bool:
        """Return whether the mix chooses the harness for this taskset."""
        return any(name == taskset for name, _ in self.environment_servers)

    def environment_servers_for(self, taskset: str) -> tuple[str, ...]:
        """Return the servers a taskset's groups can be sent to."""
        return tuple(server for _, server, _ in self.candidates(taskset))

    @property
    def all_environment_servers(self) -> frozenset[str]:
        """Return every server some group can be sent to."""
        return frozenset(
            server
            for taskset, _ in self.environment_servers
            for server in self.environment_servers_for(taskset)
        )

    def select(self, taskset: str, group_id: str) -> HarnessChoice:
        """Choose the harness for one prompt group.

        Args:
            taskset: The group's ``task_id.taskset``.
            group_id: The durable prompt-group ID shared by all its generations.

        Returns:
            The selected harness and its Environment Server for this taskset.
        """
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(
                f"Taskset {taskset!r} uses env.nemo_gym.harness_mix, which needs "
                "each row's prompt-group ID to choose its harness"
            )
        candidates = self.candidates(taskset)
        digest = hashlib.sha256(_SELECTION_DOMAIN + group_id.encode()).digest()
        point = (
            int.from_bytes(digest[:8], "big")
            / _HASH_SPAN
            * sum(weight for _, _, weight in candidates)
        )
        for harness, server, weight in candidates:
            point -= weight
            if point < 0:
                return HarnessChoice(harness, server)
        # Float rounding can leave a point at the very top of the range.
        harness, server, _ = candidates[-1]
        return HarnessChoice(harness, server)

    def candidates(self, taskset: str) -> list[tuple[str, str, float]]:
        """Return the taskset's ``(harness, server, weight)`` with positive weight."""
        weights = dict(self.weights)
        for name, table in self.environment_servers:
            if name == taskset:
                return [
                    (harness, server, weights[harness])
                    for harness, server in table
                    if weights[harness] > 0
                ]
        raise ValueError(f"env.nemo_gym.harness_mix has no taskset {taskset!r}")


def parse_harness_mix(value: object) -> HarnessMix | None:
    """Validate ``env.nemo_gym.harness_mix``.

    Without ``weights`` every harness gets weight 1. With them, every harness
    in a taskset's table needs a weight, every weight must name a harness some
    taskset uses, and each taskset needs a harness with a positive weight.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            "env.nemo_gym.harness_mix must be a mapping with environment_servers "
            "and optional weights"
        )
    unknown = sorted(str(key) for key in value if key not in _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"env.nemo_gym.harness_mix has unknown keys {unknown}")

    raw_tables = value.get("environment_servers")
    if not isinstance(raw_tables, Mapping) or not raw_tables:
        raise ValueError(
            "env.nemo_gym.harness_mix.environment_servers must map each taskset "
            "to its {harness: Environment Server} table"
        )
    tables: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    harness_by_server: dict[str, str] = {}
    for taskset, raw_table in raw_tables.items():
        if not _is_name(taskset) or not isinstance(raw_table, Mapping) or not raw_table:
            raise ValueError(
                "env.nemo_gym.harness_mix.environment_servers must map nonempty "
                "taskset names to nonempty {harness: Environment Server} tables"
            )
        pairs: list[tuple[str, str]] = []
        for harness, server in raw_table.items():
            if not _is_name(harness) or not _is_name(server):
                raise ValueError(
                    f"env.nemo_gym.harness_mix.environment_servers.{taskset} must "
                    "map nonempty harness names to nonempty Environment Server names"
                )
            listed_as = harness_by_server.setdefault(server, harness)
            if listed_as != harness:
                raise ValueError(
                    f"Environment Server {server!r} is listed as harness "
                    f"{listed_as!r} and as harness {harness!r}; a server runs one harness"
                )
            pairs.append((harness, server))
        tables.append((taskset, tuple(sorted(pairs))))
    harnesses = {harness for _, table in tables for harness, _ in table}

    raw_weights = value.get("weights")
    if raw_weights is None:
        weights = dict.fromkeys(harnesses, 1.0)
    else:
        if not isinstance(raw_weights, Mapping) or not raw_weights:
            raise ValueError(
                "env.nemo_gym.harness_mix.weights must map harness names to weights"
            )
        weights = {}
        for harness, weight in raw_weights.items():
            if (
                not _is_name(harness)
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(
                    "env.nemo_gym.harness_mix.weights must map harness names to "
                    f"finite numbers >= 0; got {harness!r}: {weight!r}"
                )
            weights[harness] = float(weight)
        unweighted = sorted(harnesses - set(weights))
        if unweighted:
            raise ValueError(
                f"env.nemo_gym.harness_mix.weights has no weight for harnesses {unweighted}"
            )
        unused = sorted(set(weights) - harnesses)
        if unused:
            raise ValueError(
                f"env.nemo_gym.harness_mix.weights names harnesses {unused} that no "
                "taskset in environment_servers uses"
            )
    for taskset, table in tables:
        if not any(weights[harness] > 0 for harness, _ in table):
            raise ValueError(
                f"Taskset {taskset!r} has no harness with a positive weight in "
                "env.nemo_gym.harness_mix"
            )
    return HarnessMix(
        weights=tuple(sorted(weights.items())),
        environment_servers=tuple(sorted(tables)),
    )


def validate_harness_mix_servers(
    harness_mix: HarnessMix, global_config: Mapping[str, Any]
) -> None:
    """Require each taskset's servers to differ only in their harness.

    The servers a taskset's groups can be sent to must be Environment Servers
    of one type that score its tasks the same way: the same resources server
    for ``single_agent_turn``, and a Harbor server must serve the taskset itself.
    Servers of zero-weight harnesses are never used, so they are not checked.

    Args:
        harness_mix: The parsed mix.
        global_config: Gym's resolved global configuration.

    Raises:
        ValueError: A server is missing, is not an Environment Server, or would
            run or score the taskset differently from the others.
    """
    for taskset, _ in harness_mix.environment_servers:
        types: set[str] = set()
        verifiers: set[str | None] = set()
        for harness, server, _ in harness_mix.candidates(taskset):
            entry = global_config.get(server)
            servers = (
                entry.get("environment_servers") if isinstance(entry, Mapping) else None
            )
            if not isinstance(servers, Mapping) or len(servers) != 1:
                raise ValueError(
                    f"env.nemo_gym.harness_mix runs taskset {taskset!r} with harness "
                    f"{harness!r} on {server!r}, which is not an Environment Server "
                    "in the Gym config"
                )
            server_type, server_config = next(iter(servers.items()))
            types.add(str(server_type))
            if not isinstance(server_config, Mapping):
                raise ValueError(
                    f"Environment Server {server!r} must have a configuration mapping"
                )
            if server_type == "harbor":
                served = server_config.get("tasksets")
                if not isinstance(served, Mapping) or taskset not in served:
                    raise ValueError(
                        f"Harbor Environment Server {server!r} does not serve "
                        f"taskset {taskset!r}"
                    )
            resources = server_config.get("resources_server")
            verifiers.add(
                resources.get("name") if isinstance(resources, Mapping) else None
            )
        if len(types) != 1:
            raise ValueError(
                f"Taskset {taskset!r} mixes Environment Server types {sorted(types)} "
                "in env.nemo_gym.harness_mix; its harnesses must share one type"
            )
        if len(verifiers) != 1:
            raise ValueError(
                f"Taskset {taskset!r} is scored by different resources servers "
                f"{sorted(map(str, verifiers))} across its harnesses in "
                "env.nemo_gym.harness_mix; every harness must use the same verifier"
            )


def harness_metric_namespace(
    attribution_key: str, full_result: Mapping[str, Any] | None
) -> str:
    """Namespace a result's metrics by taskset and, when mixed, by harness."""
    harness = (full_result or {}).get(HARNESS_RESULT_KEY)
    if isinstance(harness, str) and harness:
        return f"{attribution_key}/{harness}"
    return attribution_key


def _is_name(value: object) -> bool:
    return isinstance(value, str) and bool(value)
