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

"""Access task inputs from Gym's materialized task rows and flat rows."""

import json
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict, cast


class NemoGymHarnessRouting(TypedDict):
    """Gym's declared task identity and runtime harness execution evidence."""

    benchmark: str
    dataset: str
    env: str
    task_type: Literal["agentic", "non_agentic"]
    selected_harness: str | None
    executed_harness: str | None
    routing_version: str
    routing_seed: int
    group_id: str
    group_index: NotRequired[int | None]


def validate_nemo_gym_harness_routing(value: object) -> NemoGymHarnessRouting:
    """Validate runtime metadata without inferring task or harness identity."""
    if not isinstance(value, Mapping):
        raise TypeError("harness_routing must be a mapping")
    result = dict(value)
    for name in ("benchmark", "dataset", "env", "routing_version", "group_id"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"harness_routing.{name} must be a nonempty string")
    if result.get("task_type") not in ("agentic", "non_agentic"):
        raise ValueError("harness_routing.task_type must be agentic or non_agentic")
    if type(result.get("routing_seed")) is not int:
        raise ValueError("harness_routing.routing_seed must be an integer")
    for name in ("selected_harness", "executed_harness"):
        harness = result[name]
        if harness is not None and (not isinstance(harness, str) or not harness):
            raise ValueError(
                f"harness_routing.{name} must be a nonempty string or None"
            )
    index = result.get("group_index")
    if index is not None and (type(index) is not int or index < 0):
        raise ValueError("harness_routing.group_index must be a nonnegative integer")
    if result["task_type"] == "non_agentic":
        if (
            result["selected_harness"] is not None
            or result["executed_harness"] is not None
        ):
            raise ValueError("non_agentic tasks must have null harnesses")
    elif index is None:
        raise ValueError(
            "agentic harness routing requires its prepared benchmark group_index"
        )
    if (
        result["executed_harness"] is not None
        and result["executed_harness"] != result["selected_harness"]
    ):
        raise ValueError("executed_harness does not match selected_harness")
    return cast(NemoGymHarnessRouting, result)


def get_nemo_gym_harness_routing(
    row: Mapping[str, Any],
) -> NemoGymHarnessRouting | None:
    """Read routing reported by a result or retained environment extras."""
    value = row.get("harness_routing")
    if value is None:
        full_result = row.get("full_result")
        if isinstance(full_result, Mapping):
            value = full_result.get("harness_routing")
    return None if value is None else validate_nemo_gym_harness_routing(value)


def bind_nemo_gym_harness_group(
    row: dict[str, Any], *, group_id: str
) -> dict[str, Any]:
    """Bind the durable group before replica expansion without changing the dataset row."""
    task_input = get_nemo_gym_task_input(row)
    value = task_input.get("harness_routing")
    if value is None:
        return row
    if not isinstance(value, Mapping):
        raise TypeError("task_input.harness_routing must be a mapping")
    routing = dict(value)
    previous = routing.get("group_id")
    if previous is not None and previous != group_id:
        raise ValueError(
            "prepared harness group_id conflicts with the durable rollout group"
        )
    routing["group_id"] = group_id
    validated = validate_nemo_gym_harness_routing(routing)
    result = dict(row)
    result["harness_routing"] = validated
    if is_nemo_gym_task(row):
        result["task_input"] = dict(task_input, harness_routing=validated)
    return result


def validate_nemo_gym_harness_result(
    result: Mapping[str, Any], request: dict[str, Any]
) -> NemoGymHarnessRouting | None:
    """Require execution evidence and stable declared identity for opted-in tasks."""
    expected = get_nemo_gym_task_input(request).get("harness_routing")
    routing = get_nemo_gym_harness_routing(result)
    if expected is not None:
        if routing is None:
            raise ValueError(
                "Gym result omitted requested harness_routing execution metadata"
            )
        for key in (
            "benchmark",
            "dataset",
            "env",
            "task_type",
            "selected_harness",
            "routing_version",
            "routing_seed",
            "group_id",
            "group_index",
        ):
            if routing.get(key) != expected.get(key):
                raise ValueError(f"Gym changed harness_routing.{key} during execution")
    return routing


def serialize_nemo_gym_harness_routing(routing: NemoGymHarnessRouting) -> str:
    """Encode structured identity in TQ's primitive row tags."""
    return json.dumps(routing, sort_keys=True, separators=(",", ":"))


def is_nemo_gym_task(row: Mapping[str, Any]) -> bool:
    """Match Gym's materialized-task routing discriminator.

    Flat rows may also contain a ``task_id`` field. A structured identity
    together with ``task_input`` selects the episode protocol, even when the
    identity is invalid. Full protocol validation remains Gym's responsibility.
    """
    return isinstance(row.get("task_id"), Mapping) and "task_input" in row


def get_nemo_gym_attribution_key(row: Mapping[str, Any]) -> str | None:
    """Return a taskset for materialized tasks or the resolved agent for flat rows.

    Processed rollout results retain this identity in ``attribution_key`` when
    the original task envelope is no longer present.
    """
    if is_nemo_gym_task(row):
        key = row["task_id"].get("taskset")
    else:
        agent_ref = row.get("agent_ref")
        key = agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
        if not isinstance(key, str) or not key:
            key = row.get("attribution_key")
    return key if isinstance(key, str) and key else None


def get_nemo_gym_task_input(row: dict[str, Any]) -> dict[str, Any]:
    """Return mutable input from a materialized task row or the entire flat row.

    Callers can update Responses parameters in place while retaining materialized
    task identity, opaque task data, and RL metadata in the original envelope.

    Raises:
        TypeError: If a materialized task's ``task_input`` is not a dictionary.
    """
    if not is_nemo_gym_task(row):
        return row
    task_input = row["task_input"]
    if not isinstance(task_input, dict):
        raise TypeError("NeMo-Gym materialized task_input must be a dict")
    return task_input
