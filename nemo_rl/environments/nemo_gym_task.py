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

from collections.abc import Mapping
from typing import Any


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
