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

"""Access task inputs without changing Gym's native or legacy wire format."""

from collections.abc import Mapping
from typing import Any


def is_nemo_gym_task(row: Mapping[str, Any]) -> bool:
    """Match Gym's materialized-task routing discriminator.

    Legacy rows may also contain a ``task_id`` field. Only a structured identity
    with a nonempty taskset and a ``task_input`` field uses the native protocol.
    Full protocol validation remains Gym's responsibility.
    """
    task_id = row.get("task_id")
    if not isinstance(task_id, Mapping) or "task_input" not in row:
        return False
    taskset = task_id.get("taskset")
    return isinstance(taskset, str) and bool(taskset)


def get_nemo_gym_task_input(row: dict[str, Any]) -> dict[str, Any]:
    """Return mutable task input from a native task or the entire legacy row.

    Callers can update Responses parameters in place while retaining native
    task identity, opaque task data, and RL metadata in the original envelope.

    Raises:
        TypeError: If a native task's ``task_input`` is not a dictionary.
    """
    if not is_nemo_gym_task(row):
        return row
    task_input = row["task_input"]
    if not isinstance(task_input, dict):
        raise TypeError("NeMo-Gym materialized task_input must be a dict")
    return task_input
