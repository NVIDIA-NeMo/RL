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

"""Helpers shared by the Single-Controller functional recovery hooks.

The hooks run inside the controller actor, so this module is imported by its
package path, which needs the project root on ``PYTHONPATH`` as the base
functional scripts set it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_event(events_path: Path, event: str, **fields: Any) -> None:
    """Append one JSON event line for the functional test to assert on."""
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")


def find_group(recovery_ledger: Any, rollout_ids: list[str], *, hook: str) -> Any:
    """Return the one recovery-ledger group that owns any of ``rollout_ids``."""
    matches = [
        group
        for group in recovery_ledger.groups()
        if set(rollout_ids).intersection(group.gate_rollout_ids)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{hook} recovery hook could not uniquely resolve rollout IDs to one "
            f"ledger group: ids={rollout_ids!r}, "
            f"matches={[group.group_id for group in matches]!r}"
        )
    return matches[0]
