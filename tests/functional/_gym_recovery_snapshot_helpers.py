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


"""Readers shared by the Gym-v2 turn and prefix recovery functional tests.

Both tests inspect the same snapshot layout and controller event log, so a
ledger or manifest format change is fixed here once.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def calendar_sentinel_count(state: dict[str, Any], event_name: str) -> int:
    """Count the selected calendar event in a checkpointed Workplace state."""
    columns = state["calendar"]["_calendar_events"]["columns"]
    names = next(
        column["values"] for column in columns if column["name"] == "event_name"
    )
    return sum(str(name).lower() == event_name.lower() for name in names)


# Written once, by GenerationCutCapture when a restored prefix is admitted.
_RESTORED_PREFIX_LINE = re.compile(
    r"generation prefix restored: rollout_id=(?P<rollout_id>\S+) "
    r"model_call_id=(?P<model_call_id>\S+) "
    r"source_model_call_id=(?P<source_model_call_id>\S+) "
    r"prefix_tokens=(?P<prefix_tokens>\d+) "
    r"prefix_digest=(?P<prefix_digest>[0-9a-f]{64}) "
    r"weight_version_span=\[\d+,\d+\] "
    r"prefix_ids_sha256=(?P<prefix_ids_sha256>[0-9a-f]{64})"
)


def restored_prefixes(log: str) -> list[dict[str, Any]]:
    """Parse every "generation prefix restored" line in a run log."""
    return [
        {**match.groupdict(), "prefix_tokens": int(match["prefix_tokens"])}
        for match in _RESTORED_PREFIX_LINE.finditer(log)
    ]


def ledger_episode_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Map each ``(gym_rollout_id, gym_attempt)`` to its RL ledger sibling."""
    # Deferred: only snapshot readers need torch.
    import torch

    state = torch.load(path, weights_only=True)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for group in state["groups"]:
        for sibling in group["siblings"]:
            attempt = sibling["attempts"][-1]
            generation_index = sibling["generation_index"]
            rollout_id = (
                f"{group['group_id']}_g{generation_index}_a"
                f"{uuid.UUID(bytes=attempt['attempt_uuid']).hex}"
            )
            result[(rollout_id, attempt["gym_attempt"])] = {
                "group_id": group["group_id"],
                "generation_index": generation_index,
                "target_step": group["target_step"],
                "restore_level": group["restore_level"],
                "status": attempt["status"],
                "gym_instance_id": attempt["gym_instance_id"],
            }
    return result


def participant_records(snapshot: Path, kind: str) -> list[tuple[Path, dict[str, Any]]]:
    """Every exported row of one Gym participant kind, with its manifest path."""
    records: list[tuple[Path, dict[str, Any]]] = []
    pattern = f"gym-instances/*/replica-*/gym/{kind}/*/manifest.json"
    for manifest_path in snapshot.glob(pattern):
        manifest = read_json(manifest_path)
        if manifest.get("kind") != kind or manifest.get("record_count", 0) <= 0:
            continue
        records_path = manifest_path.parent / manifest["records_file"]
        for line in records_path.read_text().splitlines():
            if line:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(
                        f"participant row must be an object in {records_path}"
                    )
                records.append((manifest_path, record))
    return records


def matching_siblings(
    events: list[dict[str, Any]], selection: dict[str, Any], event_name: str
) -> list[dict[str, Any]]:
    """The selected episode's siblings in every ``event_name`` controller event."""
    found: list[dict[str, Any]] = []
    for event in events:
        if event["event"] != event_name or event["group_id"] != selection["group_id"]:
            continue
        siblings = event.get("siblings") or [event.get("sibling")]
        found.extend(
            sibling
            for sibling in siblings
            if sibling is not None
            and sibling["gym_rollout_id"] == selection["gym_rollout_id"]
        )
    return found
