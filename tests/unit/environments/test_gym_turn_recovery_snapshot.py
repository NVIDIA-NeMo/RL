# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib.util
import json
from pathlib import Path

import pytest


_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_turn_recovery_snapshot.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gym_turn_recovery_snapshot", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


def test_workplace_snapshot_counts_exact_sentinel_row() -> None:
    frame = {
        "columns": [
            "event_id",
            "event_name",
            "participant_email",
            "event_start",
            "duration",
        ],
        "data": [
            [
                "00000001",
                "NeMo RL checkpoint recovery sentinel",
                "checkpoint-recovery@example.com",
                "2025-01-15 10:00:00",
                "30",
            ],
            ["00000002", "Other", "other@example.com", "2025-01-16 10:00:00", "15"],
        ],
    }
    state = {
        "containers": {
            "calendar": {"_calendar_events": json.dumps(frame)},
        }
    }

    assert _HELPER._workplace_sentinel_count(state) == 1


def test_workplace_restore_audit_requires_exactly_once_mutation(tmp_path: Path) -> None:
    selected = {
        "rollout_id": "rollout-a",
        "source_attempt_index": 0,
        "restored_attempt_index": 1,
    }
    audit = tmp_path / "audit.jsonl"
    events = [
        {
            "event": "mutation_applied",
            "rollout_id": "rollout-a",
            "attempt_index": 0,
            "sentinel_count": 1,
        },
        {
            "event": "state_restored",
            "rollout_id": "rollout-a",
            "attempt_index": 1,
            "sentinel_count": 1,
        },
        {
            "event": "state_verified",
            "rollout_id": "rollout-a",
            "attempt_index": 1,
            "sentinel_count": 1,
        },
    ]
    audit.write_text("".join(json.dumps(event) + "\n" for event in events))

    _HELPER._verify_workplace_audit(selected, audit)

    events.append(
        {
            "event": "mutation_applied",
            "rollout_id": "rollout-a",
            "attempt_index": 1,
            "sentinel_count": 2,
        }
    )
    audit.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(AssertionError, match="exactly once"):
        _HELPER._verify_workplace_audit(selected, audit)
