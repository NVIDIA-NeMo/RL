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

import hashlib
import importlib.util
import io
import json
import tarfile
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_agent_records_resolve_index_from_shard_commit_root(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    commit_root = snapshot / "gym-shards" / "first"
    directory = commit_root / "gym" / "agent"
    directory.mkdir(parents=True)
    member_name = "rollout-a.a0.json"
    record = {
        "rollout_id": "rollout-a",
        "attempt_index": 0,
        "boundary_index": 1,
    }
    record_payload = json.dumps(record).encode()

    archive_path = directory / "agent-part-000000.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(record_payload)
        archive.addfile(info, io.BytesIO(record_payload))
    archive_payload = archive_path.read_bytes()

    index_record = {
        "rollout_id": "rollout-a",
        "attempt_index": 0,
        "archive": archive_path.name,
        "member": member_name,
        "sha256": _sha256(record_payload),
        "bytes": len(record_payload),
    }
    index_payload = (json.dumps(index_record) + "\n").encode()
    index_path = directory / "agent-index.jsonl"
    index_path.write_bytes(index_payload)

    manifest = {
        "schema_version": 2,
        "records": 1,
        "archives": [
            {
                "name": archive_path.name,
                "sha256": _sha256(archive_payload),
                "members": 1,
                "bytes": len(archive_payload),
            }
        ],
        "record_index": {
            "relative_path": str(index_path.relative_to(commit_root)),
            "sha256": _sha256(index_payload),
            "records": 1,
            "bytes": len(index_payload),
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    assert _HELPER._read_agent_records(
        snapshot,
        manifest_path,
        commit_root=commit_root,
    ) == [record]


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


def test_genrm_restore_allows_additional_complete_phase_two_cohort(
    tmp_path: Path,
) -> None:
    selected = {
        "rollouts": [
            {
                "rollout_id": "restored_g0",
                "source_attempt_index": 0,
                "restored_attempt_index": 1,
            },
            {
                "rollout_id": "restored_g1",
                "source_attempt_index": 0,
                "restored_attempt_index": 1,
            },
        ]
    }
    restored = ["restored_g0-a1", "restored_g1-a1"]
    extra = ["extra_g0", "extra_g1"]
    rollout_events = [
        {"event": "dispatch", "rollout_ids": restored},
        *[
            {"event": "completion_forwarded", "rollout_id": rollout_id, "reward": 1.0}
            for rollout_id in restored
        ],
    ]
    audit_events = [
        {"phase": "phase1", "event": "verify_entered"},
        {"phase": "phase1", "event": "verify_waiting"},
    ]
    for cohort in (restored, extra):
        audit_events.extend(
            [
                {
                    "phase": "phase2",
                    "event": "verify_entered",
                    "capture_rollout_id": cohort[0],
                },
                {
                    "phase": "phase2",
                    "event": "verify_waiting",
                    "cohort_size": 1,
                    "capture_rollout_ids": [cohort[0]],
                },
                {
                    "phase": "phase2",
                    "event": "verify_entered",
                    "capture_rollout_id": cohort[1],
                },
                {
                    "phase": "phase2",
                    "event": "reward_computed",
                    "cohort_size": 2,
                    "capture_rollout_ids": cohort,
                },
                *[
                    {
                        "phase": "phase2",
                        "event": "verify_returned",
                        "capture_rollout_id": rollout_id,
                    }
                    for rollout_id in cohort
                ],
            ]
        )

    audit = tmp_path / "audit.jsonl"
    audit.write_text("".join(json.dumps(event) + "\n" for event in audit_events))

    _HELPER._verify_genrm_restore(selected, rollout_events, audit)


def test_genrm_restore_rejects_duplicate_reward_for_selected_cohort(
    tmp_path: Path,
) -> None:
    selected = {
        "rollouts": [
            {
                "rollout_id": "restored_g0",
                "source_attempt_index": 0,
                "restored_attempt_index": 1,
            },
            {
                "rollout_id": "restored_g1",
                "source_attempt_index": 0,
                "restored_attempt_index": 1,
            },
        ]
    }
    restored = ["restored_g0-a1", "restored_g1-a1"]
    rollout_events = [
        {"event": "dispatch", "rollout_ids": restored},
        *[
            {"event": "completion_forwarded", "rollout_id": rollout_id, "reward": 1.0}
            for rollout_id in restored
        ],
    ]
    audit_events = [
        {"phase": "phase1", "event": "verify_entered"},
        {"phase": "phase1", "event": "verify_waiting"},
        *[
            {
                "phase": "phase2",
                "event": "verify_entered",
                "capture_rollout_id": rollout_id,
            }
            for rollout_id in restored
        ],
        {
            "phase": "phase2",
            "event": "verify_waiting",
            "cohort_size": 1,
            "capture_rollout_ids": [restored[0]],
        },
        {
            "phase": "phase2",
            "event": "reward_computed",
            "cohort_size": 2,
            "capture_rollout_ids": restored,
        },
        {
            "phase": "phase2",
            "event": "reward_computed",
            "cohort_size": 2,
            "capture_rollout_ids": restored,
        },
        *[
            {
                "phase": "phase2",
                "event": "verify_returned",
                "capture_rollout_id": rollout_id,
            }
            for rollout_id in restored
        ],
    ]
    audit = tmp_path / "audit.jsonl"
    audit.write_text("".join(json.dumps(event) + "\n" for event in audit_events))

    with pytest.raises(AssertionError, match="rewarded more than once"):
        _HELPER._verify_genrm_restore(selected, rollout_events, audit)
