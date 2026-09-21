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

import argparse
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_prefix_recovery_snapshot.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gym_prefix_recovery_snapshot", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, True),
        ({"terminal_finish_reason": "stop"}, False),
        ({"terminal_stop_reason": 128001}, False),
        ({"prefix_token_count": 64}, False),
        ({"effective_output_limit": 16}, False),
    ],
)
def test_is_recoverable_active_prefix(
    updates: dict[str, object],
    expected: bool,
) -> None:
    prefix = {
        "prefix_token_count": 16,
        "effective_output_limit": 64,
        "terminal_finish_reason": None,
        "terminal_stop_reason": None,
        **updates,
    }

    assert (
        _HELPER._is_recoverable_active_prefix(
            prefix,
            max_generation_tokens=64,
        )
        is expected
    )


def test_agent_records_reads_archive_only_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "gym" / "agent"
    directory.mkdir(parents=True)
    member_name = "rollout-a.a0.json"
    record = {
        "rollout_id": "rollout-a",
        "attempt_index": 0,
        "boundary_index": 0,
        "resource_state_revisions": {"resources": 1},
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
            "relative_path": str(index_path.relative_to(tmp_path)),
            "sha256": _sha256(index_payload),
            "records": 1,
            "bytes": len(index_payload),
        },
    }
    manifest_payload = json.dumps(manifest).encode()
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(manifest_payload)
    participant = {
        "manifest": {
            "relative_path": str(manifest_path.relative_to(tmp_path)),
            "manifest_digest": _sha256(manifest_payload),
        }
    }

    assert _HELPER._agent_records(tmp_path, participant) == [record]


def test_verify_restore_rejects_source_attempt_redispatch(tmp_path: Path) -> None:
    digest = "1" * 64
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "rollout_id": "rollout-a",
                "source_attempt_index": 0,
                "restored_attempt_index": 1,
                "source_model_call_id": "call-a",
                "prefix_token_count": 2,
                "prefix_digest": digest,
            }
        )
    )
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "dispatch",
                "rollout_ids": ["rollout-a"],
            }
        )
        + "\n"
    )
    log = tmp_path / "run.log"
    log.write_text("")

    with pytest.raises(AssertionError, match="redispatched after restore"):
        _HELPER.verify_restore(
            argparse.Namespace(selection=selection, events=events, run_log=log)
        )


def test_verify_restore_checks_prefix_digest_and_terminal_split(
    tmp_path: Path,
) -> None:
    digest = "1" * 64
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "rollout_id": "rollout-a",
                "source_attempt_index": 0,
                "restored_attempt_index": 1,
                "source_model_call_id": "call-a",
                "prefix_token_count": 2,
                "prefix_digest": digest,
            }
        )
    )
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "dispatch",
                    "rollout_ids": ["rollout-a-a1"],
                },
                {
                    "event": "completion_forwarded",
                    "rollout_id": "rollout-a-a1",
                },
            )
        )
        + "\n"
    )
    log = tmp_path / "run.log"
    log.write_text(
        "generation prefix restored: rollout_id=rollout-a-a1 model_call_id=call-b "
        f"source_model_call_id=call-a prefix_tokens=2 prefix_digest={digest}\n"
        "generation prefix completed: rollout_id=rollout-a-a1 model_call_id=call-b "
        "source_model_call_id=call-a prefix_tokens=2 tail_tokens=3 "
        "total_generation_tokens=5\n"
    )

    _HELPER.verify_restore(
        argparse.Namespace(
            selection=selection,
            events=events,
            run_log=log,
            profile="basic",
            audit_events=None,
        )
    )


def test_workplace_restore_audit_requires_one_mutation(tmp_path: Path) -> None:
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
