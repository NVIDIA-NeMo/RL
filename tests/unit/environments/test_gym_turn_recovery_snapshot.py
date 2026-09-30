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
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.functional._turn_recovery_token_evidence import (
    build_rollout_token_evidence,
    verify_token_evidence,
)

_HELPER_PATH = (
    Path(__file__).parents[2] / "functional" / "_gym_turn_recovery_snapshot.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gym_turn_recovery_snapshot", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


def test_select_snapshot_inspects_each_rejected_candidate_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"prompt": "test"}\n')
    rejected = tmp_path / "snapshot_000001"
    accepted = tmp_path / "snapshot_000002"
    polls = 0

    def published_snapshots(_checkpoint_dir: Path) -> list[Path]:
        nonlocal polls
        polls += 1
        return [rejected] if polls == 1 else [rejected, accepted]

    inspected: list[Path] = []

    def inspect_snapshot(
        snapshot: Path,
        _dataset_rows: list[dict[str, object]],
        *,
        profile: str,
    ) -> dict[str, str]:
        del profile
        inspected.append(snapshot)
        if snapshot == rejected:
            raise AssertionError("not ready")
        return {"snapshot": str(snapshot)}

    monkeypatch.setattr(_HELPER, "_published_bootstrap_snapshots", published_snapshots)
    monkeypatch.setattr(_HELPER, "inspect_snapshot", inspect_snapshot)
    monkeypatch.setattr(_HELPER.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(_HELPER.time, "sleep", lambda _seconds: None)

    selection = tmp_path / "selected.json"
    _HELPER.select_snapshot(
        SimpleNamespace(
            checkpoint_dir=tmp_path,
            dataset=dataset,
            pid=123,
            profile="counter",
            run_log=tmp_path / "run.log",
            selection=selection,
            timeout_s=1.0,
        )
    )

    assert inspected == [rejected, accepted]
    assert capsys.readouterr().out.count("snapshot candidate rejected") == 1
    assert json.loads(selection.read_text()) == {"snapshot": str(accepted)}


def test_reads_digest_bound_model_lineage_member(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    ledger_dir = snapshot / "model_ledgers" / "model-server"
    ledger_dir.mkdir(parents=True)
    payload = (
        json.dumps(
            {
                "model_call_id": "call-1",
                "staging_key": "stage-1",
                "staging_digest": "digest-1",
                "prev_len": 0,
                "cum_len": 2,
            },
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    source = tmp_path / "rollout-a.lineage.jsonl"
    source.write_bytes(payload)
    archive_path = ledger_dir / "lineage-000000.tar"
    with tarfile.open(archive_path, mode="w:") as archive:
        archive.add(source, arcname=source.name)

    lineage_index = ledger_dir / "lineage-index.jsonl"
    lineage_index_payload = (
        json.dumps(
            {
                "capture_key": "rollout-a",
                "archive": archive_path.name,
                "member": source.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "rows": 1,
                "bytes": len(payload),
            },
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    lineage_index.write_bytes(lineage_index_payload)
    manifest_path = ledger_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "lineage_index": {
                    "relative_path": str(lineage_index.relative_to(snapshot)),
                    "sha256": hashlib.sha256(lineage_index_payload).hexdigest(),
                    "records": 1,
                    "bytes": len(lineage_index_payload),
                },
                "archives": [
                    {
                        "name": archive_path.name,
                        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                        "members": 1,
                        "bytes": archive_path.stat().st_size,
                    }
                ],
            }
        )
    )

    assert _HELPER._read_model_lineage_records(
        snapshot, manifest_path, "rollout-a"
    ) == [json.loads(payload)]


def test_continued_rollout_saves_pre_crash_lineage_commitments() -> None:
    selected = _HELPER._continued_rollout(
        "rollout-a",
        0,
        [
            {
                "capture_key": "rollout-a",
                "boundary_model_call_id": "call-2",
                "key": "stage-1",
            },
            {
                "capture_key": "rollout-a",
                "boundary_model_call_id": "call-2",
                "key": "stage-2",
            },
        ],
        [
            {
                "model_call_id": "call-1",
                "staging_key": "stage-1",
                "staging_digest": "digest-1",
                "prev_len": 0,
                "cum_len": 2,
            },
            {
                "model_call_id": "call-2",
                "staging_key": "stage-2",
                "staging_digest": "digest-2",
                "prev_len": 2,
                "cum_len": 3,
            },
        ],
    )

    assert selected["pre_cut_segments"] == [
        {
            "staging_key": "stage-1",
            "staging_digest": "digest-1",
            "prev_len": 0,
            "cum_len": 2,
        },
        {
            "staging_key": "stage-2",
            "staging_digest": "digest-2",
            "prev_len": 2,
            "cum_len": 3,
        },
    ]


def test_token_evidence_compares_saved_segments_at_exact_positions() -> None:
    evidence = build_rollout_token_evidence(
        rollout_id="rollout-a",
        canonical_token_ids=[10, 11, 12, 13, 14],
        staged_segments=[
            {
                "staging_key": "call-1",
                "prev_len": 0,
                "cum_len": 3,
                "token_ids_delta": [10, 11, 12],
                "selection_staging_digest": "digest-1",
                "restored_staging_digest": "digest-1",
            },
            {
                "staging_key": "call-2",
                "prev_len": 3,
                "cum_len": 5,
                "token_ids_delta": [13, 14],
                "selection_staging_digest": "digest-2",
                "restored_staging_digest": "digest-2",
            },
        ],
    )

    assert [segment["staging_key"] for segment in evidence["segments"]] == [
        "call-1",
        "call-2",
    ]
    assert all(segment["matched"] for segment in evidence["segments"])
    assert all(
        segment["saved_delta_sha256"] == segment["canonical_slice_sha256"]
        for segment in evidence["segments"]
    )


def test_token_evidence_rejects_changed_pre_cut_token() -> None:
    with pytest.raises(AssertionError, match="changed a saved pre-cut token segment"):
        build_rollout_token_evidence(
            rollout_id="rollout-a",
            canonical_token_ids=[10, 99, 12],
            staged_segments=[
                {
                    "staging_key": "call-1",
                    "prev_len": 0,
                    "cum_len": 3,
                    "token_ids_delta": [10, 11, 12],
                    "selection_staging_digest": "digest-1",
                    "restored_staging_digest": "digest-1",
                }
            ],
        )


def test_token_evidence_rejects_restored_staging_digest_mismatch() -> None:
    with pytest.raises(AssertionError, match="pre-crash lineage commitment"):
        build_rollout_token_evidence(
            rollout_id="rollout-a",
            canonical_token_ids=[10],
            staged_segments=[
                {
                    "staging_key": "call-1",
                    "prev_len": 0,
                    "cum_len": 1,
                    "token_ids_delta": [10],
                    "selection_staging_digest": "before-crash",
                    "restored_staging_digest": "after-restore",
                }
            ],
        )


def test_verify_token_evidence_requires_every_snapshot_reference(
    tmp_path: Path,
) -> None:
    selected = {
        "continued_rollouts": [
            {
                "rollout_id": "rollout-a",
                "source_capture_key": "rollout-a",
                "pre_cut_segments": [
                    {
                        "staging_key": "call-1",
                        "staging_digest": "digest-1",
                        "prev_len": 0,
                        "cum_len": 2,
                    },
                    {
                        "staging_key": "call-2",
                        "staging_digest": "digest-2",
                        "prev_len": 2,
                        "cum_len": 3,
                    },
                ],
            }
        ]
    }
    evidence = build_rollout_token_evidence(
        rollout_id="rollout-a",
        canonical_token_ids=[10, 11, 12],
        staged_segments=[
            {
                "staging_key": "call-1",
                "prev_len": 0,
                "cum_len": 2,
                "token_ids_delta": [10, 11],
                "selection_staging_digest": "digest-1",
                "restored_staging_digest": "digest-1",
            },
            {
                "staging_key": "call-2",
                "prev_len": 2,
                "cum_len": 3,
                "token_ids_delta": [12],
                "selection_staging_digest": "digest-2",
                "restored_staging_digest": "digest-2",
            },
        ],
    )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "group.json").write_text(
        json.dumps({"group_id": "group-a", "rollouts": [evidence]})
    )

    verify_token_evidence(selected, evidence_dir)

    evidence["segments"].pop()
    (evidence_dir / "group.json").write_text(
        json.dumps({"group_id": "group-a", "rollouts": [evidence]})
    )
    with pytest.raises(AssertionError, match="does not cover"):
        verify_token_evidence(selected, evidence_dir)


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
