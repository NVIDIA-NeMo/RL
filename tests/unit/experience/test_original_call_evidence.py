# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve verified original calls across publication, cleanup and I/O faults."""

import fcntl
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.experience import original_call_evidence as evidence_module
from nemo_rl.experience import rollout_reassembler_actor as actor_module
from nemo_rl.experience.original_call_evidence import (
    ORIGINAL_CALL_EVIDENCE_PATH_TAG,
    ORIGINAL_CALL_EVIDENCE_SHA256_TAG,
    OriginalCallEvidenceConfig,
    OriginalCallEvidenceWriter,
    original_call_evidence_config,
)
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    RolloutReassembler,
)
from nemo_rl.experience.rollout_reassembler_actor import (
    RolloutReassemblerActor,
    RolloutReassemblerActorConfig,
    assert_metadata_only,
)
from tests.unit.experience.test_call_view_publication import _publish, _request
from tests.unit.experience.test_finalized_call_views import _case
from tests.unit.experience.test_logical_owner_finalization import PublicationDataPlane

pytestmark = pytest.mark.nemo_gym


def _config(path: Path, *, contexts: int = 2) -> OriginalCallEvidenceConfig:
    return OriginalCallEvidenceConfig(str(path), 1_000_000, 500_000, contexts)


def _finalizer(plane, *, media: bool, config=None) -> RolloutReassembler:
    return RolloutReassembler(
        plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        router_replay_enabled=True,
        capture_media=media,
        original_call_evidence=config,
    )


@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("call_publication", [False, True])
def test_original_wire_replay_survives_cleanup_without_changing_training(
    tmp_path: Path, media: bool, call_publication: bool
) -> None:
    case = _case(media=media)
    expected = case.finalizer.finalize_rollout_with_call_views(
        "owner", case.receipt, reward=0.0, action_flags=case.flags
    )
    assert expected.valid
    original_columns = {
        key: {name: tensor[0].clone() for name, tensor in columns.items()}
        for (partition, key), columns in case.data_plane.rows.items()
        if partition == "staged"
    }
    case.data_plane.reads.clear()
    finalizer = _finalizer(case.data_plane, media=media, config=_config(tmp_path))
    request = replace(_request([case]), capture_call_views=call_publication)
    result = _publish(finalizer, request)
    assert result.meta is not None and result.canonical_output_tokens == 4
    assert len(case.data_plane.reads) == (2 if media else 1)
    assert not any(partition == "staged" for partition, _ in case.data_plane.rows)
    references = {tag[ORIGINAL_CALL_EVIDENCE_PATH_TAG] for tag in result.meta.tags}
    assert len(references) == 1
    path = Path(references.pop())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(
        tag[ORIGINAL_CALL_EVIDENCE_SHA256_TAG] == digest for tag in result.meta.tags
    )
    assert_metadata_only(result)
    archived = torch.load(path, weights_only=True)
    assert archived["native_base_and_route_commitments_verified"]
    assert not archived["refit_free_generation_verified"]
    assert archived["rows"].keys() == original_columns.keys()
    restored = PublicationDataPlane()
    for key, columns in archived["rows"].items():
        for name, tensor in columns.items():
            assert torch.equal(tensor, original_columns[key][name]), (key, name)
        restored.rows["staged", key] = TensorDict(
            {name: tensor.unsqueeze(0) for name, tensor in columns.items()},
            batch_size=[1],
        )
    flags = tuple(ActionOutputFlags(**flag) for flag in archived["action_flags"])
    actual = _finalizer(restored, media=media).finalize_rollout_with_call_views(
        "owner", archived["receipt"], reward=0.0, action_flags=flags
    )
    assert actual.valid, actual.rejection_reason
    assert actual.token_ids == expected.token_ids
    assert actual.token_mask == expected.token_mask
    assert actual.logprobs == expected.logprobs
    assert torch.equal(actual.routed_experts, expected.routed_experts)
    for saved_call, replay_call in zip(
        expected.call_views, actual.call_views, strict=True
    ):
        assert replay_call.model_call_id == saved_call.model_call_id
        assert replay_call.row.token_ids == saved_call.row.token_ids
        assert replay_call.row.token_mask == saved_call.row.token_mask
        assert replay_call.row.output_masks == saved_call.row.output_masks
        assert torch.equal(
            replay_call.row.routed_experts, saved_call.row.routed_experts
        )
        if media and saved_call.row.media:
            for name, packed in saved_call.row.media.items():
                assert all(
                    torch.equal(a, b)
                    for a, b in zip(
                        packed.tensors, replay_call.row.media[name].tensors, strict=True
                    )
                )
    assert finalizer._evidence_writer is not None
    assert len(list(tmp_path.glob("original-calls-*.pt"))) == 1


def test_lost_publication_ack_reattaches_archive_after_worker_cap(
    tmp_path: Path,
) -> None:
    case = _case(media=True)
    finalizer = _finalizer(
        case.data_plane, media=True, config=_config(tmp_path, contexts=1)
    )
    request = replace(_request([case]), capture_call_views=False)
    case.data_plane.lose_ack = True
    with pytest.raises(OSError, match="acknowledgement lost"):
        _publish(finalizer, request)
    assert case.data_plane.delete_count == 0
    assert finalizer._evidence_writer is not None
    assert not finalizer._evidence_writer.wants_capture
    case.data_plane.lose_ack = False
    result = _publish(finalizer, request)
    assert result.meta is not None
    assert Path(result.meta.tags[0][ORIGINAL_CALL_EVIDENCE_PATH_TAG]).is_file()
    assert len(list(tmp_path.glob("original-calls-*.pt"))) == 1
    assert not any(partition == "staged" for partition, _ in case.data_plane.rows)


def test_diagnostic_failure_cannot_reject_training_or_skip_cleanup(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "a-file"
    destination.write_text("preserve")
    case = _case(media=True)
    finalizer = _finalizer(case.data_plane, media=True, config=_config(destination))
    result = _publish(finalizer, replace(_request([case]), capture_call_views=False))
    assert result.meta is not None and result.valid_row_count == 1
    assert result.canonical_output_tokens == 4
    assert ORIGINAL_CALL_EVIDENCE_PATH_TAG not in result.meta.tags[0]
    assert not any(partition == "staged" for partition, _ in case.data_plane.rows)
    assert destination.read_text() == "preserve"


def test_corrupt_generation_bytes_never_become_archive_evidence(tmp_path: Path) -> None:
    case = _case(media=True)
    case.data_plane.rows["staged", "owner/c1"]["routed_experts"][0, 0, 0, 0] += 1
    finalizer = _finalizer(case.data_plane, media=True, config=_config(tmp_path))
    result = finalizer.finalize_rollout(
        "owner",
        case.receipt,
        reward=0,
        context_compaction=True,
        action_flags=case.flags,
    )
    assert not result.valid and result.original_call_evidence is None
    assert not list(tmp_path.glob("original-calls-*.pt"))


def _write_small(writer: OriginalCallEvidenceWriter, identity: int):
    key = f"owner{identity}/call"
    return writer.write_verified_context(
        receipt={"manifest": [{"staging_key": key}]},
        rows={key: {"tensor": torch.arange(64)}},
        context_compaction=True,
        capture_media=False,
        action_flags=None,
    )


def test_byte_limit_holds_across_independent_writers_and_restarts(
    tmp_path: Path,
) -> None:
    config = OriginalCallEvidenceConfig(str(tmp_path), 10_000, 6_000, 20)
    references = []
    for index in range(10):
        reference = _write_small(OriginalCallEvidenceWriter(config), index)
        if reference is not None:
            references.append(reference)
    assert references
    index = json.loads((tmp_path / "index.json").read_text())
    assert sum(entry["bytes"] for entry in index["entries"].values()) <= 10_000
    assert sum(path.stat().st_size for path in tmp_path.glob("*.pt")) <= 10_000
    for reference in references:
        assert Path(reference.path).stat().st_size <= 6_000
        assert (
            hashlib.sha256(Path(reference.path).read_bytes()).hexdigest()
            == reference.sha256
        )
    assert not list(tmp_path.glob(".original-calls-*"))


def test_busy_lock_is_skipped_instead_of_waiting(tmp_path: Path) -> None:
    with (tmp_path / ".budget.lock").open("ab") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        writer = OriginalCallEvidenceWriter(_config(tmp_path))
        assert _write_small(writer, 0) is None
        assert writer.wants_capture
    assert _write_small(writer, 0) is not None


def test_failed_commit_keeps_conservative_write_ahead_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = evidence_module._atomic_index
    calls = 0

    def fail_commit(index: dict, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated full filesystem")
        original(index, target)

    monkeypatch.setattr(evidence_module, "_atomic_index", fail_commit)
    writer = OriginalCallEvidenceWriter(_config(tmp_path))
    assert _write_small(writer, 0) is None
    assert not writer.wants_capture
    index = json.loads((tmp_path / "index.json").read_text())
    assert list(index["entries"].values())[0]["status"] == "reserved"
    assert list(index["entries"].values())[0]["bytes"] == 500_000
    assert sum(path.stat().st_size for path in tmp_path.glob("*.pt")) < 500_000


def test_corrupt_negative_budget_cannot_authorize_more_output(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {"old": {"status": "reserved", "bytes": -100_000}},
            }
        )
    )
    writer = OriginalCallEvidenceWriter(_config(tmp_path))
    assert _write_small(writer, 0) is None
    assert not writer.wants_capture
    assert not list(tmp_path.glob("*.pt"))


def test_archive_opt_in_is_derived_without_training_configuration(
    tmp_path: Path,
) -> None:
    assert original_call_evidence_config(None) is None
    config = original_call_evidence_config(tmp_path)
    assert config is not None and config.output_path == str(tmp_path / "original_calls")
    assert config.max_total_bytes == 8 * 1024**3
    assert_metadata_only(config)


def test_actor_constructor_forwards_explicit_evidence_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        actor_module, "build_data_plane_client", lambda *a, **k: object()
    )
    config = RolloutReassemblerActorConfig(
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        router_replay_enabled=True,
        defer_routed_experts_to_policy=False,
        max_seq_len=1024,
        capture_media=True,
        original_call_evidence=_config(tmp_path),
    )
    actor_class = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor.__init__({}, config)
    assert actor._finalizer._evidence_writer is not None
    assert actor._finalizer._evidence_writer.config == config.original_call_evidence
