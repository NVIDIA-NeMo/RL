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

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import nemo_rl.experience.rollout_checkpoint as checkpoint_module
from nemo_rl.experience.interfaces import Completion
from nemo_rl.experience.rollout_checkpoint import (
    ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
    CheckpointConflictError,
    CompletedSiblingRecord,
    CorruptCheckpointError,
    IncompatibleCheckpointError,
    RolloutCheckpointStore,
    RolloutWorkItem,
    StaleAttemptError,
    StorageUnavailableError,
)


def _work(*, attempt_id: int = 0, policy_version: int = 7) -> RolloutWorkItem:
    return RolloutWorkItem(
        run_id="run-1",
        group_id="group-1",
        prompt_id="prompt-1",
        dispatch_sequence=3,
        target_step=9,
        attempt_id=attempt_id,
        policy_version=policy_version,
        prompt_fingerprint="prompt-sha",
        sampling_fingerprint="sampling-sha",
        tokenizer_fingerprint="tokenizer-sha",
        num_generations=4,
        prompt_ref={"dataset_index": 11},
    )


def _record(
    *,
    generation_index: int = 0,
    attempt_id: int = 0,
    reward: float = 1.5,
) -> CompletedSiblingRecord:
    return CompletedSiblingRecord(
        schema_version=ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
        run_id="run-1",
        group_id="group-1",
        prompt_id="prompt-1",
        generation_index=generation_index,
        attempt_id=attempt_id,
        policy_version=7,
        prompt_fingerprint="prompt-sha",
        sampling_fingerprint="sampling-sha",
        tokenizer_fingerprint="tokenizer-sha",
        phase="SIBLING_COMPLETE",
        completion=Completion(
            message_log=[
                {
                    "role": "user",
                    "content": "2+2?",
                    "token_ids": torch.tensor([2, 10, 2], dtype=torch.long),
                },
                {
                    "role": "assistant",
                    "content": "4",
                    "token_ids": torch.tensor([4], dtype=torch.long),
                    "generation_logprobs": torch.tensor([-0.25]),
                },
            ],
            env_extras={"answer": 4},
            truncated=False,
            reward=reward,
        ),
        sample_metrics={
            "turn_count": 1,
            "per_worker_token_counts": {0: 1},
        },
    )


def test_completed_sibling_round_trip(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    record = _record(generation_index=2)

    ack = store.persist_completed(record)
    loaded = store.load_completed(_work())

    assert not ack.already_existed
    assert ack.logical_key == ("run-1", "group-1", 2)
    assert ack.path == tmp_path / "run-1" / "group-1" / "g00002.pt"
    assert set(loaded) == {2}
    assert loaded[2].completion.reward == 1.5
    assert torch.equal(
        loaded[2].completion.message_log[1]["token_ids"], torch.tensor([4])
    )


def test_same_record_retry_returns_prior_ack(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    record = _record()

    first = store.persist_completed(record)
    second = store.persist_completed(record)

    assert not first.already_existed
    assert second.already_existed
    assert second.record_checksum == first.record_checksum
    assert second.path == first.path


def test_same_key_with_different_payload_is_a_conflict(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.persist_completed(_record(reward=1.0))

    with pytest.raises(CheckpointConflictError, match="different semantic"):
        store.persist_completed(_record(reward=2.0))


def test_fence_rejects_new_old_attempt_writes_but_preserves_committed_record(
    tmp_path,
) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.persist_completed(_record(generation_index=0, attempt_id=0))

    assert store.fence("run-1", "group-1", 1) == 1
    assert store.fence("run-1", "group-1", 0) == 1
    store.validate_attempt("run-1", "group-1", 1)
    with pytest.raises(StaleAttemptError, match="older than durable fence"):
        store.validate_attempt("run-1", "group-1", 0)
    with pytest.raises(StaleAttemptError, match="older than durable fence"):
        store.persist_completed(_record(generation_index=1, attempt_id=0))

    loaded = store.load_completed(_work(attempt_id=1))
    assert set(loaded) == {0}
    assert loaded[0].attempt_id == 0


def test_payload_corruption_is_detected_before_deserialization(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    ack = store.persist_completed(_record())
    contents = bytearray(ack.path.read_bytes())
    contents[-1] ^= 0xFF
    ack.path.write_bytes(contents)

    with pytest.raises(CorruptCheckpointError, match="payload checksum mismatch"):
        store.load_completed(_work())


def test_non_dictionary_file_header_is_reported_as_corrupt(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    ack = store.persist_completed(_record())
    contents = ack.path.read_bytes()
    header_start = (
        len(checkpoint_module._FILE_MAGIC) + checkpoint_module._HEADER_LENGTH_BYTES
    )
    original_header_length = int.from_bytes(
        contents[len(checkpoint_module._FILE_MAGIC) : header_start], "big"
    )
    payload = contents[header_start + original_header_length :]
    invalid_header = b"[]"
    ack.path.write_bytes(
        checkpoint_module._FILE_MAGIC
        + len(invalid_header).to_bytes(checkpoint_module._HEADER_LENGTH_BYTES, "big")
        + invalid_header
        + payload
    )

    with pytest.raises(CorruptCheckpointError, match="is not a dictionary"):
        store.load_completed(_work())


def test_attempt_fence_corruption_is_detected(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.fence("run-1", "group-1", 3)
    fence_path = tmp_path / "run-1" / "group-1" / "fence.json"
    fence_path.write_text(
        fence_path.read_text().replace('"min_attempt_id":3', '"min_attempt_id":0')
    )

    with pytest.raises(CorruptCheckpointError, match="failed its checksum"):
        store.get_fence("run-1", "group-1")


def test_attempt_fence_invalid_utf8_is_reported_as_corrupt(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.fence("run-1", "group-1", 1)
    fence_path = tmp_path / "run-1" / "group-1" / "fence.json"
    fence_path.write_bytes(b"\xff")

    with pytest.raises(CorruptCheckpointError, match="invalid UTF-8"):
        store.get_fence("run-1", "group-1")


def test_attempt_fence_rejects_boolean_attempt_id(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.fence("run-1", "group-1", 1)
    fence_path = tmp_path / "run-1" / "group-1" / "fence.json"
    fence_payload = {
        "schema_version": ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
        "run_id": "run-1",
        "group_id": "group-1",
        "min_attempt_id": True,
    }
    fence_payload["checksum"] = checkpoint_module._json_checksum(fence_payload)
    fence_path.write_text(
        json.dumps(fence_payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(CorruptCheckpointError, match="invalid min_attempt_id"):
        store.get_fence("run-1", "group-1")


def test_incomplete_temp_file_is_ignored(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.persist_completed(_record(generation_index=0))
    group_dir = tmp_path / "run-1" / "group-1"
    (group_dir / ".g00001.pt.interrupted.tmp").write_bytes(b"partial")

    loaded = store.load_completed(_work())

    assert set(loaded) == {0}


def test_logical_identity_mismatch_is_incompatible(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.persist_completed(_record())

    with pytest.raises(IncompatibleCheckpointError, match="does not match"):
        store.load_completed(_work(policy_version=8))


def test_generation_index_outside_requested_group_is_incompatible(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.persist_completed(_record(generation_index=4))

    with pytest.raises(IncompatibleCheckpointError, match="expects 4 generations"):
        store.load_completed(_work())


def test_delete_group_is_idempotent(tmp_path) -> None:
    store = RolloutCheckpointStore(tmp_path)
    store.persist_completed(_record())
    store.fence("run-1", "group-1", 1)

    store.delete_group("run-1", "group-1")
    store.delete_group("run-1", "group-1")

    assert store.load_completed(_work(attempt_id=1)) == {}
    assert store.get_fence("run-1", "group-1") == 0


def test_atomic_write_failure_never_exposes_final_path(tmp_path, monkeypatch) -> None:
    store = RolloutCheckpointStore(tmp_path)

    def _fail_replace(_src, _dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(StorageUnavailableError, match="atomically commit"):
        store.persist_completed(_record())

    group_dir = tmp_path / "run-1" / "group-1"
    assert not (group_dir / "g00000.pt").exists()
    assert list(group_dir.glob("*.tmp")) == []


def test_first_group_write_fsyncs_new_directory_entries(tmp_path, monkeypatch) -> None:
    root_dir = tmp_path / "rollout-checkpoints"
    store = RolloutCheckpointStore(root_dir)
    fsynced_directories: list[Path] = []
    monkeypatch.setattr(
        checkpoint_module,
        "_fsync_directory",
        fsynced_directories.append,
    )

    store.persist_completed(_record())

    run_dir = root_dir / "run-1"
    group_dir = run_dir / "group-1"
    assert fsynced_directories == [root_dir, run_dir, group_dir]


def test_path_traversal_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="path separators"):
        replace(_work(), group_id="../other-group")


def test_non_cpu_completion_tensor_is_rejected() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to construct a non-CPU tensor")
    record = _record()
    record.completion.message_log[1]["token_ids"] = torch.tensor([4], device="cuda")

    with pytest.raises(ValueError, match="non-CPU tensor"):
        replace(record)
