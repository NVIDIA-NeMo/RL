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

"""Durable records for completed rollout siblings.

This module deliberately has no Ray dependencies. ``RolloutCheckpointStore``
contains the durability and validation logic; a later Ray actor can wrap this
class without duplicating the storage protocol.

The checkpoint directory is trusted. Payloads use ``torch.save`` because a
``Completion`` contains tensors and nested message logs. The file envelope is
not pickled: it carries a raw-payload checksum that is verified before
``torch.load`` deserializes the trusted payload.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import shutil
import struct
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch

from nemo_rl.experience.interfaces import Completion

ROLLOUT_CHECKPOINT_SCHEMA_VERSION = 1
_FILE_MAGIC = b"NEMORL_ROLLOUT_CHECKPOINT\n"
_HEADER_LENGTH_BYTES = 8
_MAX_HEADER_BYTES = 64 * 1024


class RolloutCheckpointError(RuntimeError):
    """Base class for completed-rollout checkpoint failures."""


class CheckpointConflictError(RolloutCheckpointError):
    """A logical sibling key already contains a different record."""


class StaleAttemptError(RolloutCheckpointError):
    """A write came from an attempt older than the durable group fence."""


class CorruptCheckpointError(RolloutCheckpointError):
    """A checkpoint cannot be decoded or fails an integrity check."""


class IncompatibleCheckpointError(RolloutCheckpointError):
    """A valid checkpoint does not match the requested rollout work."""


class StorageUnavailableError(RolloutCheckpointError):
    """The configured checkpoint filesystem could not complete an operation."""


@dataclass(frozen=True)
class RolloutWorkItem:
    """Stable controller-assigned identity for one prompt group.

    ``prompt_ref`` is control-plane input used by a generation worker. It is
    intentionally absent from completed-sibling files.
    """

    run_id: str
    group_id: str
    prompt_id: str
    dispatch_sequence: int
    target_step: int | None
    attempt_id: int
    policy_version: int
    prompt_fingerprint: str
    sampling_fingerprint: str
    tokenizer_fingerprint: str
    num_generations: int
    prompt_ref: Any

    def __post_init__(self) -> None:
        _validate_path_component(self.run_id, "run_id")
        _validate_path_component(self.group_id, "group_id")
        _validate_nonempty_string(self.prompt_id, "prompt_id")
        _validate_nonnegative_int(self.dispatch_sequence, "dispatch_sequence")
        if self.target_step is not None:
            _validate_nonnegative_int(self.target_step, "target_step")
        _validate_nonnegative_int(self.attempt_id, "attempt_id")
        _validate_nonnegative_int(self.policy_version, "policy_version")
        if (
            not isinstance(self.num_generations, int)
            or isinstance(self.num_generations, bool)
            or self.num_generations <= 0
        ):
            raise ValueError("num_generations must be > 0")
        _validate_fingerprint(self.prompt_fingerprint, "prompt_fingerprint")
        _validate_fingerprint(self.sampling_fingerprint, "sampling_fingerprint")
        _validate_fingerprint(self.tokenizer_fingerprint, "tokenizer_fingerprint")


@dataclass(frozen=True)
class CompletedSiblingRecord:
    """Training-ready result for one generation index of a prompt group."""

    schema_version: int
    run_id: str
    group_id: str
    prompt_id: str
    generation_index: int
    attempt_id: int
    policy_version: int
    prompt_fingerprint: str
    sampling_fingerprint: str
    tokenizer_fingerprint: str
    phase: Literal["SIBLING_COMPLETE"]
    completion: Completion
    sample_metrics: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != ROLLOUT_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_version must equal "
                f"{ROLLOUT_CHECKPOINT_SCHEMA_VERSION}, got {self.schema_version}"
            )
        _validate_path_component(self.run_id, "run_id")
        _validate_path_component(self.group_id, "group_id")
        _validate_nonempty_string(self.prompt_id, "prompt_id")
        _validate_nonnegative_int(self.generation_index, "generation_index")
        _validate_nonnegative_int(self.attempt_id, "attempt_id")
        _validate_nonnegative_int(self.policy_version, "policy_version")
        if self.phase != "SIBLING_COMPLETE":
            raise ValueError(f"unsupported completed-sibling phase: {self.phase!r}")
        _validate_fingerprint(self.prompt_fingerprint, "prompt_fingerprint")
        _validate_fingerprint(self.sampling_fingerprint, "sampling_fingerprint")
        _validate_fingerprint(self.tokenizer_fingerprint, "tokenizer_fingerprint")
        if not isinstance(self.completion, Completion):
            raise TypeError("completion must be a Completion")
        if not isinstance(self.completion.message_log, list):
            raise TypeError("completion.message_log must be a list")
        if self.completion.env_extras is not None and not isinstance(
            self.completion.env_extras, dict
        ):
            raise TypeError("completion.env_extras must be a dictionary or None")
        if not isinstance(self.completion.truncated, bool):
            raise TypeError("completion.truncated must be a bool")
        if not isinstance(self.completion.reward, (int, float)) or isinstance(
            self.completion.reward, bool
        ):
            raise TypeError("completion.reward must be numeric")
        if not isinstance(self.sample_metrics, dict):
            raise TypeError("sample_metrics must be a dictionary")
        _validate_cpu_tree(self.completion, "completion")
        _validate_cpu_tree(self.sample_metrics, "sample_metrics")

    @property
    def logical_key(self) -> tuple[str, str, int]:
        """Return the retry-stable idempotency key."""
        return (self.run_id, self.group_id, self.generation_index)


@dataclass(frozen=True)
class PersistAck:
    """Acknowledgement returned only after a sibling is durably committed."""

    logical_key: tuple[str, str, int]
    record_checksum: str
    path: Path
    already_existed: bool


@dataclass
class _GroupLockEntry:
    """Reference-counted lock for one rollout group."""

    lock: threading.Lock
    users: int = 0


class RolloutCheckpointStore:
    """Atomic filesystem store for completed siblings and attempt fences.

    One store instance is safe for concurrent calls in a process. Milestone 1
    still requires exactly one writer actor per ``root_dir``; group-scoped
    locks coordinate that actor's threads but are not distributed filesystem
    locks.
    """

    def __init__(self, root_dir: str | os.PathLike[str]) -> None:
        self.root_dir = Path(root_dir)
        self._group_locks_guard = threading.Lock()
        self._group_locks: dict[tuple[str, str], _GroupLockEntry] = {}
        _create_directory_durably(self.root_dir)

    def persist_completed(self, record: CompletedSiblingRecord) -> PersistAck:
        """Atomically persist one completed sibling.

        A retry with the same logical key and semantic checksum returns the
        original acknowledgement. A different semantic payload for an existing
        logical key raises ``CheckpointConflictError``.
        """
        record_payload = _record_to_payload(record)
        record_checksum = _semantic_checksum(record_payload)
        payload_bytes = _serialize_payload(record_payload)
        file_bytes = _encode_file(
            payload_bytes=payload_bytes,
            record_checksum=record_checksum,
        )

        with self._locked_group(record.run_id, record.group_id):
            min_attempt_id = self._load_fence_unlocked(record.run_id, record.group_id)
            if record.attempt_id < min_attempt_id:
                raise StaleAttemptError(
                    f"attempt {record.attempt_id} for {record.run_id}/{record.group_id} "
                    f"is older than durable fence {min_attempt_id}"
                )

            path = self._record_path(
                record.run_id, record.group_id, record.generation_index
            )
            if path.exists():
                existing, existing_checksum = self._load_record_file(path)
                if existing.logical_key != record.logical_key:
                    raise CorruptCheckpointError(
                        f"checkpoint key at {path} does not match its path"
                    )
                if existing_checksum != record_checksum:
                    raise CheckpointConflictError(
                        f"checkpoint {record.logical_key} already exists with a "
                        "different semantic checksum"
                    )
                return PersistAck(
                    logical_key=record.logical_key,
                    record_checksum=existing_checksum,
                    path=path,
                    already_existed=True,
                )

            self._atomic_write(path, file_bytes)
            committed, committed_checksum = self._load_record_file(path)
            if (
                committed.logical_key != record.logical_key
                or committed_checksum != record_checksum
            ):
                raise CorruptCheckpointError(
                    f"rollout checkpoint {path} failed post-commit verification"
                )
            return PersistAck(
                logical_key=record.logical_key,
                record_checksum=committed_checksum,
                path=path,
                already_existed=False,
            )

    def load_completed(
        self, work: RolloutWorkItem
    ) -> dict[int, CompletedSiblingRecord]:
        """Load and validate durable siblings reusable by ``work``."""
        with self._locked_group(work.run_id, work.group_id):
            group_dir = self._group_dir(work.run_id, work.group_id)
            if not group_dir.exists():
                return {}
            try:
                paths = sorted(group_dir.glob("g[0-9][0-9][0-9][0-9][0-9]*.pt"))
            except OSError as exc:
                raise StorageUnavailableError(
                    f"failed to list rollout checkpoints in {group_dir}"
                ) from exc

            records: dict[int, CompletedSiblingRecord] = {}
            for path in paths:
                record, _ = self._load_record_file(path)
                expected_path = self._record_path(
                    record.run_id, record.group_id, record.generation_index
                )
                if path != expected_path:
                    raise CorruptCheckpointError(
                        f"checkpoint identity at {path} resolves to {expected_path}"
                    )
                self._validate_record_for_work(record, work, path)
                if record.generation_index in records:
                    raise CorruptCheckpointError(
                        f"duplicate generation index {record.generation_index} in {group_dir}"
                    )
                records[record.generation_index] = record
            return records

    def fence(self, run_id: str, group_id: str, min_attempt_id: int) -> int:
        """Durably advance a group's minimum accepted attempt.

        Returns the resulting fence. Calling this with an older value is an
        idempotent no-op; fences never move backward.
        """
        _validate_path_component(run_id, "run_id")
        _validate_path_component(group_id, "group_id")
        _validate_nonnegative_int(min_attempt_id, "min_attempt_id")

        with self._locked_group(run_id, group_id):
            current = self._load_fence_unlocked(run_id, group_id)
            if min_attempt_id <= current:
                return current
            fence_payload = {
                "schema_version": ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
                "run_id": run_id,
                "group_id": group_id,
                "min_attempt_id": min_attempt_id,
            }
            fence_payload["checksum"] = _json_checksum(fence_payload)
            encoded = json.dumps(
                fence_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            self._atomic_write(self._fence_path(run_id, group_id), encoded)
            return min_attempt_id

    def get_fence(self, run_id: str, group_id: str) -> int:
        """Return the durable minimum accepted attempt for a group."""
        _validate_path_component(run_id, "run_id")
        _validate_path_component(group_id, "group_id")
        with self._locked_group(run_id, group_id):
            return self._load_fence_unlocked(run_id, group_id)

    def validate_attempt(self, run_id: str, group_id: str, attempt_id: int) -> None:
        """Raise if an attempt is older than the group's durable fence."""
        _validate_path_component(run_id, "run_id")
        _validate_path_component(group_id, "group_id")
        _validate_nonnegative_int(attempt_id, "attempt_id")
        with self._locked_group(run_id, group_id):
            min_attempt_id = self._load_fence_unlocked(run_id, group_id)
            if attempt_id < min_attempt_id:
                raise StaleAttemptError(
                    f"attempt {attempt_id} for {run_id}/{group_id} is older than "
                    f"durable fence {min_attempt_id}"
                )

    def delete_group(self, run_id: str, group_id: str) -> None:
        """Idempotently delete every sibling and fence for a group."""
        _validate_path_component(run_id, "run_id")
        _validate_path_component(group_id, "group_id")
        with self._locked_group(run_id, group_id):
            group_dir = self._group_dir(run_id, group_id)
            if not group_dir.exists():
                return
            try:
                shutil.rmtree(group_dir)
                _fsync_directory(group_dir.parent)
            except OSError as exc:
                raise StorageUnavailableError(
                    f"failed to delete rollout checkpoint group {group_dir}"
                ) from exc

    @contextmanager
    def _locked_group(self, run_id: str, group_id: str) -> Iterator[None]:
        """Serialize operations for one group while allowing other groups."""
        key = (run_id, group_id)
        with self._group_locks_guard:
            entry = self._group_locks.get(key)
            if entry is None:
                entry = _GroupLockEntry(lock=threading.Lock())
                self._group_locks[key] = entry
            entry.users += 1

        try:
            with entry.lock:
                yield
        finally:
            with self._group_locks_guard:
                entry.users -= 1
                if entry.users == 0:
                    del self._group_locks[key]

    def _validate_record_for_work(
        self,
        record: CompletedSiblingRecord,
        work: RolloutWorkItem,
        path: Path,
    ) -> None:
        expected = {
            "run_id": work.run_id,
            "group_id": work.group_id,
            "prompt_id": work.prompt_id,
            "policy_version": work.policy_version,
            "prompt_fingerprint": work.prompt_fingerprint,
            "sampling_fingerprint": work.sampling_fingerprint,
            "tokenizer_fingerprint": work.tokenizer_fingerprint,
        }
        actual = {
            "run_id": record.run_id,
            "group_id": record.group_id,
            "prompt_id": record.prompt_id,
            "policy_version": record.policy_version,
            "prompt_fingerprint": record.prompt_fingerprint,
            "sampling_fingerprint": record.sampling_fingerprint,
            "tokenizer_fingerprint": record.tokenizer_fingerprint,
        }
        mismatches = {
            key: (actual[key], expected_value)
            for key, expected_value in expected.items()
            if actual[key] != expected_value
        }
        if mismatches:
            raise IncompatibleCheckpointError(
                f"checkpoint {path} does not match rollout work: {mismatches}"
            )
        if record.generation_index >= work.num_generations:
            raise IncompatibleCheckpointError(
                f"checkpoint {path} has generation index {record.generation_index}, "
                f"but work expects {work.num_generations} generations"
            )

    def _load_record_file(self, path: Path) -> tuple[CompletedSiblingRecord, str]:
        try:
            file_bytes = path.read_bytes()
        except OSError as exc:
            raise StorageUnavailableError(f"failed to read checkpoint {path}") from exc

        payload_bytes, record_checksum = _decode_file(file_bytes, path)
        try:
            payload = torch.load(
                io.BytesIO(payload_bytes), map_location="cpu", weights_only=False
            )
        except (
            EOFError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.PickleError,
        ) as exc:
            raise CorruptCheckpointError(
                f"failed to deserialize rollout checkpoint {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise CorruptCheckpointError(
                f"rollout checkpoint payload at {path} is not a dictionary"
            )
        try:
            actual_record_checksum = _semantic_checksum(payload)
        except ValueError as exc:
            raise CorruptCheckpointError(
                f"rollout checkpoint {path} contains an unsupported payload"
            ) from exc
        if actual_record_checksum != record_checksum:
            raise CorruptCheckpointError(
                f"semantic checksum mismatch for rollout checkpoint {path}"
            )
        return _payload_to_record(payload, path), record_checksum

    def _load_fence_unlocked(self, run_id: str, group_id: str) -> int:
        path = self._fence_path(run_id, group_id)
        if not path.exists():
            return 0
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise CorruptCheckpointError(
                f"invalid UTF-8 in attempt fence {path}"
            ) from exc
        except OSError as exc:
            raise StorageUnavailableError(
                f"failed to read attempt fence {path}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorruptCheckpointError(
                f"invalid attempt fence JSON at {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise CorruptCheckpointError(f"attempt fence at {path} is not a dictionary")
        expected = {
            "schema_version": ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "group_id": group_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise CorruptCheckpointError(
                    f"attempt fence {path} has invalid {key}: {payload.get(key)!r}"
                )
        min_attempt_id = payload.get("min_attempt_id")
        if (
            not isinstance(min_attempt_id, int)
            or isinstance(min_attempt_id, bool)
            or min_attempt_id < 0
        ):
            raise CorruptCheckpointError(
                f"attempt fence {path} has invalid min_attempt_id"
            )
        checksum = payload.get("checksum")
        fence_body = {key: value for key, value in payload.items() if key != "checksum"}
        if not isinstance(checksum, str) or checksum != _json_checksum(fence_body):
            raise CorruptCheckpointError(f"attempt fence {path} failed its checksum")
        return min_attempt_id

    def _atomic_write(self, path: Path, data: bytes) -> None:
        _create_directory_durably(path.parent)
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
        except OSError as exc:
            raise StorageUnavailableError(
                f"failed to create temporary checkpoint for {path}"
            ) from exc

        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, path)
            _fsync_directory(path.parent)
        except OSError as exc:
            raise StorageUnavailableError(
                f"failed to atomically commit rollout checkpoint {path}"
            ) from exc
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                # The primary write exception, if any, is more actionable. A
                # stale uniquely named temp file is ignored during recovery.
                pass

    def _group_dir(self, run_id: str, group_id: str) -> Path:
        _validate_path_component(run_id, "run_id")
        _validate_path_component(group_id, "group_id")
        return self.root_dir / run_id / group_id

    def _record_path(self, run_id: str, group_id: str, generation_index: int) -> Path:
        _validate_nonnegative_int(generation_index, "generation_index")
        return self._group_dir(run_id, group_id) / f"g{generation_index:05d}.pt"

    def _fence_path(self, run_id: str, group_id: str) -> Path:
        return self._group_dir(run_id, group_id) / "fence.json"


def _record_to_payload(record: CompletedSiblingRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "run_id": record.run_id,
        "group_id": record.group_id,
        "prompt_id": record.prompt_id,
        "generation_index": record.generation_index,
        "attempt_id": record.attempt_id,
        "policy_version": record.policy_version,
        "prompt_fingerprint": record.prompt_fingerprint,
        "sampling_fingerprint": record.sampling_fingerprint,
        "tokenizer_fingerprint": record.tokenizer_fingerprint,
        "phase": record.phase,
        "completion": {
            "message_log": record.completion.message_log,
            "env_extras": record.completion.env_extras,
            "truncated": record.completion.truncated,
            "reward": record.completion.reward,
        },
        "sample_metrics": record.sample_metrics,
    }


def _payload_to_record(payload: dict[str, Any], path: Path) -> CompletedSiblingRecord:
    required_keys = {
        "schema_version",
        "run_id",
        "group_id",
        "prompt_id",
        "generation_index",
        "attempt_id",
        "policy_version",
        "prompt_fingerprint",
        "sampling_fingerprint",
        "tokenizer_fingerprint",
        "phase",
        "completion",
        "sample_metrics",
    }
    missing = required_keys - payload.keys()
    if missing:
        raise CorruptCheckpointError(
            f"rollout checkpoint {path} is missing fields: {sorted(missing)}"
        )
    if payload["schema_version"] != ROLLOUT_CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            f"rollout checkpoint {path} uses schema {payload['schema_version']!r}; "
            f"expected {ROLLOUT_CHECKPOINT_SCHEMA_VERSION}"
        )
    completion_payload = payload["completion"]
    if not isinstance(completion_payload, dict):
        raise CorruptCheckpointError(
            f"rollout checkpoint {path} has an invalid completion payload"
        )
    completion_keys = {"message_log", "env_extras", "truncated", "reward"}
    missing_completion = completion_keys - completion_payload.keys()
    if missing_completion:
        raise CorruptCheckpointError(
            f"rollout checkpoint {path} completion is missing fields: "
            f"{sorted(missing_completion)}"
        )
    try:
        return CompletedSiblingRecord(
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            group_id=payload["group_id"],
            prompt_id=payload["prompt_id"],
            generation_index=payload["generation_index"],
            attempt_id=payload["attempt_id"],
            policy_version=payload["policy_version"],
            prompt_fingerprint=payload["prompt_fingerprint"],
            sampling_fingerprint=payload["sampling_fingerprint"],
            tokenizer_fingerprint=payload["tokenizer_fingerprint"],
            phase=payload["phase"],
            completion=Completion(
                message_log=completion_payload["message_log"],
                env_extras=completion_payload["env_extras"],
                truncated=completion_payload["truncated"],
                reward=completion_payload["reward"],
            ),
            sample_metrics=payload["sample_metrics"],
        )
    except (TypeError, ValueError) as exc:
        raise CorruptCheckpointError(
            f"rollout checkpoint {path} failed schema validation"
        ) from exc


def _serialize_payload(payload: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    try:
        torch.save(payload, buffer)
    except (OSError, RuntimeError, TypeError, ValueError, pickle.PickleError) as exc:
        raise RolloutCheckpointError(
            "failed to serialize completed rollout sibling"
        ) from exc
    return buffer.getvalue()


def _encode_file(*, payload_bytes: bytes, record_checksum: str) -> bytes:
    header = {
        "file_format_version": ROLLOUT_CHECKPOINT_SCHEMA_VERSION,
        "payload_checksum": hashlib.sha256(payload_bytes).hexdigest(),
        "record_checksum": record_checksum,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return b"".join(
        (
            _FILE_MAGIC,
            len(header_bytes).to_bytes(_HEADER_LENGTH_BYTES, "big"),
            header_bytes,
            payload_bytes,
        )
    )


def _decode_file(file_bytes: bytes, path: Path) -> tuple[bytes, str]:
    prefix_size = len(_FILE_MAGIC) + _HEADER_LENGTH_BYTES
    if len(file_bytes) < prefix_size or not file_bytes.startswith(_FILE_MAGIC):
        raise CorruptCheckpointError(f"invalid rollout checkpoint header at {path}")
    header_start = len(_FILE_MAGIC) + _HEADER_LENGTH_BYTES
    header_length = int.from_bytes(file_bytes[len(_FILE_MAGIC) : header_start], "big")
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
        raise CorruptCheckpointError(
            f"invalid rollout checkpoint header length at {path}: {header_length}"
        )
    payload_start = header_start + header_length
    if payload_start >= len(file_bytes):
        raise CorruptCheckpointError(f"truncated rollout checkpoint at {path}")
    try:
        header = json.loads(file_bytes[header_start:payload_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptCheckpointError(
            f"invalid rollout checkpoint metadata at {path}"
        ) from exc
    if not isinstance(header, dict):
        raise CorruptCheckpointError(
            f"rollout checkpoint metadata at {path} is not a dictionary"
        )
    if header.get("file_format_version") != ROLLOUT_CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            f"rollout checkpoint {path} uses file format "
            f"{header.get('file_format_version')!r}; expected "
            f"{ROLLOUT_CHECKPOINT_SCHEMA_VERSION}"
        )
    payload_bytes = file_bytes[payload_start:]
    payload_checksum = header.get("payload_checksum")
    if not isinstance(payload_checksum, str):
        raise CorruptCheckpointError(
            f"rollout checkpoint {path} has no payload checksum"
        )
    if not isinstance(header.get("record_checksum"), str):
        raise CorruptCheckpointError(
            f"rollout checkpoint {path} has no semantic checksum"
        )
    actual_payload_checksum = hashlib.sha256(payload_bytes).hexdigest()
    if actual_payload_checksum != payload_checksum:
        raise CorruptCheckpointError(
            f"payload checksum mismatch for rollout checkpoint {path}"
        )
    return payload_bytes, header["record_checksum"]


def _semantic_checksum(value: Any) -> str:
    digest = hashlib.sha256()
    _update_semantic_digest(digest, value, "payload")
    return digest.hexdigest()


def _json_checksum(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_semantic_digest(digest: Any, value: Any, location: str) -> None:
    def emit(tag: bytes, data: bytes = b"") -> None:
        digest.update(tag)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    if value is None:
        emit(b"none")
    elif isinstance(value, bool):
        emit(b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        emit(b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        if math.isnan(value):
            emit(b"float", b"nan")
        else:
            emit(b"float", struct.pack(">d", value))
    elif isinstance(value, str):
        emit(b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        emit(b"bytes", value)
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{location} contains a non-CPU tensor")
        if value.layout != torch.strided:
            raise ValueError(
                f"{location} contains unsupported tensor layout {value.layout}"
            )
        tensor = value.detach().contiguous()
        emit(b"tensor-dtype", str(tensor.dtype).encode("ascii"))
        emit(b"tensor-shape", json.dumps(list(tensor.shape)).encode("ascii"))
        emit(
            b"tensor-data",
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    elif isinstance(value, np.ndarray):
        emit(b"ndarray-dtype", value.dtype.str.encode("ascii"))
        emit(b"ndarray-shape", json.dumps(list(value.shape)).encode("ascii"))
        if value.dtype == object:
            for index, item in enumerate(value.flat):
                _update_semantic_digest(digest, item, f"{location}[{index}]")
        else:
            emit(b"ndarray-data", np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.generic):
        _update_semantic_digest(digest, value.item(), location)
    elif isinstance(value, dict):
        emit(b"dict-size", str(len(value)).encode("ascii"))
        sorted_items = sorted(
            value.items(), key=lambda item: _semantic_checksum(item[0])
        )
        for key, item in sorted_items:
            _update_semantic_digest(digest, key, f"{location}.key")
            _update_semantic_digest(digest, item, f"{location}[{key!r}]")
    elif isinstance(value, list):
        emit(b"list-size", str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _update_semantic_digest(digest, item, f"{location}[{index}]")
    elif isinstance(value, tuple):
        emit(b"tuple-size", str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _update_semantic_digest(digest, item, f"{location}[{index}]")
    else:
        raise ValueError(
            f"{location} contains unsupported checkpoint type {type(value).__name__}"
        )


def _validate_cpu_tree(value: Any, location: str) -> None:
    if isinstance(value, Completion):
        _validate_cpu_tree(value.message_log, f"{location}.message_log")
        _validate_cpu_tree(value.env_extras, f"{location}.env_extras")
        _validate_cpu_tree(value.truncated, f"{location}.truncated")
        _validate_cpu_tree(value.reward, f"{location}.reward")
        return
    # Semantic hashing validates the complete supported type tree and CPU
    # placement without persisting anything.
    _semantic_checksum(value)


def _validate_path_component(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{field_name} must be a non-empty path component")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must not contain path separators")


def _validate_fingerprint(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must not be empty")


def _validate_nonempty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _create_directory_durably(path: Path) -> None:
    """Create a directory tree and persist every new parent entry."""
    missing_directories: list[Path] = []
    current = path
    while not current.exists():
        missing_directories.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageUnavailableError(
            f"failed to create rollout checkpoint directory {path}"
        ) from exc
    if not path.is_dir():
        raise StorageUnavailableError(
            f"rollout checkpoint path is not a directory: {path}"
        )

    # Creating ``a/b`` changes ``a``. Persist each such directory entry from
    # the top of the newly created tree down to the requested leaf.
    for directory in reversed(missing_directories):
        _fsync_directory(directory.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise StorageUnavailableError(
            f"failed to open directory for fsync: {path}"
        ) from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise StorageUnavailableError(f"failed to fsync directory: {path}") from exc
    finally:
        os.close(fd)
