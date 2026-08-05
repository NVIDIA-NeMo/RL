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

"""Filesystem contract for frequent Single Controller rollout snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from nemo_rl.algorithms.single_controller_utils.config import MasterConfig

ROLLOUT_SNAPSHOT_SCHEMA_VERSION = 1
BOOTSTRAP_DIRNAME = "bootstrap"
BOOTSTRAP_MANIFEST_FILENAME = "manifest.json"
ROLLOUT_SNAPSHOTS_DIRNAME = "rollout_snapshots"
ROLLOUT_SNAPSHOT_MANIFEST_FILENAME = "manifest.json"
ROLLOUT_SNAPSHOT_COMMITTED_FILENAME = "COMMITTED"
ROLLOUT_SNAPSHOT_LATEST_FILENAME = "LATEST"

_SNAPSHOT_RE = re.compile(r"snapshot_(\d+)")

# These sections contain paths, credentials, cluster addresses, retention, and
# other operational values that may legitimately change across a restart.
# Every other top-level field is included by default so newly added semantic
# configuration cannot be silently omitted from bootstrap identity.
_BOOTSTRAP_RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "checkpointing",
        "cluster",
        "data_plane",
        "logger",
        "rollout_checkpointing",
        "token_capture",
    }
)


def _fsync_file(path: Path) -> None:
    """Flush one completed regular file to its backing filesystem."""
    with path.open("rb") as file_obj:
        os.fsync(file_obj.fileno())


def _fsync_directory(path: Path) -> None:
    """Flush directory-entry updates such as rename and replace."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_tree(root: Path) -> None:
    """Flush every snapshot payload before publishing its commit marker."""
    for directory, _, filenames in os.walk(root, topdown=False):
        directory_path = Path(directory)
        for filename in filenames:
            file_path = directory_path / filename
            if not file_path.is_symlink() and file_path.is_file():
                _fsync_file(file_path)
        _fsync_directory(directory_path)


def _snapshot_sequence(path: Path) -> int:
    match = _SNAPSHOT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a rollout snapshot directory: {path}")
    return int(match.group(1))


@dataclass(frozen=True)
class RolloutSnapshotManifest:
    """Identity binding one rollout-state cut to reconstructable trainer state."""

    schema_version: int
    base_train_step: int
    trainer_version: int
    current_epoch: int
    mutation_version: int
    bootstrap_fingerprint: Optional[str]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RolloutSnapshotManifest:
        """Parse and validate a committed snapshot manifest."""
        required_ints = (
            "schema_version",
            "base_train_step",
            "trainer_version",
            "current_epoch",
            "mutation_version",
        )
        for key in required_ints:
            value = raw.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"rollout snapshot manifest {key!r} must be an integer"
                )
        fingerprint = raw.get("bootstrap_fingerprint")
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise ValueError(
                "rollout snapshot bootstrap_fingerprint must be a string or null"
            )
        manifest = cls(
            schema_version=raw["schema_version"],
            base_train_step=raw["base_train_step"],
            trainer_version=raw["trainer_version"],
            current_epoch=raw["current_epoch"],
            mutation_version=raw["mutation_version"],
            bootstrap_fingerprint=fingerprint,
        )
        if manifest.schema_version != ROLLOUT_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported rollout snapshot schema version: "
                f"{manifest.schema_version}"
            )
        if (
            min(
                manifest.base_train_step,
                manifest.trainer_version,
                manifest.current_epoch,
                manifest.mutation_version,
            )
            < 0
        ):
            raise ValueError("rollout snapshot counters must be non-negative")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedRolloutCheckpoint:
    """A committed rollout snapshot selected for startup recovery."""

    path: Path
    manifest: RolloutSnapshotManifest


def bootstrap_fingerprint(master_config: MasterConfig) -> str:
    """Hash stable inputs needed to reconstruct the untrained policy exactly.

    Runtime-only settings such as logging paths, checkpoint deadlines, and the
    per-process Gym control token are deliberately excluded so a fresh process
    can resume the same bootstrap state with different operational settings.
    """
    dumped = master_config.model_dump(mode="json")
    identity = {
        key: value
        for key, value in dumped.items()
        if key not in _BOOTSTRAP_RUNTIME_CONFIG_FIELDS
    }
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def prune_bootstrap_snapshots(
    checkpoint_dir: Path,
    *,
    durable_trainer_checkpoint: Path,
) -> bool:
    """Remove trainer-version-zero snapshots once a trainer checkpoint exists."""
    if not durable_trainer_checkpoint.is_dir():
        raise FileNotFoundError(
            "cannot prune bootstrap snapshots without a durable trainer "
            f"checkpoint: {durable_trainer_checkpoint}"
        )
    snapshot_root = checkpoint_dir / BOOTSTRAP_DIRNAME / ROLLOUT_SNAPSHOTS_DIRNAME
    if not snapshot_root.is_dir():
        return False
    shutil.rmtree(snapshot_root)
    return True


def ensure_bootstrap_anchor(checkpoint_dir: Path, *, fingerprint: str) -> Path:
    """Create or validate the lightweight trainer-version-zero anchor."""
    anchor = checkpoint_dir / BOOTSTRAP_DIRNAME
    anchor.mkdir(parents=True, exist_ok=True)
    manifest_path = anchor / BOOTSTRAP_MANIFEST_FILENAME
    expected = {
        "schema_version": ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        "base_train_step": 0,
        "trainer_version": 0,
        "bootstrap_fingerprint": fingerprint,
    }
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text())
        if raw != expected:
            raise ValueError(
                "existing rollout bootstrap anchor does not match the current "
                f"trainer configuration: checkpoint={raw!r}, expected={expected!r}"
            )
        return anchor

    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
    _fsync_file(tmp_path)
    os.replace(tmp_path, manifest_path)
    _fsync_directory(anchor)
    _fsync_directory(anchor.parent)
    return anchor


def validate_bootstrap_anchor(anchor: Path, *, fingerprint: str) -> None:
    """Fail loudly when bootstrap snapshots belong to different initial state."""
    manifest_path = anchor / BOOTSTRAP_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"rollout bootstrap manifest is missing at {manifest_path}"
        )
    raw = json.loads(manifest_path.read_text())
    expected = {
        "schema_version": ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        "base_train_step": 0,
        "trainer_version": 0,
        "bootstrap_fingerprint": fingerprint,
    }
    if raw != expected:
        raise ValueError(
            "rollout bootstrap anchor does not match the current trainer "
            f"configuration: checkpoint={raw!r}, expected={expected!r}"
        )


def prepare_snapshot_paths(anchor: Path) -> tuple[Path, Path, int]:
    """Allocate the next temporary/final snapshot directory pair."""
    root = anchor / ROLLOUT_SNAPSHOTS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    sequences = [
        int(match.group(1))
        for child in root.iterdir()
        if child.is_dir() and (match := _SNAPSHOT_RE.fullmatch(child.name))
    ]
    sequence = max(sequences, default=0) + 1
    final_path = root / f"snapshot_{sequence:06d}"
    tmp_path = root / f"tmp_snapshot_{sequence:06d}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    return tmp_path, final_path, sequence


def commit_snapshot(
    tmp_path: Path,
    final_path: Path,
    *,
    keep_latest_k: int,
) -> None:
    """Atomically publish one validated snapshot and retain recent fallbacks."""
    if keep_latest_k < 1:
        raise ValueError("rollout snapshot retention must keep at least one snapshot")
    _fsync_tree(tmp_path)
    committed_path = tmp_path / ROLLOUT_SNAPSHOT_COMMITTED_FILENAME
    committed_path.write_text("committed\n")
    _fsync_file(committed_path)
    _fsync_directory(tmp_path)
    os.rename(tmp_path, final_path)

    root = final_path.parent
    _fsync_directory(root)
    latest_path = root / ROLLOUT_SNAPSHOT_LATEST_FILENAME
    latest_tmp = latest_path.with_suffix(".tmp")
    latest_tmp.write_text(final_path.name + "\n")
    _fsync_file(latest_tmp)
    os.replace(latest_tmp, latest_path)
    _fsync_directory(root)

    committed = sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir()
            and _SNAPSHOT_RE.fullmatch(child.name)
            and (child / ROLLOUT_SNAPSHOT_COMMITTED_FILENAME).is_file()
        ),
        key=_snapshot_sequence,
        reverse=True,
    )
    stale_snapshots = committed[keep_latest_k:]
    for stale in stale_snapshots:
        shutil.rmtree(stale)
    if stale_snapshots:
        _fsync_directory(root)


def resolve_latest_snapshot(
    anchor: Path,
    *,
    expected_train_step: int,
    expected_trainer_version: int,
    expected_bootstrap_fingerprint: Optional[str],
) -> Optional[ResolvedRolloutCheckpoint]:
    """Select the newest complete snapshot compatible with its trainer anchor."""
    root = anchor / ROLLOUT_SNAPSHOTS_DIRNAME
    if not root.is_dir():
        return None

    candidates = sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and _SNAPSHOT_RE.fullmatch(child.name)
        ),
        key=_snapshot_sequence,
        reverse=True,
    )
    errors: list[str] = []
    for candidate in candidates:
        if not (candidate / ROLLOUT_SNAPSHOT_COMMITTED_FILENAME).is_file():
            continue
        manifest_path = candidate / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME
        if not manifest_path.is_file():
            errors.append(f"{candidate.name}: missing manifest")
            continue
        try:
            raw = json.loads(manifest_path.read_text())
            manifest = RolloutSnapshotManifest.from_mapping(raw)
        except (json.JSONDecodeError, OSError, ValueError) as error:
            errors.append(f"{candidate.name}: {error}")
            continue
        if (
            manifest.base_train_step != expected_train_step
            or manifest.trainer_version != expected_trainer_version
            or manifest.bootstrap_fingerprint != expected_bootstrap_fingerprint
        ):
            errors.append(f"{candidate.name}: trainer-anchor mismatch")
            continue
        return ResolvedRolloutCheckpoint(candidate, manifest)

    if errors:
        raise ValueError(
            "no committed rollout snapshot matches the selected trainer anchor: "
            + "; ".join(errors)
        )
    return None
