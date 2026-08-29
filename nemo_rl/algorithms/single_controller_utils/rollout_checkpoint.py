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

ROLLOUT_SNAPSHOT_SCHEMA_VERSION = 2
BOOTSTRAP_COMPATIBILITY_SCHEMA_VERSION = 2
BOOTSTRAP_DIRNAME = "bootstrap"
BOOTSTRAP_MANIFEST_FILENAME = "manifest.json"
ROLLOUT_SNAPSHOTS_DIRNAME = "rollout_snapshots"
ROLLOUT_SNAPSHOT_MANIFEST_FILENAME = "manifest.json"
ROLLOUT_SNAPSHOT_COMMITTED_FILENAME = "COMMITTED"

_SNAPSHOT_RE = re.compile(r"snapshot_(\d+)")

# Keep this projection limited to values needed to interpret persisted rollout
# state or execute missing siblings. Trainer-only and post-rollout settings do
# not belong in bootstrap compatibility.
_BOOTSTRAP_POLICY_FIELDS = frozenset(
    {
        "model_name",
        "pretrained_checkpoint",
        "hf_config_overrides",
        "max_total_sequence_length",
        "tokenizer",
    }
)
_BOOTSTRAP_GENERATION_FIELDS = frozenset(
    {
        "backend",
        "max_new_tokens",
        "stop_strings",
        "stop_token_ids",
        "temperature",
        "top_k",
        "top_p",
    }
)
_BOOTSTRAP_VLLM_FIELDS = frozenset(
    {
        "http_server_serving_chat_kwargs",
        "max_model_len",
        "reasoning_parser_plugin",
    }
)
_BOOTSTRAP_GRPO_FIELDS = frozenset(
    {
        "max_rollout_turns",
        "num_generations_per_prompt",
        "num_prompts_per_step",
        "seed",
    }
)
_BOOTSTRAP_DATA_FIELDS = frozenset(
    {
        "default",
        "max_input_seq_length",
        "shuffle",
        "train",
    }
)
_BOOTSTRAP_TOKEN_CAPTURE_FIELDS = frozenset(
    {
        "enabled",
        "min_valid_fraction_per_group",
        "mixed_weight_version_policy",
        "staging_partition",
    }
)
_BOOTSTRAP_ENV_RUNTIME_FIELDS = frozenset(
    {
        "apptainer_memory_limit_mb",
        "concurrency",
        "nemo_gym_log_dir",
        "num_gpu_nodes",
        "port_range_high",
        "port_range_low",
        "should_log_nemo_gym_responses",
        "skip_venv_if_present",
        "use_absolute_ip",
    }
)


def _select_fields(
    mapping: Mapping[str, Any] | None,
    fields: frozenset[str],
) -> dict[str, Any]:
    """Select explicitly rollout-semantic fields from one config section."""
    if mapping is None:
        return {}
    return {key: mapping[key] for key in sorted(fields) if key in mapping}


def _drop_runtime_fields(value: Any, runtime_fields: frozenset[str]) -> Any:
    """Recursively strip known operational leaves from an environment."""
    if isinstance(value, Mapping):
        return {
            key: _drop_runtime_fields(child, runtime_fields)
            for key, child in value.items()
            if key not in runtime_fields
        }
    if isinstance(value, list):
        return [_drop_runtime_fields(child, runtime_fields) for child in value]
    return value


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
    sampler_dispatch_index: int
    mutation_version: int
    rolled_back_train_group_count: int
    bootstrap_fingerprint: Optional[str]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RolloutSnapshotManifest:
        """Parse and validate a committed snapshot manifest."""
        required_ints = (
            "schema_version",
            "base_train_step",
            "trainer_version",
            "current_epoch",
            "sampler_dispatch_index",
            "mutation_version",
            "rolled_back_train_group_count",
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
            sampler_dispatch_index=raw["sampler_dispatch_index"],
            mutation_version=raw["mutation_version"],
            rolled_back_train_group_count=raw["rolled_back_train_group_count"],
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
                manifest.rolled_back_train_group_count,
            )
            < 0
        ):
            raise ValueError("rollout snapshot counters must be non-negative")
        if manifest.sampler_dispatch_index < -1:
            raise ValueError(
                "rollout snapshot sampler_dispatch_index must be at least -1"
            )
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedRolloutCheckpoint:
    """A committed rollout snapshot selected for startup recovery."""

    path: Path
    manifest: RolloutSnapshotManifest


@dataclass(frozen=True)
class BootstrapCompatibilityIdentity:
    """Rollout-semantic inputs that must match a trainer-version-zero cut."""

    schema_version: int
    model: Mapping[str, Any]
    generation: Mapping[str, Any]
    rollout: Mapping[str, Any]
    dataset: Mapping[str, Any]
    environment: Mapping[str, Any]
    sampler: Mapping[str, Any]
    token_capture: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_compatibility_identity(
    master_config: MasterConfig,
) -> BootstrapCompatibilityIdentity:
    """Project a full run config onto inputs that affect recovered rollouts.

    Dataset identity is intentionally retained because a bootstrap snapshot
    restores the dataloader cursor together with its unfinished prompt ledger.
    Cluster shape, logging, checkpoint paths, worker counts, and other runtime
    tuning are excluded so they may change across a restart.
    """
    dumped = master_config.model_dump(mode="json")
    policy = dumped.get("policy", {})
    generation = policy.get("generation", {})
    generation_identity = _select_fields(
        generation,
        _BOOTSTRAP_GENERATION_FIELDS,
    )
    vllm_identity = _select_fields(
        generation.get("vllm_cfg", {}),
        _BOOTSTRAP_VLLM_FIELDS,
    )
    if vllm_identity:
        generation_identity["vllm_cfg"] = vllm_identity

    async_rl = dumped.get("async_rl", {})
    sampler = async_rl.get("sampler", {})
    if not isinstance(sampler, Mapping):
        sampler = {}

    rollout = dumped.get("ppo") or dumped.get("grpo") or {}
    return BootstrapCompatibilityIdentity(
        schema_version=BOOTSTRAP_COMPATIBILITY_SCHEMA_VERSION,
        model=_select_fields(policy, _BOOTSTRAP_POLICY_FIELDS),
        generation=generation_identity,
        rollout=_select_fields(rollout, _BOOTSTRAP_GRPO_FIELDS),
        dataset=_select_fields(dumped.get("data", {}), _BOOTSTRAP_DATA_FIELDS),
        environment=_drop_runtime_fields(
            dumped.get("env", {}),
            _BOOTSTRAP_ENV_RUNTIME_FIELDS,
        ),
        sampler=dict(sampler),
        token_capture=_select_fields(
            dumped.get("token_capture", {}),
            _BOOTSTRAP_TOKEN_CAPTURE_FIELDS,
        ),
    )


def bootstrap_fingerprint(master_config: MasterConfig) -> str:
    """Hash rollout-semantic inputs needed to reuse a bootstrap snapshot.

    This is a compatibility guard, not a hash of the full training recipe.
    Operational settings are deliberately excluded so a restart may use a
    different cluster shape, checkpoint interval, or logging destination.
    """
    payload = json.dumps(
        bootstrap_compatibility_identity(master_config).to_dict(),
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


def reset_bootstrap_anchor(checkpoint_dir: Path, *, fingerprint: str) -> Path:
    """Discard skipped pre-step snapshots and start a new bootstrap lineage.

    This is used only when restore mode deliberately skips partial-rollout
    recovery and no trainer checkpoint exists. The user's restore choice makes
    the previous state intentionally unreachable; removing it also prevents a
    later periodic save from appending to an incompatible bootstrap anchor.
    """
    anchor = checkpoint_dir / BOOTSTRAP_DIRNAME
    snapshot_root = anchor / ROLLOUT_SNAPSHOTS_DIRNAME
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    manifest_path = anchor / BOOTSTRAP_MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()
    return ensure_bootstrap_anchor(checkpoint_dir, fingerprint=fingerprint)


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
