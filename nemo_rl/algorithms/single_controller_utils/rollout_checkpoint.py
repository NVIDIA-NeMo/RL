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
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Optional

from nemo_rl.algorithms.single_controller_utils.config import MasterConfig

ROLLOUT_SNAPSHOT_SCHEMA_VERSION = 3
BOOTSTRAP_COMPATIBILITY_SCHEMA_VERSION = 7
BOOTSTRAP_DIRNAME = "bootstrap"
BOOTSTRAP_MANIFEST_FILENAME = "manifest.json"
ROLLOUT_SNAPSHOTS_DIRNAME = "rollout_snapshots"
ROLLOUT_SNAPSHOT_MANIFEST_FILENAME = "manifest.json"

_SNAPSHOT_RE = re.compile(r"snapshot_(\d+)")
_TMP_SNAPSHOT_RE = re.compile(r"tmp_snapshot_(\d+)")
_TRASH_SNAPSHOT_RE = re.compile(r"trash_snapshot_(\d+)")

# A bootstrap fingerprint is fail-closed: every config value participates unless
# this one denylist says it is operational. ``**`` matches any number of mapping
# levels, which keeps nested runtime fields and credential-shaped keys concise.
# User-defined exclusions are appended from RolloutCheckpointConfig.
_BOOTSTRAP_FINGERPRINT_EXCLUDED_PATHS = frozenset(
    {
        "async_rl.diagnostics",
        "async_rl.generation_fleet_health",
        "async_rl.generation_router",
        "async_rl.stall_watchdog",
        "checkpointing",
        "cluster",
        "data.num_workers",
        "data.validation",
        "logger",
        "policy.generation.colocated",
        "policy.generation.port_range_high",
        "policy.generation.port_range_low",
        "policy.generation.val_temperature",
        "policy.generation.val_top_k",
        "policy.generation.val_top_p",
        "policy.generation.vllm_cfg.env_vars",
        "policy.generation.vllm_cfg.http_refit_api_key_env_var",
        "policy.generation.vllm_cfg.http_refit_server_port",
        "policy.generation.vllm_cfg.zmq_refit_server_port",
        "policy.optimizer",
        "policy.scheduler",
        "rollout_checkpointing",
        "token_capture.capture_dir",
        "token_capture.control_auth_token",
        "token_capture.control_timeout_s",
        "token_capture.num_reassembler_workers",
        "**.api_key",
        "**.apikey",
        "**.password",
        "**.secret",
        "**.token",
        "**.*_api_key",
        "**.*_password",
        "**.*_secret",
        "**.*_token",
        "env.**._copy",
        "env.**._inherit_from",
        "env.**.allow_openai_version_skew",
        "env.**.api_server_count",
        "env.**.apptainer_memory_limit_mb",
        "env.**.cache_dir",
        "env.**.component_name",
        "env.**.concurrency",
        "env.**.config_paths",
        "env.**.debug",
        "env.**.default_host",
        "env.**.disallowed_ports",
        "env.**.dry_run",
        "env.**.entrypoint",
        "env.**.global_aiohttp_connector_limit",
        "env.**.global_aiohttp_connector_limit_per_host",
        "env.**.head_server",
        "env.**.head_server_deps",
        "env.**.json",
        "env.**.model_call_capture_dir",
        "env.**.model_endpoint_readiness_timeout_seconds",
        "env.**.nemo_gym_log_dir",
        "env.**.num_gpu_nodes",
        "env.**.num_processes",
        "env.**.num_workers",
        "env.**.observability_enabled",
        "env.**.pip_install_verbose",
        "env.**.policy_base_url",
        "env.**.port_range_high",
        "env.**.port_range_low",
        "env.**.python_version",
        "env.**.query",
        "env.**.ray_head_node_address",
        "env.**.ray_worker_py_executable",
        "env.**.results_dir",
        "env.**.should_log_nemo_gym_responses",
        "env.**.skip_venv_if_present",
        "env.**.token_id_capture",
        "env.**.use_absolute_ip",
        "env.**.uv_cache_dir",
        "env.**.uv_pip_set_python",
        "env.**.uv_venv_dir",
        "env.**.verbose",
        "env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url",
        "env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.base_url",
    }
)


def _path_matches(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    """Return whether one segmented dotpath pattern matches a concrete path."""
    if not pattern:
        return not path
    if pattern[0] == "**":
        return _path_matches(pattern[1:], path) or (
            bool(path) and _path_matches(pattern, path[1:])
        )
    return (
        bool(path)
        and fnmatchcase(path[0], pattern[0])
        and _path_matches(pattern[1:], path[1:])
    )


def _drop_excluded_paths(
    value: Any,
    *,
    excluded_paths: tuple[tuple[str, ...], ...],
    path: tuple[str, ...] = (),
) -> Any:
    """Recursively remove denylisted mapping paths from a JSON config dump."""
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("fingerprinted config mappings must use string keys")
            child_path = (*path, key)
            if any(_path_matches(pattern, child_path) for pattern in excluded_paths):
                continue
            projected[key] = _drop_excluded_paths(
                child,
                excluded_paths=excluded_paths,
                path=child_path,
            )
        return projected
    if isinstance(value, list):
        projected_list: list[Any] = []
        for index, child in enumerate(value):
            child_path = (*path, str(index))
            if any(_path_matches(pattern, child_path) for pattern in excluded_paths):
                continue
            projected_list.append(
                _drop_excluded_paths(
                    child,
                    excluded_paths=excluded_paths,
                    path=child_path,
                )
            )
        return projected_list
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
    """Flush every snapshot payload before publishing its directory."""
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
    excluded_paths: tuple[str, ...]
    config: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "excluded_paths": list(self.excluded_paths),
            "config": self.config,
        }

    def fingerprint(self) -> str:
        """Return the canonical digest stored in snapshot manifests."""
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def bootstrap_compatibility_identity(
    master_config: MasterConfig,
) -> BootstrapCompatibilityIdentity:
    """Remove explicitly operational paths from the fail-closed run identity."""
    dumped = master_config.model_dump(mode="json")
    rollout_checkpointing = dumped.get("rollout_checkpointing", {})
    if not isinstance(rollout_checkpointing, Mapping):
        raise TypeError("rollout_checkpointing config must be a mapping")
    extra_excluded_paths = rollout_checkpointing.get(
        "extra_fingerprint_excluded_paths", []
    )
    if not isinstance(extra_excluded_paths, list) or not all(
        isinstance(path, str) for path in extra_excluded_paths
    ):
        raise TypeError("extra_fingerprint_excluded_paths must be a list of strings")
    excluded_paths = tuple(
        sorted(_BOOTSTRAP_FINGERPRINT_EXCLUDED_PATHS | frozenset(extra_excluded_paths))
    )
    parsed_excluded_paths = tuple(
        tuple(excluded_path.split(".")) for excluded_path in excluded_paths
    )
    return BootstrapCompatibilityIdentity(
        schema_version=BOOTSTRAP_COMPATIBILITY_SCHEMA_VERSION,
        excluded_paths=excluded_paths,
        config=_drop_excluded_paths(
            dumped,
            excluded_paths=parsed_excluded_paths,
        ),
    )


def bootstrap_fingerprint(master_config: MasterConfig) -> str:
    """Hash rollout-semantic inputs needed to reuse a bootstrap snapshot.

    This is a compatibility guard, not a hash of the full training recipe.
    Operational settings are deliberately excluded so a restart may use a
    different cluster shape, checkpoint interval, or logging destination.
    """
    return bootstrap_compatibility_identity(master_config).fingerprint()


_MISSING = object()


def _format_changed_value(value: Any) -> str:
    """Format one redacted compatibility value without flooding an error."""
    if value is _MISSING:
        return "<missing>"
    rendered = repr(value)
    if len(rendered) > 160:
        return rendered[:157] + "..."
    return rendered


def _compatibility_differences(
    checkpoint: Any,
    expected: Any,
    *,
    path: tuple[str, ...] = (),
) -> list[str]:
    """Describe changed compatibility leaves using user-facing dotpaths."""
    if isinstance(checkpoint, Mapping) and isinstance(expected, Mapping):
        differences: list[str] = []
        for key in sorted(set(checkpoint) | set(expected)):
            if key == "bootstrap_fingerprint" and not path:
                continue
            differences.extend(
                _compatibility_differences(
                    checkpoint.get(key, _MISSING),
                    expected.get(key, _MISSING),
                    path=(*path, key),
                )
            )
        return differences
    if checkpoint == expected:
        return []

    display_path = path
    if display_path[:2] == ("bootstrap_identity", "config"):
        display_path = display_path[2:]
    elif display_path[:1] == ("bootstrap_identity",):
        display_path = display_path[1:]
    name = ".".join(display_path) or "bootstrap_identity"
    return [
        f"{name}: {_format_changed_value(checkpoint)} -> "
        f"{_format_changed_value(expected)}"
    ]


def _bootstrap_anchor_manifest(
    identity: BootstrapCompatibilityIdentity,
) -> dict[str, Any]:
    """Build the bootstrap manifest from one self-consistent identity."""
    return {
        "schema_version": ROLLOUT_SNAPSHOT_SCHEMA_VERSION,
        "base_train_step": 0,
        "trainer_version": 0,
        "bootstrap_fingerprint": identity.fingerprint(),
        "bootstrap_identity": identity.to_dict(),
    }


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


def ensure_bootstrap_anchor(
    checkpoint_dir: Path,
    *,
    identity: BootstrapCompatibilityIdentity,
) -> Path:
    """Create or validate the lightweight trainer-version-zero anchor."""
    anchor = checkpoint_dir / BOOTSTRAP_DIRNAME
    anchor.mkdir(parents=True, exist_ok=True)
    manifest_path = anchor / BOOTSTRAP_MANIFEST_FILENAME
    expected = _bootstrap_anchor_manifest(identity)
    if manifest_path.is_file():
        validate_bootstrap_anchor(anchor, identity=identity)
        return anchor

    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
    _fsync_file(tmp_path)
    os.replace(tmp_path, manifest_path)
    _fsync_directory(anchor)
    _fsync_directory(anchor.parent)
    return anchor


def validate_bootstrap_anchor(
    anchor: Path,
    *,
    identity: BootstrapCompatibilityIdentity,
) -> None:
    """Validate a bootstrap anchor without modifying checkpoint state."""
    manifest_path = anchor / BOOTSTRAP_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"rollout bootstrap manifest is missing at {manifest_path}"
        )
    raw = json.loads(manifest_path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"rollout bootstrap manifest at {manifest_path} must be a mapping"
        )
    expected = _bootstrap_anchor_manifest(identity)
    if raw != expected:
        differences = _compatibility_differences(raw, expected)
        if not differences:
            differences = [
                "bootstrap_fingerprint: checkpoint digest does not match its "
                "persisted compatibility identity"
            ]
        visible = differences[:20]
        if len(differences) > len(visible):
            visible.append(f"... and {len(differences) - len(visible)} more change(s)")
        details = "\n".join(f"  {difference}" for difference in visible)
        raise ValueError(
            f"rollout bootstrap anchor at {manifest_path} is incompatible "
            "with the current rollout-semantic configuration. Changed "
            f"compatibility fields:\n{details}\n"
            "If a changed field is operational only, list its dotpath in "
            "rollout_checkpointing.extra_fingerprint_excluded_paths in both "
            "the original and restarted configurations. Otherwise reuse the "
            "original configuration or choose a new checkpoint_dir. Existing "
            "checkpoint state was not modified."
        )


def prepare_snapshot_paths(anchor: Path) -> tuple[Path, Path, int]:
    """Allocate the next temporary/final snapshot directory pair."""
    root = anchor / ROLLOUT_SNAPSHOTS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    garbage = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and (
            _TMP_SNAPSHOT_RE.fullmatch(child.name)
            or _TRASH_SNAPSHOT_RE.fullmatch(child.name)
        )
    ]
    for child in garbage:
        shutil.rmtree(child)
    if garbage:
        _fsync_directory(root)
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
    os.rename(tmp_path, final_path)

    root = final_path.parent
    _fsync_directory(root)

    committed = sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and _SNAPSHOT_RE.fullmatch(child.name)
        ),
        key=_snapshot_sequence,
        reverse=True,
    )
    stale_snapshots = committed[keep_latest_k:]
    for stale in stale_snapshots:
        trash = root / f"trash_{stale.name}"
        if trash.exists():
            shutil.rmtree(trash)
        os.rename(stale, trash)
        _fsync_directory(root)
        shutil.rmtree(trash)
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
