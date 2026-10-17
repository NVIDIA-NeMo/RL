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

"""Verify a Torch Distributed Checkpoint before attempting to load it.

This tool asks Torch's ``FileSystemReader`` to parse the checkpoint metadata,
then verifies that every referenced storage object is a stable, non-empty
regular file and that every recorded byte range fits in that file. With
``--sha256``, it also emits a complete SHA256 manifest for the metadata and all
referenced storage files.

The command writes exactly one JSON object to stdout and exits nonzero on any
validation failure, making it suitable as a pre-load gate in a launcher::

    python tools/verify_torch_dcp_checkpoint.py \
        --checkpoint-dir /path/to/step_20/policy/weights --sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex


SCHEMA_VERSION = 1
METADATA_FILENAME = ".metadata"
HASH_CHUNK_BYTES = 8 * 1024 * 1024


class CheckpointVerificationError(ValueError):
    """Raised when a directory is not a complete, stable Torch DCP checkpoint."""


def _fail(message: str) -> NoReturn:
    raise CheckpointVerificationError(message)


def _stat_identity(file_stat: os.stat_result) -> tuple[int, int, int, int]:
    """Return fields that detect replacement or mutation during verification."""
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def _regular_file_stat(path: Path, label: str) -> os.stat_result:
    """Stat a required regular file while rejecting symlinks and empty files."""
    try:
        file_stat = path.lstat()
    except OSError as error:
        _fail(f"{label} is missing or inaccessible: {path}: {error}")
    if stat.S_ISLNK(file_stat.st_mode):
        _fail(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        _fail(f"{label} is not a regular file: {path}")
    if file_stat.st_size <= 0:
        _fail(f"{label} is empty: {path}")
    return file_stat


def _safe_storage_path(checkpoint_dir: Path, relative_path: object) -> tuple[str, Path]:
    """Resolve one DCP storage path without permitting checkpoint-root escape."""
    if not isinstance(relative_path, (str, os.PathLike)):
        _fail(f"storage relative_path has invalid type: {type(relative_path).__name__}")
    try:
        raw_path = os.fspath(relative_path)
    except TypeError as error:
        _fail(f"storage relative_path is invalid: {error}")
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        _fail(f"storage relative_path is invalid: {raw_path!r}")

    relative = Path(raw_path)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        _fail(f"unsafe storage relative_path: {raw_path!r}")

    candidate = checkpoint_dir / relative
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        _fail(
            f"referenced storage file is missing or inaccessible: {raw_path}: {error}"
        )
    try:
        resolved.relative_to(checkpoint_dir)
    except ValueError:
        _fail(f"storage relative_path escapes checkpoint directory: {raw_path!r}")
    if resolved != candidate.absolute():
        _fail(f"referenced storage path contains a symbolic link: {raw_path!r}")
    return relative.as_posix(), resolved


def _exact_nonnegative_int(value: object, label: str) -> int:
    """Validate a DCP offset without silently accepting booleans or floats."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(f"{label} must be a nonnegative integer, got {value!r}")
    return value


def _exact_positive_int(value: object, label: str) -> int:
    """Validate a DCP byte length without silently accepting zero or floats."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _fail(f"{label} must be a positive integer, got {value!r}")
    return value


def _sha256_file(path: Path, expected_stat: os.stat_result) -> str:
    """Hash a file and fail if it changes during the read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            if _stat_identity(os.fstat(handle.fileno())) != _stat_identity(
                expected_stat
            ):
                _fail(f"file changed before SHA256 read: {path}")
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
            if _stat_identity(os.fstat(handle.fileno())) != _stat_identity(
                expected_stat
            ):
                _fail(f"file changed during SHA256 read: {path}")
    except OSError as error:
        _fail(f"cannot read file for SHA256: {path}: {error}")
    return digest.hexdigest()


def _verify_readable_range(
    path: Path,
    expected_stat: os.stat_result,
    required_end: int,
) -> None:
    """Open a shard and prove that its final referenced byte is readable."""
    try:
        with path.open("rb") as handle:
            if _stat_identity(os.fstat(handle.fileno())) != _stat_identity(
                expected_stat
            ):
                _fail(f"referenced storage file changed before validation: {path}")
            handle.seek(required_end - 1)
            if len(handle.read(1)) != 1:
                _fail(
                    f"cannot read final referenced byte at offset {required_end - 1}: {path}"
                )
            if _stat_identity(os.fstat(handle.fileno())) != _stat_identity(
                expected_stat
            ):
                _fail(f"referenced storage file changed during validation: {path}")
    except OSError as error:
        _fail(f"cannot read referenced storage file: {path}: {error}")


def _read_metadata(
    checkpoint_dir: Path,
    metadata_path: Path,
    metadata_stat: os.stat_result,
) -> Metadata:
    """Parse and type-check checkpoint metadata with Torch's DCP reader."""
    # Torch wraps malformed or version-incompatible pickle data in different
    # exception types across releases. Normalize this external parser boundary.
    try:
        metadata = FileSystemReader(checkpoint_dir).read_metadata()
    except Exception as error:  # noqa: BLE001
        _fail(
            "Torch FileSystemReader rejected checkpoint metadata: "
            f"{type(error).__name__}: {error}"
        )
    if not isinstance(metadata, Metadata):
        _fail(
            "Torch FileSystemReader returned invalid metadata type: "
            f"{type(metadata).__name__}"
        )
    try:
        metadata_stat_after = metadata_path.lstat()
    except OSError as error:
        _fail(f"checkpoint metadata disappeared after parsing: {error}")
    if _stat_identity(metadata_stat_after) != _stat_identity(metadata_stat):
        _fail("checkpoint metadata changed while Torch was parsing it")
    return metadata


def verify_checkpoint(checkpoint_dir: Path, *, sha256: bool = False) -> dict[str, Any]:
    """Verify a Torch Distributed Checkpoint directory.

    Args:
        checkpoint_dir: Directory containing ``.metadata`` and DCP storage files.
        sha256: Whether to hash every byte of the metadata and referenced files.

    Returns:
        A JSON-serializable verification report.

    Raises:
        CheckpointVerificationError: If the checkpoint is malformed, incomplete,
            unsafe, corrupt by size, or changes during verification.
    """
    try:
        resolved_checkpoint_dir = checkpoint_dir.expanduser().resolve(strict=True)
    except OSError as error:
        _fail(
            f"checkpoint directory is missing or inaccessible: {checkpoint_dir}: {error}"
        )
    if not resolved_checkpoint_dir.is_dir():
        _fail(f"checkpoint path is not a directory: {resolved_checkpoint_dir}")

    metadata_path = resolved_checkpoint_dir / METADATA_FILENAME
    metadata_stat = _regular_file_stat(metadata_path, "checkpoint metadata")
    metadata = _read_metadata(
        resolved_checkpoint_dir,
        metadata_path,
        metadata_stat,
    )

    state_dict_metadata = metadata.state_dict_metadata
    if not isinstance(state_dict_metadata, Mapping) or not state_dict_metadata:
        _fail("Torch DCP metadata has no state_dict_metadata entries")
    storage_data = metadata.storage_data
    if not isinstance(storage_data, Mapping) or not storage_data:
        _fail("Torch DCP metadata has no storage_data records")

    file_records: dict[str, dict[str, Any]] = {}
    referenced_state_keys: set[str] = set()
    for metadata_index, storage_info in storage_data.items():
        if not isinstance(metadata_index, MetadataIndex):
            _fail(
                "storage_data contains an invalid metadata index type: "
                f"{type(metadata_index).__name__}"
            )
        if metadata_index.fqn not in state_dict_metadata:
            _fail(
                "storage_data references an unknown state_dict key: "
                f"{metadata_index.fqn!r}"
            )
        referenced_state_keys.add(metadata_index.fqn)

        try:
            relative_path = storage_info.relative_path
            raw_offset = storage_info.offset
            raw_length = storage_info.length
        except AttributeError as error:
            _fail(f"storage_data contains an invalid storage record: {error}")
        path_key, storage_path = _safe_storage_path(
            resolved_checkpoint_dir,
            relative_path,
        )
        offset = _exact_nonnegative_int(raw_offset, f"{path_key} offset")
        length = _exact_positive_int(raw_length, f"{path_key} length")
        required_end = offset + length

        record = file_records.setdefault(
            path_key,
            {
                "path": path_key,
                "absolute_path": storage_path,
                "required_end_offset": 0,
                "storage_records": 0,
            },
        )
        if record["absolute_path"] != storage_path:
            _fail(f"storage path normalizes inconsistently: {path_key!r}")
        record["required_end_offset"] = max(
            record["required_end_offset"],
            required_end,
        )
        record["storage_records"] += 1

    missing_state_keys = sorted(set(state_dict_metadata) - referenced_state_keys)
    if missing_state_keys:
        preview = missing_state_keys[:10]
        suffix = " ..." if len(missing_state_keys) > len(preview) else ""
        _fail(f"state_dict entries have no storage_data records: {preview!r}{suffix}")

    referenced_files: list[dict[str, Any]] = []
    file_stats: dict[str, os.stat_result] = {}
    for path_key in sorted(file_records):
        record = file_records[path_key]
        storage_path = record.pop("absolute_path")
        storage_stat = _regular_file_stat(storage_path, "referenced storage file")
        required_end = record["required_end_offset"]
        if required_end > storage_stat.st_size:
            _fail(
                f"referenced byte range ends at {required_end}, beyond file size "
                f"{storage_stat.st_size}: {path_key}"
            )
        _verify_readable_range(storage_path, storage_stat, required_end)
        record["size_bytes"] = storage_stat.st_size
        file_stats[path_key] = storage_stat
        referenced_files.append(record)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "metadata": {
            "path": METADATA_FILENAME,
            "size_bytes": metadata_stat.st_size,
            "state_dict_entries": len(state_dict_metadata),
            "storage_records": len(storage_data),
        },
        "referenced_file_count": len(referenced_files),
        "referenced_bytes": sum(record["size_bytes"] for record in referenced_files),
        "referenced_files": referenced_files,
        "sha256_enabled": sha256,
    }

    if sha256:
        manifest_files = [
            {
                "path": METADATA_FILENAME,
                "size_bytes": metadata_stat.st_size,
                "sha256": _sha256_file(metadata_path, metadata_stat),
            }
        ]
        for record in referenced_files:
            path_key = record["path"]
            manifest_files.append(
                {
                    "path": path_key,
                    "size_bytes": record["size_bytes"],
                    "sha256": _sha256_file(
                        resolved_checkpoint_dir / path_key,
                        file_stats[path_key],
                    ),
                }
            )
        canonical_manifest = json.dumps(
            manifest_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        report["sha256_manifest"] = {
            "files": manifest_files,
            "manifest_sha256": hashlib.sha256(canonical_manifest).hexdigest(),
        }

    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Torch DCP directory containing .metadata and storage files.",
    )
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Hash .metadata and every referenced storage file.",
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        help="Require the full SHA256 manifest to match this attested digest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the verifier CLI and emit one machine-readable JSON object."""
    args = _parser().parse_args(argv)
    # The CLI is a machine-readable validation boundary. Even unexpected Torch
    # errors must become a JSON failure instead of a traceback on stdout.
    try:
        if args.expected_manifest_sha256 and not args.sha256:
            _fail("--expected-manifest-sha256 requires --sha256")
        report = verify_checkpoint(args.checkpoint_dir, sha256=args.sha256)
        if args.expected_manifest_sha256:
            expected = args.expected_manifest_sha256.lower()
            if len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                _fail("expected manifest SHA256 must be 64 lowercase hex characters")
            actual = report["sha256_manifest"]["manifest_sha256"]
            if actual != expected:
                _fail(f"checkpoint manifest SHA256 is {actual}, expected {expected}")
    except Exception as error:  # noqa: BLE001
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "FAIL",
            "checkpoint_dir": str(args.checkpoint_dir.expanduser()),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        return 2
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
