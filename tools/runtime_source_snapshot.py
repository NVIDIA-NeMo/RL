#!/usr/bin/env python3
"""Create or verify an immutable, committed-source runtime snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBMODULES = (
    Path("3rdparty/Gym-workspace/Gym"),
    Path("3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"),
    Path("3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"),
)

RUNTIME_PATHS = (
    Path("nemo_rl"),
    Path("examples"),
    Path("tools"),
    Path("scripts"),
    Path("tests"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    *SUBMODULES,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _extract_commit(repo: Path, destination: Path, commit: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError(f"git archive produced no stdout for {repo}")
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
    finally:
        process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"git archive failed for {repo} with exit {return_code}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    mode = stat.S_IMODE(path.lstat().st_mode)
    if path.is_symlink():
        target = os.readlink(path)
        content_sha256 = hashlib.sha256(
            b"symlink\0" + target.encode("utf-8")
        ).hexdigest()
        size = len(target.encode("utf-8"))
        kind = "symlink"
    elif path.is_file():
        content_sha256 = _sha256_file(path)
        size = path.stat().st_size
        kind = "file"
    else:
        raise ValueError(f"unsupported snapshot entry: {path}")
    return {
        "path": relative,
        "kind": kind,
        "mode": mode,
        "size": size,
        "sha256": content_sha256,
    }


def _manifest_digest(records: list[dict[str, Any]]) -> str:
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def _inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for relative in RUNTIME_PATHS:
        target = root / relative
        if not target.exists() and not target.is_symlink():
            raise ValueError(f"runtime snapshot path is missing: {target}")
        if target.is_dir():
            paths.update(path for path in target.rglob("*") if not path.is_dir())
        else:
            paths.add(target)
    for path in sorted(paths):
        records.append(_record(path, root))
    if not records:
        raise ValueError(f"runtime snapshot is empty: {root}")
    return records, _manifest_digest(records)


def create_snapshot(
    repo: Path, destination: Path, manifest_output: Path
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"snapshot destination already exists: {destination}")
    destination = destination.resolve()
    destination.mkdir(parents=True)

    source_commit = _git(repo, "rev-parse", "HEAD")
    _extract_commit(repo, destination, source_commit)
    submodule_commits: dict[str, str] = {}
    for relative in SUBMODULES:
        submodule_repo = repo / relative
        commit = _git(submodule_repo, "rev-parse", "HEAD")
        submodule_commits[relative.as_posix()] = commit
        _extract_commit(submodule_repo, destination / relative, commit)

    files, digest = _inventory(destination)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": str(repo),
        "snapshot_directory": str(destination),
        "source_commit": source_commit,
        "submodule_commits": submodule_commits,
        "file_count": len(files),
        "runtime_manifest_sha256": digest,
        "files": files,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_output)
    return payload


def verify_snapshot(
    root: Path, manifest_path: Path, *, allow_extra: bool = False
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read runtime manifest: {error}") from error
    if not isinstance(expected, dict) or expected.get("schema_version") != 1:
        raise ValueError("runtime manifest must be a schema-version-1 JSON object")
    expected_files = expected.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("runtime manifest has no file inventory")
    if allow_extra:
        files = []
        for expected_record in expected_files:
            if not isinstance(expected_record, dict) or not isinstance(
                expected_record.get("path"), str
            ):
                raise ValueError("runtime manifest contains an invalid file record")
            path = root / expected_record["path"]
            if not path.exists() and not path.is_symlink():
                raise ValueError(f"runtime snapshot entry is missing: {path}")
            files.append(_record(path, root))
        digest = _manifest_digest(files)
    else:
        files, digest = _inventory(root)
    if files != expected_files:
        raise ValueError("runtime snapshot inventory differs from its launch manifest")
    if digest != expected.get("runtime_manifest_sha256"):
        raise ValueError(
            f"runtime snapshot digest is {digest}, expected "
            f"{expected.get('runtime_manifest_sha256')}"
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(root),
        "runtime_manifest": str(manifest_path.resolve(strict=True)),
        "file_count": len(files),
        "runtime_manifest_sha256": digest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-extra", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.create:
            if args.repo is None or args.destination is None or args.root is not None:
                raise ValueError("--create requires --repo and --destination")
            manifest = create_snapshot(args.repo, args.destination, args.manifest)
            report = {
                "schema_version": manifest["schema_version"],
                "status": "PASS",
                "snapshot_directory": manifest["snapshot_directory"],
                "runtime_source_manifest": str(args.manifest.resolve()),
                "source_commit": manifest["source_commit"],
                "submodule_commits": manifest["submodule_commits"],
                "file_count": manifest["file_count"],
                "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
            }
        else:
            if (
                args.root is None
                or args.repo is not None
                or args.destination is not None
            ):
                raise ValueError("--verify requires --root")
            report = verify_snapshot(
                args.root, args.manifest, allow_extra=args.allow_extra
            )
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        print(f"runtime source snapshot failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
