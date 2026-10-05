#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove builder metadata from a disposable Nano runtime filesystem."""

import argparse
import configparser
import hashlib
import io
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath


def build_users(passwd: bytes) -> set[str]:
    """Identify interactive build accounts without removing system accounts."""
    return {
        fields[0]
        for line in passwd.decode().splitlines()
        if len(fields := line.split(":")) == 7 and fields[5].startswith("/home/")
    }


def remove_path(name: str) -> bool:
    """Return whether an image-relative path is disposable author metadata."""
    path = PurePosixPath(name)
    if {".ruff_cache", ".pytest_cache", ".mypy_cache"}.intersection(path.parts):
        return True
    if (
        name.startswith("usr/local/cuda")
        and path.parent.name == "compat"
        and path.name.startswith(".")
        and path.name.endswith(".checked")
    ):
        return True
    prefixes = (
        "home/",
        "root/.cache/huggingface/",
        "root/.config/huggingface/",
        "root/.config/go/telemetry/",
        "root/.config/mlflow/",
        "root/.ssh/",
        "root/.aws/",
        "root/.docker/",
        "opt/nemo-rl/tests/unit/unit_results/",
    )
    if any(name == prefix[:-1] or name.startswith(prefix) for prefix in prefixes):
        return name != "home"
    if name in {"root/.netrc", "root/.git-credentials", "root/.bash_history"}:
        return True
    if ".git" in path.parts:
        metadata = path.parts[path.parts.index(".git") + 1 :]
        if "logs" in metadata or path.name in {
            "FETCH_HEAD",
            "ORIG_HEAD",
            "COMMIT_EDITMSG",
            "MERGE_MSG",
        }:
            return True
    if path.parent == PurePosixPath("opt/nano35-metadata"):
        return path.name.startswith("build-attempt-") or path.name == (
            "frozen-source-archive.sha256"
        )
    return False


def rewrite_path(name: str) -> bool:
    """Identify small metadata files whose contents require normalization."""
    path = PurePosixPath(name)
    return (
        name
        in {
            f"etc/{file}{backup}"
            for file in ("passwd", "group", "shadow", "gshadow", "subuid", "subgid")
            for backup in ("", "-")
        }
        or name in {"root/.gitconfig", "etc/gitconfig", "etc/machine-id"}
        or (".git" in path.parts and path.name in {"config", "config.worktree"})
        or name == "opt/nano35-metadata/request-stats-fix.json"
        or name
        == "opt/nemo-rl/examples/configs/recipes/llm/"
        "grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml"
    )


def clean_git_config(data: bytes) -> bytes:
    """Strip author identity, local includes and authentication configuration."""
    config = configparser.ConfigParser(interpolation=None, strict=False)
    config.read_string(data.decode())
    for section in list(config.sections()):
        kind = section.split()[0].lower()
        if kind in {"user", "credential", "http", "https", "include", "includeif"}:
            config.remove_section(section)
    output = io.StringIO()
    config.write(output)
    return output.getvalue().encode()


def rewrite_content(name: str, data: bytes, users: set[str]) -> bytes:
    """Clean selected records while preserving runtime and Git version state."""
    path = PurePosixPath(name)
    if path.suffix == ".yaml":
        # The historical expanded recipe retained the builder's project name.
        # Logging is disabled in the recipe; keep the neutral public default.
        return re.sub(rb"(?m)^( +project:)[^\n]*$", rb"\1 nano3.5-e2e", data)
    if name == "etc/machine-id":
        return b""
    if name == "opt/nano35-metadata/request-stats-fix.json":
        record = json.loads(data)
        for field in (
            "base_image",
            "native_regression_evidence",
            "gpu_refit_probe",
            "gpu_refit_job",
            "gpu_refit_gate",
            "host_verification",
        ):
            record.pop(field, None)
        return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    if path.name in {"config", "config.worktree", ".gitconfig", "gitconfig"}:
        return clean_git_config(data)
    account_file = path.name.rstrip("-")
    lines = []
    for line in data.decode().splitlines():
        fields = line.split(":")
        if fields[0] in users:
            continue
        if account_file == "group":
            fields[3] = ",".join(x for x in fields[3].split(",") if x not in users)
        elif account_file == "gshadow":
            for index in (2, 3):
                fields[index] = ",".join(
                    x for x in fields[index].split(",") if x not in users
                )
        lines.append(":".join(fields))
    return ("\n".join(lines) + ("\n" if lines else "")).encode()


def sanitize(root: Path) -> dict[str, int]:
    """Sanitize a disposable image root; never follow filesystem symlinks.

    Args:
        root: Container filesystem containing the Nano runtime-ready marker.

    Returns:
        Counts only, so the cleanup receipt does not reintroduce private names.
    """
    marker = root / "opt/nano35-metadata/runtime-ready"
    if not marker.is_file():
        raise ValueError("Expected a finalized Nano runtime filesystem")
    users = build_users((root / "etc/passwd").read_bytes())
    counts = {"removed_entries": 0, "rewritten_files": 0}
    for directory, directories, files in os.walk(root, followlinks=False):
        if Path(directory) == root:
            directories[:] = [
                name
                for name in directories
                if name in {"etc", "home", "root", "opt", "usr"}
            ]
            files = []
        for basename in directories.copy() + files:
            path = Path(directory) / basename
            relative = path.relative_to(root).as_posix()
            if remove_path(relative):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    directories.remove(basename)
                else:
                    path.unlink()
                counts["removed_entries"] += 1
            elif rewrite_path(relative) and path.is_file():
                if path.is_symlink():
                    raise ValueError("Refusing to rewrite symlinked image metadata")
                before = path.read_bytes()
                after = rewrite_content(relative, before, users)
                if after != before:
                    # Replace the inode so an unexpected hardlink cannot modify a
                    # runtime file outside the selected metadata path.
                    temporary = path.with_name(path.name + ".sanitized")
                    temporary.write_bytes(after)
                    shutil.copystat(path, temporary)
                    os.chown(temporary, path.stat().st_uid, path.stat().st_gid)
                    temporary.replace(path)
                    counts["rewritten_files"] += 1
    checksum_file = root / "opt/nano35-metadata/runtime-source-checksums.txt"
    if checksum_file.is_file():
        rows = []
        repo = root / "opt/nemo-rl"
        for line in checksum_file.read_text().splitlines():
            _, name = line.split(maxsplit=1)
            source = repo / name
            if not source.resolve().is_relative_to(repo.resolve()):
                raise ValueError("Source checksum refers outside the image repository")
            with source.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            rows.append(f"{digest}  {name}\n")
        text = "".join(rows)
        if checksum_file.read_text() != text:
            checksum_file.write_text(text)
            counts["rewritten_files"] += 1
    return counts


def main() -> None:
    """Run inside the final image build stage or against an unpacked image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
