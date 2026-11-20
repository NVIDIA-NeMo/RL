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

"""Apply the qualified NIXL lease backport before starting vLLM processes."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

BUNDLE_DIR = Path(__file__).parent / "vllm_patches" / "nixl-lease-v0.30"


@dataclass(frozen=True)
class FilePatch:
    path: Path
    original_sha256: str | None
    patched_sha256: str


@dataclass(frozen=True)
class PatchBundle:
    patch: Path
    patch_sha256: str
    files: tuple[FilePatch, ...]


@dataclass(frozen=True)
class OriginalFile:
    path: Path
    mode: int
    uid: int
    gid: int


@dataclass(frozen=True)
class PreparedFile:
    original: OriginalFile
    replacement: Path
    backup: Path | None


def digest(content: bytes) -> str:
    """Return the SHA-256 of file contents."""
    return hashlib.sha256(content).hexdigest()


def load_bundle(directory: Path) -> PatchBundle:
    """Load and validate the checked-in patch and file identities."""
    manifest = json.loads((directory / "manifest.json").read_text())
    files: list[FilePatch] = []
    for record in manifest["files"]:
        path = Path(record["path"])
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError(f"Invalid patch path: {path}")
        files.append(
            FilePatch(path, record["original_sha256"], record["patched_sha256"])
        )
    if not files or len({item.path for item in files}) != len(files):
        raise ValueError("Patch manifest must contain distinct source files")
    bundle = PatchBundle(
        directory / "lease-verification.patch", manifest["patch_sha256"], tuple(files)
    )
    if digest(bundle.patch.read_bytes()) != bundle.patch_sha256:
        raise ValueError("Patch content does not match the manifest SHA-256")
    return bundle


def package_state(root: Path, bundle: PatchBundle) -> str:
    """Reject unsupported or partly patched packages before modifying any file."""
    if not root.is_dir():
        raise ValueError(f"vLLM package root is not a directory: {root}")
    states = []
    for item in bundle.files:
        path = root / item.path
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Source path escapes the selected package: {path}")
        observed = digest(path.read_bytes()) if path.exists() else None
        if observed == item.patched_sha256:
            states.append("applied")
        elif observed == item.original_sha256:
            states.append("original")
        else:
            raise ValueError(f"Unsupported vLLM source: {path} (SHA-256 {observed})")
    if len(set(states)) != 1:
        raise ValueError(
            f"Partly patched vLLM package: {root}; rebuild from the original package"
        )
    return states[0]


@contextmanager
def locked_packages(roots: Sequence[Path]) -> Iterator[None]:
    """Serialize installers in a consistent order across worker environments."""
    with ExitStack() as stack:
        for root in roots:
            lock = stack.enter_context((root / ".nrl-nixl-lease-patch.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def staged_contents(root: Path, bundle: PatchBundle) -> dict[Path, bytes]:
    """Apply the patch in an isolated Git worktree and verify every result."""
    with tempfile.TemporaryDirectory(prefix="nrl-nixl-lease-") as temporary:
        staging = Path(temporary)
        for item in bundle.files:
            original = root / item.path
            if original.exists():
                target = staging / "vllm" / item.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
        subprocess.run(
            ["git", "init", "--quiet", str(staging)], check=True, capture_output=True
        )
        for options in [["--check"], []]:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(staging),
                    "apply",
                    *options,
                    str(bundle.patch.resolve()),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        result = {}
        for item in bundle.files:
            content = (staging / "vllm" / item.path).read_bytes()
            if digest(content) != item.patched_sha256:
                raise ValueError(f"Patched source SHA-256 mismatch: {item.path}")
            result[item.path] = content
        return result


def prepare_replacement(path: Path, content: bytes, original: OriginalFile) -> None:
    """Write and sync a replacement with the original access metadata."""
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fchown(stream.fileno(), original.uid, original.gid)
        os.fchmod(stream.fileno(), original.mode)
        os.fsync(stream.fileno())


def install_contents(roots: Sequence[Path], contents: dict[Path, bytes]) -> None:
    """Prepare every replacement and rollback inode before publishing sources."""
    directories: list[Path] = []
    prepared: list[PreparedFile] = []
    published: list[PreparedFile] = []
    retain_backups = False
    try:
        for root in roots:
            for relative, content in contents.items():
                target = root / relative
                existing = target.stat() if target.exists() else None
                owner = existing if existing is not None else target.parent.stat()
                original = OriginalFile(
                    target,
                    stat.S_IMODE(existing.st_mode) if existing is not None else 0o644,
                    owner.st_uid,
                    owner.st_gid,
                )
                directory = Path(
                    tempfile.mkdtemp(prefix=".nrl-lease-", dir=target.parent)
                )
                directories.append(directory)
                backup = directory / "original" if existing is not None else None
                if backup is not None:
                    os.link(target, backup)
                replacement = directory / "replacement"
                prepare_replacement(replacement, content, original)
                prepared.append(PreparedFile(original, replacement, backup))
        try:
            for item in prepared:
                os.replace(item.replacement, item.original.path)
                published.append(item)
        except OSError as error:
            failures: list[OSError] = []
            for item in reversed(published):
                try:
                    if item.backup is None:
                        item.original.path.unlink()
                    else:
                        os.replace(item.backup, item.original.path)
                except OSError as failure:
                    failures.append(failure)
            if failures:
                retain_backups = True
                locations = ", ".join(map(str, directories))
                raise OSError(
                    f"Rollback failed; restore backups or rebuild packages. Staging retained: {locations}"
                ) from ExceptionGroup(
                    "Installation and rollback failures", [error, *failures]
                )
            raise
    finally:
        if not retain_backups:
            for directory in reversed(directories):
                shutil.rmtree(directory)


def apply_bundle(
    roots: Sequence[Path], bundle: PatchBundle, *, apply: bool
) -> dict[str, str]:
    """Check all selected packages, then optionally install the qualified patch."""
    selected = sorted({root.resolve() for root in roots})
    if not selected:
        raise ValueError("At least one explicit vLLM package root is required")
    if not apply:
        return {str(root): package_state(root, bundle) for root in selected}
    for root in selected:
        if not root.is_dir():
            raise ValueError(f"vLLM package root is not a directory: {root}")
    with locked_packages(selected):
        states = {root: package_state(root, bundle) for root in selected}
        pending = [root for root in selected if states[root] == "original"]
        if pending:
            contents = staged_contents(pending[0], bundle)
            install_contents(pending, contents)
        return {str(root): package_state(root, bundle) for root in selected}


def main() -> None:
    """Patch explicitly selected build-time packages before vLLM is imported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-package-root", type=Path, action="append", required=True
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true", help="Install the qualified backport"
    )
    mode.add_argument("--check", action="store_true", help="Check only; the default")
    args = parser.parse_args()
    try:
        if "vllm" in sys.modules:
            raise RuntimeError(
                "Run this tool before importing vLLM and restart every P/D process"
            )
        bundle = load_bundle(BUNDLE_DIR)
        states = apply_bundle(args.vllm_package_root, bundle, apply=args.apply)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"NIXL lease patch failed: {error}\n")
    print(
        json.dumps(
            {"mode": "apply" if args.apply else "check", "packages": states}, indent=2
        )
    )


if __name__ == "__main__":
    main()
