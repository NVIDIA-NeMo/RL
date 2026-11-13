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

"""Safety checks for the build-time NIXL lease backport installer."""

import difflib
import errno
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/apply_vllm_nixl_lease_patch.py"
spec = importlib.util.spec_from_file_location("lease_installer", SCRIPT)
assert spec is not None and spec.loader is not None
installer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = installer
spec.loader.exec_module(installer)


@pytest.fixture
def fixture_bundle(tmp_path):
    """Create a real Git patch including a newly added module."""
    before = {
        "worker.py": b"old = True\n",
        "metadata.py": b"version = 1\n",
        "lease.py": None,
    }
    after = {
        "worker.py": b"old = False\n",
        "metadata.py": b"version = 2\n",
        "lease.py": b"nonce = 42\n",
    }
    chunks = []
    for name, content in before.items():
        chunks.extend(
            difflib.unified_diff(
                (content or b"").decode().splitlines(True),
                after[name].decode().splitlines(True),
                fromfile=f"a/vllm/{name}" if content is not None else "/dev/null",
                tofile=f"b/vllm/{name}",
            )
        )
    patch = tmp_path / "fixture.patch"
    patch.write_text("".join(chunks))
    bundle = installer.PatchBundle(
        patch,
        installer.digest(patch.read_bytes()),
        tuple(
            installer.FilePatch(
                Path(name),
                installer.digest(content) if content is not None else None,
                installer.digest(after[name]),
            )
            for name, content in before.items()
        ),
    )
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
        for name, content in before.items():
            if content is not None:
                (root / name).write_bytes(content)
    return roots, bundle, before, after


def test_check_is_read_only(fixture_bundle):
    roots, bundle, _, _ = fixture_bundle
    installer.apply_bundle(roots, bundle, apply=False)
    assert all(not (root / ".nrl-nixl-lease-patch.lock").exists() for root in roots)
    assert all(installer.package_state(root, bundle) == "original" for root in roots)


def test_all_packages_are_preflighted(fixture_bundle):
    roots, bundle, _, _ = fixture_bundle
    (roots[1] / "worker.py").write_bytes(b"unsupported\n")
    with pytest.raises(ValueError, match="Unsupported"):
        installer.apply_bundle(roots, bundle, apply=True)
    assert installer.package_state(roots[0], bundle) == "original"


def test_install_idempotent_preserves_hardlinks_and_modes(fixture_bundle, tmp_path):
    roots, bundle, before, after = fixture_bundle
    cached = tmp_path / "uv-cached.py"
    os.link(roots[0] / "worker.py", cached)
    (roots[0] / "worker.py").chmod(0o640)
    installer.apply_bundle(roots[:1], bundle, apply=True)
    installer.apply_bundle(roots, bundle, apply=True)
    installer.apply_bundle(roots, bundle, apply=True)
    assert cached.read_bytes() == before["worker.py"]
    metadata = (roots[0] / "worker.py").stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o640
    assert (metadata.st_uid, metadata.st_gid) == (
        cached.stat().st_uid,
        cached.stat().st_gid,
    )
    for root in roots:
        for name, content in after.items():
            assert (root / name).read_bytes() == content


def test_write_failure_rolls_back_all_packages(fixture_bundle, monkeypatch):
    roots, bundle, before, _ = fixture_bundle
    replace = installer.os.replace
    publications = 0

    def fail_publication(source, target):
        nonlocal publications
        if Path(source).name == "replacement":
            publications += 1
            if publications == 5:
                raise OSError(errno.ENOSPC, "injected persistent disk pressure")
        return replace(source, target)

    monkeypatch.setattr(installer.os, "replace", fail_publication)
    with pytest.raises(OSError, match="persistent disk pressure"):
        installer.apply_bundle(roots, bundle, apply=True)
    for root in roots:
        assert installer.package_state(root, bundle) == "original"
        assert not (root / "lease.py").exists()
        assert (root / "worker.py").read_bytes() == before["worker.py"]
        assert not list(root.glob(".nrl-lease-*"))


@pytest.mark.parametrize("failure_point", ["backup", "replacement"])
def test_persistent_preparation_failure_never_publishes(
    fixture_bundle, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_point: str
) -> None:
    roots, bundle, before, _ = fixture_bundle
    cached = tmp_path / "cached.py"
    os.link(roots[0] / "worker.py", cached)
    identity = cached.stat()
    operation = (
        installer.os.link
        if failure_point == "backup"
        else installer.prepare_replacement
    )
    calls = 0

    def exhaust_space(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError(errno.ENOSPC, "persistent space exhaustion")
        return operation(*args, **kwargs)

    if failure_point == "backup":
        monkeypatch.setattr(installer.os, "link", exhaust_space)
    else:
        monkeypatch.setattr(installer, "prepare_replacement", exhaust_space)
    with pytest.raises(OSError, match="persistent space exhaustion"):
        installer.apply_bundle(roots, bundle, apply=True)
    for root in roots:
        assert installer.package_state(root, bundle) == "original"
        assert not (root / "lease.py").exists()
        assert not list(root.glob(".nrl-lease-*"))
    assert cached.read_bytes() == before["worker.py"]
    assert (roots[0] / "worker.py").stat().st_ino == identity.st_ino
    assert cached.stat().st_nlink == identity.st_nlink


def test_rollback_does_not_allocate_or_copy_data(
    fixture_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, bundle, _, _ = fixture_bundle
    replace = installer.os.replace
    prepare = installer.prepare_replacement
    full = False
    publications = 0
    identities = {
        root / name: (root / name).stat().st_ino
        for root in roots
        for name in ["worker.py", "metadata.py"]
    }

    def prepare_until_full(*args, **kwargs):
        if full:
            raise OSError(errno.ENOSPC, "restoration cannot allocate storage")
        return prepare(*args, **kwargs)

    def replace_until_full(source, target):
        nonlocal publications, full
        if Path(source).name == "replacement":
            publications += 1
            if publications == 5:
                full = True
                raise OSError(errno.ENOSPC, "disk stays full")
        return replace(source, target)

    monkeypatch.setattr(installer, "prepare_replacement", prepare_until_full)
    monkeypatch.setattr(installer.os, "replace", replace_until_full)
    with pytest.raises(OSError, match="disk stays full"):
        installer.apply_bundle(roots, bundle, apply=True)
    for path, inode in identities.items():
        assert path.stat().st_ino == inode
    for root in roots:
        assert installer.package_state(root, bundle) == "original"
        assert not (root / "lease.py").exists()
        assert not list(root.glob(".nrl-lease-*"))


def test_unpreservable_ownership_fails_before_any_publish(
    fixture_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, bundle, _, _ = fixture_bundle

    def deny_ownership(*args) -> None:
        raise PermissionError(errno.EPERM, "cannot preserve owner")

    monkeypatch.setattr(installer.os, "fchown", deny_ownership)
    with pytest.raises(PermissionError, match="preserve owner"):
        installer.apply_bundle(roots, bundle, apply=True)
    for root in roots:
        assert installer.package_state(root, bundle) == "original"
        assert not list(root.glob(".nrl-lease-*"))


@pytest.mark.skipif(
    os.geteuid() != 0, reason="Changing to worker UID/GID requires root"
)
@pytest.mark.parametrize("fail_publication", [False, True])
def test_root_install_keeps_worker_read_access(
    fixture_bundle, monkeypatch: pytest.MonkeyPatch, fail_publication: bool
) -> None:
    roots, bundle, before, after = fixture_bundle
    with tempfile.TemporaryDirectory(prefix="lease-owner-", dir="/tmp") as directory:
        parent = Path(directory)
        parent.chmod(0o755)
        worker_roots = [parent / "first", parent / "second"]
        for source, destination in zip(roots, worker_roots):
            shutil.copytree(source, destination)
            destination.chmod(0o755)
            os.chown(destination, 1000, 1000)
            for path in destination.glob("*.py"):
                os.chown(path, 1000, 1000)
                path.chmod(0o640)
        replace = installer.os.replace
        publications = 0

        def publish(source, target):
            nonlocal publications
            if Path(source).name == "replacement":
                publications += 1
                if fail_publication and publications == 5:
                    raise OSError(errno.ENOSPC, "forced publication failure")
            return replace(source, target)

        monkeypatch.setattr(installer.os, "replace", publish)
        if fail_publication:
            with pytest.raises(OSError, match="forced publication"):
                installer.apply_bundle(worker_roots, bundle, apply=True)
        else:
            installer.apply_bundle(worker_roots, bundle, apply=True)
        for root in worker_roots:
            for name in ["worker.py", "metadata.py"]:
                path = root / name
                metadata = path.stat()
                assert (
                    metadata.st_uid,
                    metadata.st_gid,
                    stat.S_IMODE(metadata.st_mode),
                ) == (1000, 1000, 0o640)
                subprocess.run(
                    ["/bin/cat", str(path)],
                    user=1000,
                    group=1000,
                    extra_groups=(),
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
                assert path.read_bytes() == (
                    before[name] if fail_publication else after[name]
                )
            if fail_publication:
                assert not (root / "lease.py").exists()
            else:
                assert (root / "lease.py").stat().st_uid == 1000
                subprocess.run(
                    ["/bin/cat", str(root / "lease.py")],
                    user=1000,
                    group=1000,
                    extra_groups=(),
                    stdout=subprocess.DEVNULL,
                    check=True,
                )


def test_partial_package_is_rejected(fixture_bundle):
    roots, bundle, _, after = fixture_bundle
    (roots[0] / "worker.py").write_bytes(after["worker.py"])
    with pytest.raises(ValueError, match="Partly patched"):
        installer.apply_bundle(roots, bundle, apply=True)
    assert installer.package_state(roots[1], bundle) == "original"


def test_symlink_is_rejected(fixture_bundle, tmp_path):
    roots, bundle, _, _ = fixture_bundle
    target = roots[0] / "worker.py"
    outside = tmp_path / "outside.py"
    target.rename(outside)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        installer.apply_bundle(roots, bundle, apply=True)


def test_real_bundle_integrity():
    bundle = installer.load_bundle(installer.BUNDLE_DIR)
    assert len(bundle.files) == 7


def test_tampered_patch_is_rejected(tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(installer.BUNDLE_DIR, target)
    (target / "lease-verification.patch").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        installer.load_bundle(target)


def test_kernel_write_limit_leaves_original_packages(
    fixture_bundle, tmp_path: Path
) -> None:
    """Exercise real short writes under a child-only kernel size limit."""
    roots, bundle, before, _ = fixture_bundle
    cached = tmp_path / "cache-link.py"
    os.link(roots[0] / "worker.py", cached)
    program = """
import errno
import importlib.util
from pathlib import Path
import resource
import signal
import sys

spec = importlib.util.spec_from_file_location('limited_installer', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))
try:
    module.install_contents([Path(value) for value in sys.argv[2:]], {
        Path('worker.py'): b'x' * 100,
        Path('metadata.py'): b'x' * 2048,
        Path('lease.py'): b'nonce = 42\\n',
    })
except OSError as error:
    assert error.errno == errno.EFBIG
else:
    raise AssertionError('the kernel did not enforce the write limit')
"""
    subprocess.run(
        [sys.executable, "-c", program, str(SCRIPT), *map(str, roots)],
        check=True,
        capture_output=True,
        text=True,
    )
    for root in roots:
        assert installer.package_state(root, bundle) == "original"
        assert not (root / "lease.py").exists()
        assert not list(root.glob(".nrl-lease-*"))
    assert cached.read_bytes() == before["worker.py"]
    assert cached.stat().st_ino == (roots[0] / "worker.py").stat().st_ino


def test_restore_failure_keeps_backups_and_restores_other_roots(
    fixture_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, bundle, before, _ = fixture_bundle
    replace = installer.os.replace
    publications = 0

    def fail_one_restore(source, target):
        nonlocal publications
        if Path(source).name == "replacement":
            publications += 1
            if publications == 5:
                raise OSError(errno.ENOSPC, "publication failed")
        if Path(source).name == "original" and Path(target) == roots[1] / "worker.py":
            raise OSError(errno.EIO, "restore I/O failure")
        return replace(source, target)

    monkeypatch.setattr(installer.os, "replace", fail_one_restore)
    with pytest.raises(OSError, match="Staging retained"):
        installer.apply_bundle(roots, bundle, apply=True)
    assert installer.package_state(roots[0], bundle) == "original"
    assert not (roots[0] / "lease.py").exists()
    backups = [
        path
        for path in roots[1].glob(".nrl-lease-*/original")
        if path.read_bytes() == before["worker.py"]
    ]
    assert len(backups) == 1
    assert list(roots[0].glob(".nrl-lease-*"))
