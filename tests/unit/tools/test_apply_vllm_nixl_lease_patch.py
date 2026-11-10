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
import importlib.util
import os
import stat
import sys
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
    assert stat.S_IMODE((roots[0] / "worker.py").stat().st_mode) == 0o640
    for root in roots:
        for name, content in after.items():
            assert (root / name).read_bytes() == content


def test_write_failure_rolls_back_all_packages(fixture_bundle, monkeypatch):
    roots, bundle, before, _ = fixture_bundle
    original = installer.replace_bytes
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise OSError("injected write failure")
        original(*args, **kwargs)

    monkeypatch.setattr(installer, "replace_bytes", fail_once)
    with pytest.raises(OSError, match="injected"):
        installer.apply_bundle(roots, bundle, apply=True)
    for root in roots:
        assert installer.package_state(root, bundle) == "original"
        assert not (root / "lease.py").exists()
        assert (root / "worker.py").read_bytes() == before["worker.py"]


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
    assert len(bundle.files) == 6


def test_tampered_patch_is_rejected(tmp_path):
    import shutil

    target = tmp_path / "bundle"
    shutil.copytree(installer.BUNDLE_DIR, target)
    (target / "lease-verification.patch").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        installer.load_bundle(target)
