# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic staging of the pinned Gym tree with patches and overlays."""

import json
from pathlib import Path
import subprocess

import pytest

from tools.super_rl import stage_gym

GOOD_PATCH = """--- a/nemo_gym/app.py
+++ b/nemo_gym/app.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""


def make_repo(root: Path, content: str) -> tuple[Path, str]:
    repo = root / "gym"
    (repo / "nemo_gym").mkdir(parents=True)
    (repo / "nemo_gym" / "app.py").write_text(content)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@x"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    return repo, head


def make_assets(root: Path, patch: str) -> Path:
    assets = root / "assets"
    (assets / "patches").mkdir(parents=True)
    (assets / "gym_overlays").mkdir()
    (assets / "patches" / "gym_value.patch").write_text(patch)
    (assets / "gym_overlays" / "helper.py").write_text("HELPER = True\n")
    return assets


def test_stage_applies_patches_and_overlays_then_publishes(tmp_path, monkeypatch):
    repo, head = make_repo(tmp_path, "VALUE = 1\n")
    monkeypatch.setattr(stage_gym, "GYM_BASE", head)
    monkeypatch.setattr(stage_gym, "ASSETS", make_assets(tmp_path, GOOD_PATCH))
    output = tmp_path / "runtime" / "gym"
    stage_gym.stage(repo, output)
    assert (output / "nemo_gym" / "app.py").read_text() == "VALUE = 2\n"
    assert (output / "nemo_gym" / "helper.py").read_text() == "HELPER = True\n"
    manifest = json.loads((output / "super-rl-overlay-manifest.json").read_text())
    assert manifest["base_commit"] == head
    assert set(manifest["overlays"]) == {
        "patches/gym_value.patch",
        "gym_overlays/helper.py",
    }
    assert [path.name for path in output.parent.iterdir()] == ["gym"]
    assert output.stat().st_mode & 0o077, "staged tree must not stay mkdtemp-private"
    with pytest.raises(FileExistsError):
        stage_gym.stage(repo, output)


def test_failed_patch_leaves_no_partial_output(tmp_path, monkeypatch):
    repo, head = make_repo(tmp_path, "VALUE = 1\n")
    monkeypatch.setattr(stage_gym, "GYM_BASE", head)
    bad = GOOD_PATCH.replace("-VALUE = 1", "-VALUE = 7")
    monkeypatch.setattr(stage_gym, "ASSETS", make_assets(tmp_path, bad))
    output = tmp_path / "runtime" / "gym"
    with pytest.raises(subprocess.CalledProcessError):
        stage_gym.stage(repo, output)
    assert not output.exists()
    assert list(output.parent.iterdir()) == []


def test_already_applied_patch_fails_instead_of_reversing(tmp_path, monkeypatch):
    # The pinned tree already contains the change: GNU patch would assume -R
    # and silently undo it without --forward.
    repo, head = make_repo(tmp_path, "VALUE = 2\n")
    monkeypatch.setattr(stage_gym, "GYM_BASE", head)
    monkeypatch.setattr(stage_gym, "ASSETS", make_assets(tmp_path, GOOD_PATCH))
    output = tmp_path / "runtime" / "gym"
    with pytest.raises(subprocess.CalledProcessError):
        stage_gym.stage(repo, output)
    assert not output.exists()
