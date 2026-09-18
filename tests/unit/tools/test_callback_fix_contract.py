# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from tools.callback_fix_contract import (
    CALLBACK_EDITS,
    expected_callback_source,
    require_callback_sources,
)
from tools.check_visual_image_tools_launch import require_same_files


@pytest.mark.parametrize("relative", list(CALLBACK_EDITS))
def test_exact_callback_backport_and_unrelated_change_rejection(tmp_path, relative):
    root = Path(__file__).resolve().parents[3]
    fixed = (root / relative).read_text()
    baseline = fixed
    for before, after in reversed(CALLBACK_EDITS[relative]):
        assert baseline.count(after) == 1
        baseline = baseline.replace(after, before, 1)
    assert expected_callback_source(relative, baseline.encode()) == fixed.encode()
    assert expected_callback_source(relative, fixed.encode()) == fixed.encode()
    current, qualified = tmp_path / "current", tmp_path / "qualified"
    for directory, source in ((current, fixed), (qualified, baseline)):
        path = directory / relative
        path.parent.mkdir(parents=True)
        path.write_text(source)
    args = dict(current=current, qualified=qualified, paths=[relative])
    with pytest.raises(ValueError, match="source mismatch"):
        require_same_files(**args)
    assert len(require_same_files(**args, callback_fixes=True)) == 64
    (current / relative).write_text(fixed + "\n# Unrelated runtime edit\n")
    with pytest.raises(ValueError, match="source mismatch"):
        require_same_files(**args, callback_fixes=True)


def test_both_call_sites_are_required(tmp_path):
    root = Path(__file__).resolve().parents[3]
    require_callback_sources(root)
    for relative in CALLBACK_EDITS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((root / relative).read_text())
    require_callback_sources(tmp_path)
    relative = next(iter(CALLBACK_EDITS))
    before, after = CALLBACK_EDITS[relative][-1]
    path = tmp_path / relative
    path.write_text(path.read_text().replace(after, before, 1))
    with pytest.raises(ValueError, match="Missing approved"):
        require_callback_sources(tmp_path)


def test_native_pr_tests_precede_training_and_cannot_be_deselected():
    root = Path(__file__).resolve().parents[3]
    source = (root / "tools/image_tools_train_hsg.sh").read_text()
    end = source.index("exec /opt/nemo_rl_venv/bin/python -u")
    assert source.index("bash tools/run_callback_regressions.sh") < end
    runner = (root / "tools/run_callback_regressions.sh").read_text()
    assert "test_megatron_setup.py::TestFinalizeMegatronSetup" in runner
    assert "tests/unit_tests/training/test_runtime_callbacks.py" in runner
    assert runner.count("-m pytest -q --noconftest") == 2
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in runner
    assert 'export PYTHONPATH="$test_wheels' in runner
    assert "test_wheels" not in source
    assert "MIXED_CHECKPOINT_RETENTION_OK every=20 keep=all" in source


def test_test_wheels_are_complete_verified_and_required_before_submission(tmp_path):
    from tools.callback_test_runtime import PACKAGES, wheel_paths

    root = Path(__file__).resolve().parents[3]
    paths = wheel_paths(root)
    assert {path.name.split("-")[0] for path in paths} == PACKAGES
    (tmp_path / "uv.lock").write_bytes((root / "uv.lock").read_bytes())
    directory = tmp_path / "cluster_inputs/callback_test_wheels"
    directory.mkdir(parents=True)
    for path in paths:
        (directory / path.name).write_bytes(path.read_bytes())
    (directory / paths[0].name).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        wheel_paths(tmp_path)
    launcher = (root / "tools/launch_image_tools_train_hsg.sh").read_text()
    assert launcher.index("tools/callback_test_runtime.py") < launcher.index(
        "submit=(sbatch"
    )
