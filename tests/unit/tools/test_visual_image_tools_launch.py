# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest

from tools.check_visual_image_tools_launch import (
    qualify,
    require_pytest_pass,
    require_same_files,
    runtime_files,
)


def test_source_comparison_rejects_mutation_and_missing_files(tmp_path):
    current, qualified = tmp_path / "current", tmp_path / "qualified"
    for root in (current, qualified):
        root.mkdir()
        (root / "app.py").write_text("same")
    identity = require_same_files(
        current=current, qualified=qualified, paths=["app.py"]
    )
    assert len(identity) == 64
    (current / "app.py").write_text("changed")
    with pytest.raises(ValueError, match="app.py"):
        require_same_files(current=current, qualified=qualified, paths=["app.py"])
    with pytest.raises(FileNotFoundError):
        require_same_files(current=current, qualified=qualified, paths=["new.py"])


@pytest.mark.parametrize(
    "text, valid",
    [
        ("115 passed in 4.1s\n", True),
        ("=== 115 passed, 2 warnings in 4.1s ===\n", True),
        ("114 passed in 4.1s\n", False),
        ("1 failed, 115 passed in 4.1s\n", False),
        ("115 passed, 1 error in 4.1s\n", False),
        ("collecting tests ...\n", False),
    ],
)
def test_pytest_completion_gate(tmp_path, text, valid):
    log = tmp_path / "tests.log"
    log.write_text(text)
    if valid:
        require_pytest_pass(log, minimum=115)
    else:
        with pytest.raises(ValueError):
            require_pytest_pass(log, minimum=115)


def test_runtime_inventory_excludes_test_and_environment_files(tmp_path):
    for name in (
        "server/app.py",
        "server/config.yaml",
        "server/tests/test_app.py",
        "server/.venv/pkg.py",
        "server/data.png",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    assert runtime_files(tmp_path, ["server"]) == [
        "server/app.py",
        "server/config.yaml",
    ]


def test_qualification_requires_actual_checkpoint_all_games_and_both_test_suites(
    tmp_path, monkeypatch
):
    from tools.check_visual_image_tools_mix import (
        GYM_V_TRAIN,
        GYM_V_VALIDATION,
        VISGYM_TRAIN,
    )

    roots = {
        key: tmp_path / key
        for key in (
            "project_root",
            "image_source",
            "image_run",
            "component_source",
            "component_run",
            "games_venv_root",
            "overlay",
        )
    }
    for root in roots.values():
        root.mkdir()

    def write(root, name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    (roots["project_root"] / "uv.lock").write_bytes(b"lock")
    write(
        roots["overlay"],
        "READY.json",
        {"source_lock_sha256": hashlib.sha256(b"lock").hexdigest()},
    )
    write(
        roots["image_run"],
        "checkpoints/latest_checkpoint_status.json",
        {"last_checkpoint_step": 20, "last_successful_ckpt_save_completion": 1},
    )
    write(
        roots["image_run"],
        "tokenizer/provenance.json",
        {"model": "/step_120/hf", "fix_mistral_regex": True},
    )
    certificate = roots["games_venv_root"] / ".visual-games-suite-prefetch-complete"
    certificate.write_bytes(b"qualified")
    report = {
        "results": [
            {"env_id": game, "passed": True}
            for game in sorted(GYM_V_TRAIN | GYM_V_VALIDATION | VISGYM_TRAIN)
        ],
        "certificate_unchanged": True,
        "certificate_sha256": hashlib.sha256(b"qualified").hexdigest(),
    }
    write(roots["component_run"], "component-summary.json", report)
    (roots["component_run"] / "agent-guard-tests.log").write_text("115 passed in 1s\n")
    (roots["component_run"] / "rl-metric-test.log").write_text("1 passed in 1s\n")
    # Source comparison is exercised separately with actual files above.
    compared_paths = []

    def compare(**kwargs):
        compared_paths.extend(kwargs["paths"])
        return "source-hash"

    monkeypatch.setattr(
        "tools.check_visual_image_tools_launch.require_same_files", compare
    )
    monkeypatch.setattr(
        "tools.check_visual_image_tools_launch.require_callback_sources",
        lambda root: None,
    )
    args = {**roots, "model_checkpoint": "/step_120/hf"}
    assert qualify(**args)["mixed_training_verified"] is False
    assert "tools/visual_image_tools_components_hsg.sh" not in compared_paths
    assert "tools/check_visual_image_tools_components.py" in compared_paths
    assert "nemo_rl/experience/rollouts.py" in compared_paths
    with pytest.raises(ValueError, match="model/tokenizer"):
        qualify(**{**args, "model_checkpoint": "/wrong/model"})
    report["results"][0]["passed"] = False
    write(roots["component_run"], "component-summary.json", report)
    with pytest.raises(ValueError, match="42 game"):
        qualify(**args)
    report["results"][0]["passed"] = True
    write(roots["component_run"], "component-summary.json", report)
    (roots["component_run"] / "rl-metric-test.log").write_text("1 failed in 1s\n")
    with pytest.raises(ValueError, match="tests did not complete"):
        qualify(**args)
