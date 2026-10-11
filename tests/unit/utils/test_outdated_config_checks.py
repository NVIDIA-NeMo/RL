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
import ast
from pathlib import Path

import pytest

from nemo_rl.utils.outdated_config_checks import check_outdated_config

REPO_ROOT = Path(__file__).resolve().parents[3]

# The entrypoints under test are every file under ENTRYPOINTS_DIRS that imports
# MasterConfig, plus ENTRYPOINTS_EXTRA_FILES.
ENTRYPOINTS_DIRS = ["examples", "research"]
ENTRYPOINTS_EXTRA_FILES = [REPO_ROOT / "tools" / "refit_verifier.py"]


def _entrypoints() -> list[Path]:
    """Every module under ENTRYPOINTS_DIRS that imports MasterConfig, plus the extras."""
    found = []
    for directory in ENTRYPOINTS_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if "tests" in path.parts:
                continue
            # Importing it, not defining it
            if any(
                alias.name == "MasterConfig"
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.ImportFrom)
                for alias in node.names
            ):
                found.append(path)
    assert found, "No entrypoints found"
    return found + ENTRYPOINTS_EXTRA_FILES


def _call_lines(tree: ast.AST, name: str) -> list[int]:
    """Lines calling name, as name(...) or name.anything(...)."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            func = func.value
        if isinstance(func, ast.Name) and func.id == name:
            lines.append(node.lineno)
    return sorted(lines)


# ============================================================================
# Entrypoint wiring
# ============================================================================


@pytest.mark.parametrize(
    "entrypoint", _entrypoints(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_every_entrypoint_checks_outdated_config(entrypoint):
    """An entrypoint that resolves a user config must reject outdated config first.

    Without this, a stale config gets as far as worker startup before failing.
    """
    if entrypoint == REPO_ROOT / "examples" / "run_eval.py":
        pytest.skip("eval configs have a different structure")

    tree = ast.parse(entrypoint.read_text())
    rel = entrypoint.relative_to(REPO_ROOT)

    check_lines = _call_lines(tree, "check_outdated_config")
    assert check_lines, (
        f"{rel} resolves a config but never calls check_outdated_config(). Add the "
        "call just after OmegaConf.to_container, before the config is used."
    )
    # The rest is about ordering against the MasterConfig build, which this one has not.
    if entrypoint == REPO_ROOT / "tools" / "refit_verifier.py":
        return

    build_lines = _call_lines(tree, "MasterConfig")
    assert build_lines, (
        f"{rel} mentions MasterConfig but this test sees no call that builds one, so "
        "it cannot check the ordering. Build it by calling MasterConfig directly, or "
        "teach _call_lines the new form."
    )
    assert check_lines[0] < build_lines[0], (
        f"{rel} calls check_outdated_config() after building the MasterConfig. "
        "Validation rejects a missing required key first, so the migration message "
        "would never be reached; move the call before the build."
    )


# ============================================================================
# reject_outdated_automodel_block
# ============================================================================


@pytest.mark.parametrize("section", ["policy", "value", "teacher"])
def test_outdated_dtensor_cfg_key_is_rejected(section):
    with pytest.raises(
        ValueError,
        match=rf"{section}\.dtensor_cfg has been renamed to {section}\.automodel_cfg\.",
    ):
        check_outdated_config({section: {"dtensor_cfg": {"enabled": True}}})


def test_each_teacher_dtensor_cfg_key_is_rejected():
    with pytest.raises(
        ValueError,
        match=r"teachers\.1\.dtensor_cfg has been renamed to teachers\.1\.automodel_cfg\.",
    ):
        check_outdated_config(
            {"teachers": [{"automodel_cfg": {}}, {"dtensor_cfg": {"enabled": True}}]}
        )


def test_renamed_key_passes():
    check_outdated_config({"policy": {"automodel_cfg": {"enabled": True}}})


def test_megatron_block_keeps_its_dtensor_cfg():
    """A Megatron run's block is inert, so the old name there is not its problem."""
    check_outdated_config(
        {
            "policy": {
                "megatron_cfg": {"enabled": True},
                "dtensor_cfg": {"enabled": False},
            }
        }
    )


def test_megatron_disabled_still_checks():
    with pytest.raises(
        ValueError,
        match=r"policy\.dtensor_cfg has been renamed to policy\.automodel_cfg\.",
    ):
        check_outdated_config(
            {
                "policy": {
                    "megatron_cfg": {"enabled": False},
                    "dtensor_cfg": {"enabled": True},
                }
            }
        )


def test_reward_model_env_is_checked():
    with pytest.raises(
        ValueError,
        match=r"env\.reward_model\.dtensor_cfg has been renamed to "
        r"env\.reward_model\.automodel_cfg\.",
    ):
        check_outdated_config(
            {"env": {"reward_model": {"dtensor_cfg": {"enabled": True}}}}
        )


@pytest.mark.parametrize("value", [True, False])
def test_outdated_v2_key_under_the_new_name_is_rejected(value):
    """Following the rename message must not carry _v2 across."""
    with pytest.raises(
        ValueError, match=r"policy\.automodel_cfg\._v2 has been removed"
    ):
        check_outdated_config({"policy": {"automodel_cfg": {"_v2": value}}})


@pytest.mark.parametrize("section", ["value", "teacher"])
def test_each_top_level_section_is_checked_for_v2(section):
    with pytest.raises(
        ValueError, match=rf"{section}\.automodel_cfg\._v2 has been removed"
    ):
        check_outdated_config({section: {"automodel_cfg": {"_v2": True}}})


def test_each_teacher_is_checked_for_v2():
    with pytest.raises(
        ValueError, match=r"teachers\.1\.automodel_cfg\._v2 has been removed"
    ):
        check_outdated_config(
            {"teachers": [{"automodel_cfg": {}}, {"automodel_cfg": {"_v2": False}}]}
        )


def test_megatron_block_keeps_its_v2():
    """A Megatron run reads neither key, so a stale _v2 there is not its problem."""
    check_outdated_config(
        {
            "policy": {
                "megatron_cfg": {"enabled": True},
                "automodel_cfg": {"enabled": False, "_v2": False},
            }
        }
    )


def test_the_rename_is_reported_before_the_v2_key():
    """A config still on the old block name is told to rename it, not about _v2."""
    with pytest.raises(ValueError, match="has been renamed"):
        check_outdated_config(
            {"policy": {"dtensor_cfg": {"_v2": False}, "automodel_cfg": {"_v2": False}}}
        )


# ============================================================================
# reject_outdated_dataset_config
# ============================================================================


def test_flat_dataset_config_is_rejected():
    with pytest.raises(ValueError, match="data has no train section"):
        check_outdated_config({"data": {"dataset_name": "AIME2024"}})


def test_split_dataset_config_passes():
    check_outdated_config({"data": {"train": {}, "validation": {}}})


def test_absent_data_section_passes():
    check_outdated_config({"policy": {}})


# ============================================================================
# reject_outdated_metric_name_format
# ============================================================================


@pytest.mark.parametrize("metric_name", ["val:accuracy", "train:loss", None])
def test_current_metric_name_format_passes(metric_name):
    check_outdated_config({"checkpointing": {"metric_name": metric_name}})


@pytest.mark.parametrize("metric_name", ["val_loss", "accuracy", "reward"])
def test_bare_metric_name_is_rejected(metric_name):
    with pytest.raises(ValueError, match=r"must start with 'train:' or 'val:'"):
        check_outdated_config({"checkpointing": {"metric_name": metric_name}})


def test_absent_checkpointing_passes():
    check_outdated_config({"policy": {}})
