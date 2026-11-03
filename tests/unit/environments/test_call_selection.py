# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy discovery and configuration work without optional Gym imports."""

from pathlib import Path

import pytest

from nemo_rl.algorithms.single_controller_utils.config import TokenCaptureConfig
from nemo_rl.environments.call_selection import (
    SelectionPolicy,
    resolve_selection_policy,
    select_all_captured,
    select_captured_calls,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("returned_history", select_captured_calls),
        ("all_captured", select_all_captured),
    ],
)
def test_builtin_selection_policy(name: str, expected: SelectionPolicy) -> None:
    assert resolve_selection_policy(name) is expected


def test_external_selection_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "custom_selection.py").write_text(
        "def choose(records, result):\n"
        "    return records[:1], result['terminal_response_id']\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    name = "custom_selection:choose"
    config = TokenCaptureConfig(forest_training=True, selection_policy=name)
    select = resolve_selection_policy(config.selection_policy)
    assert select([], {"terminal_response_id": "served-id"}) == ([], "served-id")
    with pytest.raises(ValueError, match="requires forest_training"):
        TokenCaptureConfig(selection_policy=name)


@pytest.mark.parametrize(
    "name,error",
    [
        ("guess", "Unknown selection_policy"),
        ("", "Unknown selection_policy"),
        (":choose", "Unknown selection_policy"),
        ("json:", "Unknown selection_policy"),
        ("json:loads:extra", "Unknown selection_policy"),
        ("missing_selection_package:choose", "Cannot load selection_policy"),
        ("json:missing_selector", "Cannot load selection_policy"),
        ("json:__name__", "not callable"),
    ],
)
def test_bad_selection_policy_fails_in_config(name: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        TokenCaptureConfig(forest_training=True, selection_policy=name)
