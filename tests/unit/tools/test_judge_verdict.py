# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

G_PATH = (
    Path(__file__).resolve().parents[3] / "tools/super_rl/gym_overlays/judge_verdict.py"
)
spec = importlib.util.spec_from_file_location("judge_verdict", G_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "text,expected",
    [("[[A=B]]", True), ("[[A!=B]]", False), ("[[A!=B]] again [[A!=B]]", False)],
)
def test_unambiguous_verdict(text: str, expected: bool) -> None:
    assert (
        module.first_verdict(text, equal_label="[[A=B]]", not_equal_label="[[A!=B]]")
        is expected
    )


@pytest.mark.parametrize(
    "text",
    [
        "[[A!=B]] then [[A=B]]",
        "Need final with [[A=B]] or [[A!=B]]. Let me continue reasoning.",
    ],
)
def test_conflicting_verdicts_are_not_rewards(text: str) -> None:
    with pytest.raises(ValueError, match="conflicting verdicts"):
        module.first_verdict(text, equal_label="[[A=B]]", not_equal_label="[[A!=B]]")


@pytest.mark.parametrize("text", ["", "I cannot decide", "[[broken]]"])
def test_missing_verdict_is_not_a_negative(text: str) -> None:
    with pytest.raises(ValueError, match="no valid verdict"):
        module.first_verdict(text, equal_label="[[A=B]]", not_equal_label="[[A!=B]]")


def test_response_failure_records_cap_without_answer_text():
    response = SimpleNamespace(
        output_text="PRIVATE_JUDGE_TEXT",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=SimpleNamespace(output_tokens=8192),
    )
    with pytest.raises(ValueError) as caught:
        module.response_verdict(
            response, equal_label="[[A=B]]", not_equal_label="[[A!=B]]"
        )
    message = str(caught.value)
    assert "not complete" in message
    assert "max_output_tokens" in message
    assert "8192" in message
    assert "PRIVATE_JUDGE_TEXT" not in message


def test_completed_response_verdict_does_not_need_usage() -> None:
    response = SimpleNamespace(output_text="[[A!=B]]", status="completed")
    assert not module.response_verdict(
        response, equal_label="[[A=B]]", not_equal_label="[[A!=B]]"
    )


@pytest.mark.parametrize(
    "status", [None, "incomplete", "failed", "cancelled", "in_progress", "queued"]
)
def test_unfinished_response_with_label_is_not_a_reward(status: str | None) -> None:
    response = SimpleNamespace(output_text="[[A=B]]", status=status)
    with pytest.raises(ValueError, match="not complete"):
        module.response_verdict(
            response, equal_label="[[A=B]]", not_equal_label="[[A!=B]]"
        )


def test_incomplete_details_override_completed_status() -> None:
    response = SimpleNamespace(
        output_text="[[A=B]]",
        status="completed",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    with pytest.raises(ValueError, match="not complete"):
        module.response_verdict(
            response, equal_label="[[A=B]]", not_equal_label="[[A!=B]]"
        )


def test_completed_conflicting_response_preserves_private_text() -> None:
    response = SimpleNamespace(
        output_text="PRIVATE_JUDGE_TEXT [[A=B]] or [[A!=B]]", status="completed"
    )
    with pytest.raises(ValueError, match="conflicting verdicts") as caught:
        module.response_verdict(
            response, equal_label="[[A=B]]", not_equal_label="[[A!=B]]"
        )
    assert "PRIVATE_JUDGE_TEXT" not in str(caught.value)
