# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preflight gates must not turn partial or failed GPU checks into training."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from nano35.start_after_preflight import bootstrap, preflight_state


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "COMPLETING"])
def test_active_preflight_never_reads_a_success_report(tmp_path, state):
    with patch(
        "nano35.start_after_preflight.subprocess.check_output",
        return_value=f"123|{state}\n",
    ):
        assert preflight_state("123", tmp_path / "missing.json") == "wait"


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMEOUT", ""])
def test_unsuccessful_preflight_cannot_start_training(tmp_path, state):
    with patch(
        "nano35.start_after_preflight.subprocess.check_output",
        side_effect=["", f"123|{state}\n" if state else ""],
    ):
        with pytest.raises(RuntimeError):
            preflight_state("123", tmp_path / "missing.json")


def test_completed_job_with_partial_gpu_checks_is_rejected(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "passed", "checks": {}}))
    with patch(
        "nano35.start_after_preflight.subprocess.check_output",
        side_effect=["", "123|COMPLETED\n"],
    ):
        with pytest.raises(RuntimeError, match="all required checks"):
            preflight_state("123", report)


@pytest.mark.parametrize("already_submitted", [False, True])
@pytest.mark.parametrize("once", [False, True])
def test_fresh_submission_happens_once_before_continuation(
    tmp_path, already_submitted, once
):
    run = tmp_path / "fresh-run"
    run.mkdir()
    if already_submitted:
        (run / "job-id.txt").write_text("456\n")
    with (
        patch("nano35.start_after_preflight.preflight_state", return_value="ready"),
        patch("nano35.start_after_preflight.subprocess.run") as command,
        patch.dict("os.environ", {}, clear=True),
    ):
        bootstrap("123", tmp_path / "report.json", run, once=once)
    calls = [call.args[0] for call in command.call_args_list]
    if not already_submitted:
        assert calls[0][0] == "bash"
        assert calls[0][-2:] == ["fresh", str(run)]
    assert len(calls) == (1 if already_submitted else 2)
    assert Path(calls[-1][1]).name == "continue_training.py"
    assert ("--once" in calls[-1]) is once


def test_timer_check_returns_without_submitting_while_preflight_is_active(tmp_path):
    with (
        patch("nano35.start_after_preflight.preflight_state", return_value="wait"),
        patch("nano35.start_after_preflight.subprocess.run") as command,
        patch("nano35.start_after_preflight.time.sleep") as sleep,
    ):
        bootstrap("123", tmp_path / "report.json", tmp_path / "run", once=True)
    command.assert_not_called()
    sleep.assert_not_called()
