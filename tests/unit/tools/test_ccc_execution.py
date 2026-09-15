# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the patched, pinned CCC function with controlled sandbox replies."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
GYM = ROOT / "3rdparty/Gym-workspace/Gym"
BASE = "749432dc5de23b8eeb3d044c80350a7c0ae9a03f"  # pragma: allowlist secret
RELATIVE = "resources_servers/competitive_coding_challenges/ccc_eval.py"


@pytest.fixture(scope="module")
def evaluator(tmp_path_factory):
    source = subprocess.run(
        ["git", "-C", str(GYM), "show", f"{BASE}:{RELATIVE}"],
        capture_output=True,
        check=False,
    )
    if source.returncode:
        pytest.skip("Initialize the pinned Gym submodule to test its execution overlay")
    stage = tmp_path_factory.mktemp("ccc-execution")
    path = stage / RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(source.stdout)
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(stage), "-i",
         str(ROOT / "tools/super_rl/patches/gym_ccc_execution.patch")],
        check=True,
        capture_output=True,
    )
    spec = importlib.util.spec_from_file_location("ccc_execution_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def execute(evaluator, tmp_path, compile_reply, run_reply):
    with (
        patch.object(evaluator, "_get_thread_test_sandbox", return_value=None),
        patch.object(evaluator, "_exec_sync", side_effect=[compile_reply, run_reply]) as calls,
    ):
        result = evaluator.run_test_case(
            {"shared_dir": str(tmp_path), "problem_id": "sum",
             "generated_code": "int main() {}", "test_input": "", "test_output": ""},
            worker_id=0,
        )
    return result, calls.call_count


@pytest.mark.parametrize("stderr", ["", "warning: unused variable"])
def test_successful_compilation_accepts_warnings(evaluator, tmp_path, stderr):
    result, calls = execute(
        evaluator, tmp_path,
        {"process_status": "completed", "stdout": "", "stderr": stderr},
        {"process_status": "completed", "stdout": "1", "stderr": ""},
    )
    assert result["compile_success"] is True
    assert result["compile_stderr"] == stderr
    assert result["score"] == 1.0
    assert calls == 2


@pytest.mark.parametrize("status", ["error", "failed", "timeout", None])
def test_non_successful_compile_never_runs_grader(evaluator, tmp_path, status):
    result, calls = execute(
        evaluator, tmp_path,
        {"process_status": status, "stdout": "", "stderr": ""},
        {"process_status": "completed", "stdout": "1", "stderr": ""},
    )
    assert result["compile_success"] is False
    assert result["compile_status"] == status
    assert result["score"] == 0.0
    assert calls == 1


@pytest.mark.parametrize("status", ["error", "failed", "timeout", None])
def test_failed_grader_stdout_is_not_a_score(evaluator, tmp_path, status):
    result, calls = execute(
        evaluator, tmp_path,
        {"process_status": "completed", "stdout": "", "stderr": ""},
        {"process_status": status, "stdout": "1", "stderr": "diagnostic"},
    )
    assert result["score"] == 0.0
    assert result["run_status"] == status
    assert result["run_stdout"] == "1"
    assert result["run_stderr"] == "diagnostic"
    assert calls == 2


def test_valid_wrong_answer_and_fractional_scores_are_preserved(evaluator, tmp_path):
    for score in ("0", "0.5"):
        result, _ = execute(
            evaluator, tmp_path,
            {"process_status": "completed", "stdout": "", "stderr": ""},
            {"process_status": "completed", "stdout": score, "stderr": ""},
        )
        assert result["score"] == float(score)
