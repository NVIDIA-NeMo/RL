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

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "nixl_recovery_bisect",
    Path(__file__).resolve().parents[2] / "tools/nixl_recovery_bisect.py",
)
bisect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bisect)


@pytest.mark.parametrize(
    ("status", "log", "expected"),
    [
        (
            0,
            "=== Phase 2: restore\nNative TQ checkpoint restored and validated\nPASS",
            "good",
        ),
        (0, "[SKIP] no RDMA", "inconclusive"),
        (0, "", "inconclusive"),
        (1, "ModuleNotFoundError", "inconclusive"),
        (1, "=== Phase 1: save\nNIXL_ERR_NOT_FOUND", "inconclusive"),
        (
            1,
            "=== Phase 1: save\nDataPlaneTransferFailed NIXL_ERR_REMOTE_DISCONNECT",
            "bad",
        ),
        (
            1,
            "=== Phase 1: save\n=== Phase 2: restore\nDataPlaneTransferFailed NIXL_ERR_REMOTE_DISCONNECT",
            "inconclusive",
        ),
        (124, "timed out", "inconclusive"),
    ],
)
def test_classify(status: int, log: str, expected: str) -> None:
    assert bisect.classify(status, log) == expected


def test_find_and_repeat_adjacent_boundary() -> None:
    calls = []
    commits = [str(i) for i in range(18)]

    def measure(revision: str) -> str:
        calls.append(revision)
        return "bad" if int(revision) >= 11 else "good"

    assert bisect.locate_boundary(commits, measure) == ("10", "11")
    assert calls[-2:] == ["10", "11"]
    assert len(calls) < len(commits)


def test_unrelated_failure_is_not_an_introducer() -> None:
    with pytest.raises(RuntimeError, match="Inconclusive revision"):
        bisect.locate_boundary(["a", "b", "c"], lambda revision: "inconclusive")


def test_unstable_boundary_is_not_an_introducer() -> None:
    with pytest.raises(RuntimeError, match="Boundary did not repeat"):
        bisect.locate_boundary(["a", "b"], lambda revision: "bad")


@pytest.mark.parametrize("status", [124, 137, -15])
def test_terminated_transfer_is_inconclusive(status: int) -> None:
    assert (
        bisect.classify(
            status,
            "=== Phase 1: save\nDataPlaneTransferFailed NIXL_ERR_REMOTE_DISCONNECT",
        )
        == "inconclusive"
    )


def test_time_budget_blocks_new_trials_but_allows_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = bisect.Experiment(tmp_path, tmp_path)
    experiment.deadline = bisect.time.monotonic() - 1
    with pytest.raises(RuntimeError, match="Time budget exhausted"):
        experiment.run(
            ["not-executed"], cwd=tmp_path, log=tmp_path / "trial.log", seconds=10
        )
    result = type("Finished", (), {"wait": lambda self, timeout: 0})()
    monkeypatch.setattr(bisect.subprocess, "Popen", lambda *args, **kwargs: result)
    assert (
        experiment.run(
            ["ray", "stop", "--force"],
            cwd=tmp_path,
            log=tmp_path / "cleanup.log",
            seconds=45,
        )
        == 0
    )


def test_mixed_repetitions_do_not_choose_a_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = bisect.Experiment(tmp_path, tmp_path)
    outcomes = iter(["good", "bad"])
    monkeypatch.setattr(experiment, "trial", lambda revision: next(outcomes))
    with pytest.raises(RuntimeError, match="Unstable or unrelated"):
        experiment.measure("example")


def test_endpoints_must_reproduce_before_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = bisect.Experiment(tmp_path, tmp_path)
    monkeypatch.setattr(
        bisect,
        "git",
        lambda repo, *args: {"merge-base": "base", "rev-parse": "false"}.get(
            args[0], ""
        ),
    )
    measured = []

    def trial(revision: str) -> str:
        measured.append(revision)
        return "bad"

    monkeypatch.setattr(experiment, "trial", trial)
    with pytest.raises(RuntimeError, match="Endpoints did not reproduce"):
        experiment.execute()
    assert measured == [bisect.GOOD, bisect.BAD, bisect.BAD, bisect.GOOD]


def test_shallow_history_never_runs_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = bisect.Experiment(tmp_path, tmp_path)
    monkeypatch.setattr(bisect, "git", lambda repo, *args: "true")
    monkeypatch.setattr(
        experiment,
        "trial",
        lambda revision: pytest.fail("No trial may run with shallow history"),
    )
    with pytest.raises(RuntimeError, match="Complete Git history"):
        experiment.execute()
