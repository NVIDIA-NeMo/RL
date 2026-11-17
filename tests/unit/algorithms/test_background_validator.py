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

import threading
import time

import pytest

from nemo_rl.algorithms.async_utils import BackgroundValidator


def _gated_validator():
    """A validator whose validations block until their step's gate is set."""
    gates: dict[int, threading.Event] = {}

    def validate_fn(step: int):
        gates[step].wait(timeout=10)
        return {"accuracy": step / 10}, {"total_validation_time": 1.0}

    validator = BackgroundValidator(validate_fn)

    def launch(step: int) -> threading.Event:
        gates[step] = threading.Event()
        validator.launch(step)
        return gates[step]

    return validator, launch


def _wait_until(condition, what: str) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.01)


def test_collect_finished_returns_only_done_validations_in_order():
    validator, launch = _gated_validator()
    gate_2 = launch(2)
    gate_4 = launch(4)
    assert validator.pending_steps == [2, 4]
    assert validator.collect_finished() == []

    gate_4.set()
    finished = []
    _wait_until(
        lambda: finished.extend(validator.collect_finished()) or finished,
        "validation for step 4 never finished",
    )
    (result,) = finished
    assert result.val_step == 4
    assert result.metrics == {"accuracy": 0.4}
    assert result.timings == {"total_validation_time": 1.0}
    assert validator.pending_steps == [2]

    gate_2.set()
    (result,) = validator.wait_all()
    assert result.val_step == 2
    assert validator.pending_steps == []


def test_duplicate_launch_is_rejected():
    validator, launch = _gated_validator()
    gate = launch(1)
    with pytest.raises(ValueError, match="already running"):
        validator.launch(1)
    gate.set()
    validator.wait_all()


def test_validation_error_is_raised_on_collection():
    def validate_fn(step: int):
        raise RuntimeError(f"validation {step} failed")

    validator = BackgroundValidator(validate_fn)
    validator.launch(1)
    with pytest.raises(RuntimeError, match="validation 1 failed"):
        validator.wait_all()
    assert validator.pending_steps == []


def test_validation_runs_on_a_daemon_thread():
    seen: dict[str, bool] = {}

    def validate_fn(step: int):
        thread = threading.current_thread()
        seen["daemon"] = thread.daemon
        seen["main"] = thread is threading.main_thread()
        return {}, {}

    validator = BackgroundValidator(validate_fn)
    validator.launch(1)
    validator.wait_all()
    assert seen == {"daemon": True, "main": False}


def test_is_running_ignores_finished_but_uncollected_validations():
    validator, launch = _gated_validator()
    assert not validator.is_running()
    gate = launch(2)
    assert validator.is_running()

    gate.set()
    _wait_until(
        lambda: not validator.is_running(), "validation for step 2 never finished"
    )
    assert validator.pending_steps == [2]
