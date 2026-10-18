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


def _gated_validator(current_step: list[int]):
    """A validator whose validations block until their step's gate is set."""
    gates: dict[int, threading.Event] = {}

    def validate_fn(step: int):
        gates[step].wait(timeout=10)
        return {"accuracy": step / 10}, {"total_validation_time": 1.0}

    validator = BackgroundValidator(validate_fn, lambda: current_step[0])

    def launch(step: int) -> threading.Event:
        gates[step] = threading.Event()
        validator.launch(step)
        return gates[step]

    return validator, launch


def test_collect_finished_returns_only_done_validations_in_order():
    current_step = [2]
    validator, launch = _gated_validator(current_step)
    gate_2 = launch(2)
    with pytest.warns(UserWarning, match="still running"):
        gate_4 = launch(4)
    assert validator.pending_steps == [2, 4]
    assert validator.collect_finished() == []

    current_step[0] = 5
    gate_4.set()
    deadline = time.monotonic() + 10
    while not (finished := validator.collect_finished()):
        assert time.monotonic() < deadline, "validation for step 4 never finished"
        time.sleep(0.01)
    (result,) = finished
    assert (result.val_step, result.end_step) == (4, 5)
    assert result.metrics == {"accuracy": 0.4}
    assert result.timings == {"total_validation_time": 1.0}
    assert validator.pending_steps == [2]

    current_step[0] = 6
    gate_2.set()
    (result,) = validator.wait_all()
    assert (result.val_step, result.end_step) == (2, 6)
    assert validator.pending_steps == []


def test_checkpoint_pending_is_reported_once():
    validator, launch = _gated_validator([3])
    launch(2).set()
    assert validator.is_pending(2)
    validator.mark_checkpoint_pending(2)
    (result,) = validator.wait_all()
    assert result.checkpoint_pending
    assert not validator.is_pending(2)

    launch(3).set()
    (result,) = validator.wait_all()
    assert not result.checkpoint_pending


def test_mark_checkpoint_pending_requires_running_validation():
    validator, _ = _gated_validator([1])
    with pytest.raises(ValueError, match="No validation is running"):
        validator.mark_checkpoint_pending(1)


def test_duplicate_launch_is_rejected():
    validator, launch = _gated_validator([1])
    gate = launch(1)
    with pytest.raises(ValueError, match="already running"):
        validator.launch(1)
    gate.set()
    validator.wait_all()


def test_validation_error_is_raised_on_collection():
    def validate_fn(step: int):
        raise RuntimeError(f"validation {step} failed")

    validator = BackgroundValidator(validate_fn, lambda: 1)
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

    validator = BackgroundValidator(validate_fn, lambda: 1)
    validator.launch(1)
    validator.wait_all()
    assert seen == {"daemon": True, "main": False}
