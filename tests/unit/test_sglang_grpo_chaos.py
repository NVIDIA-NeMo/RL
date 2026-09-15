# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU checks for the real-GRPO chaos harness's acceptance conditions."""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.functional._sglang_grpo_chaos import (
    Engine,
    KillReceipt,
    ReplacementReceipt,
    completed_steps,
    engine_endpoints,
    training_command,
    validate_outcome,
)


def survival_case() -> dict:
    return {
        "expect": "survival",
        "returncode": 0,
        "log_after_kill": "Restarting SGLang engine 0 (attempt 1/1)",
        "completed": [1, 2, 3, 4, 5],
        "max_steps": 5,
        "kill": KillReceipt(Engine(10, 1.0, "http://host:3001"), 2, 4, 1, 10.0),
        "replacement": ReplacementReceipt(
            Engine(20, 2.0, "http://host:3002"), 3, 3, 2, 20.0
        ),
        "metrics": {
            "train/grad_norm": {"3": 0.0, "4": 0.5, "5": 0.0},
            "train/global_valid_toks": {"4": 32.0},
        },
    }


def test_real_training_results_not_step_banners() -> None:
    log = (
        "========================= Step 1/12 =========================\n"
        "▶ Generating responses for batch of size 16...\n"
    )
    assert completed_steps(log) == []
    log += "📊 Training Results:\n  • Loss: 0.1\n"
    assert completed_steps(log) == [1]
    log += "========================= Step 2/12 =========================\n"
    assert completed_steps(log) == [1]


def test_advertised_actor_endpoints() -> None:
    log = (
        "\x1b[36m(SGLangGenerationWorker pid=42)\x1b[0m INFO: "
        "Launch HttpServerEngineAdapter at: 10.1.2.3:3001\n"
        "(OtherWorker pid=99) Launch HttpServerEngineAdapter at: 10.1.2.3:9999\n"
    )
    assert engine_endpoints(log) == {42: "http://10.1.2.3:3001"}


def test_survival_requires_only_one_nonzero_gradient_step() -> None:
    validate_outcome(**survival_case())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("returncode", 1, "Training failed"),
        ("replacement", None, "No replacement"),
        ("completed", [1, 2, 3, 4], "did not finish"),
        ("log_after_kill", "", "No recovery"),
        ("metrics", {}, "nonzero gradients"),
        (
            "metrics",
            {
                "train/grad_norm": {"4": float("nan")},
                "train/global_valid_toks": {"4": 10},
            },
            "nonzero gradients",
        ),
    ],
)
def test_survival_rejects_false_positives(
    field: str, value: object, message: str
) -> None:
    case = survival_case()
    case[field] = value
    with pytest.raises(AssertionError, match=message):
        validate_outcome(**case)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("weight_version", 2, "Stale replacement"),
        ("running_requests", 1, "never served"),
        ("completed_step", 4, "Fewer than two"),
        ("engine", Engine(10, 1.0, "http://host:3001"), "mistaken for replacement"),
    ],
)
def test_replacement_must_rejoin_training(
    field: str, value: object, message: str
) -> None:
    case = survival_case()
    case["replacement"] = replace(case["replacement"], **{field: value})
    with pytest.raises(AssertionError, match=message):
        validate_outcome(**case)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("running_requests", 1, "not observably serving"),
        ("completed_step", 0, "first completed"),
        ("weight_version", 0, "no trainer-transferred"),
    ],
)
def test_fault_must_land_on_busy_trained_engine(
    field: str, value: int, message: str
) -> None:
    case = survival_case()
    case["kill"] = replace(case["kill"], **{field: value})
    with pytest.raises(AssertionError, match=message):
        validate_outcome(**case)


@pytest.mark.parametrize(
    ("returncode", "log", "passes"),
    [
        (
            1,
            "SGLang engines [0] exhausted rollout_max_restart_attempts=0; aborting refit.",
            True,
        ),
        (
            0,
            "SGLang engines [0] exhausted rollout_max_restart_attempts=0; aborting refit.",
            False,
        ),
        (1, "CUDA out of memory", False),
        (1, "Timeout waiting for training", False),
        (
            1,
            "SGLang engines [0] exhausted rollout_max_restart_attempts=1; aborting refit.",
            False,
        ),
    ],
)
def test_bounded_failure_is_attributable(
    returncode: int, log: str, passes: bool
) -> None:
    case = survival_case()
    case.update(expect="bounded_failure", returncode=returncode, log_after_kill=log)
    if passes:
        validate_outcome(**case)
    else:
        with pytest.raises(AssertionError):
            validate_outcome(**case)


@pytest.mark.parametrize(
    ("expect", "budget"), [("survival", 1), ("bounded_failure", 0)]
)
def test_launches_real_grpo_with_canonical_config(expect: str, budget: int) -> None:
    command = training_command(
        Path("/source"), Path("/artifacts"), expect=expect, steps=12
    )
    assert "/source/examples/run_grpo.py" in command
    assert "/source/examples/configs/grpo_math_1B_sglang.yaml" in command
    prefix = "policy.generation.sglang_cfg.sglang_fault_tolerance_config"
    assert f"{prefix}.use_fault_tolerance=true" in command
    assert f"{prefix}.rollout_max_restart_attempts={budget}" in command
    assert "grpo.max_num_steps=12" in command
    assert "logger.tensorboard_enabled=true" in command
    assert "policy.generation.use_async_rollouts=false" in command
    assert not any("test_fault_tolerance_real.py" in arg for arg in command)
