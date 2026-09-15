# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral regression tests for the pinned Gym rollout-budget overlay."""

import importlib.util
from pathlib import Path
import sys

import pytest

G_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools/super_rl/gym_overlays/rollout_budget.py"
)
spec = importlib.util.spec_from_file_location("rollout_budget", G_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
OutputTokenBudget = module.OutputTokenBudget


def test_cumulative_calls() -> None:
    budget = OutputTokenBudget(100)
    assert budget.limit(80) == 80
    budget.consume(70)
    assert budget.limit(80) == 30
    budget.consume(30)
    assert budget.exhausted
    assert (budget.used, budget.calls) == (100, 2)
    with pytest.raises(RuntimeError, match="exhausting"):
        budget.limit(80)


@pytest.mark.parametrize("invalid", [None, -1, True, 1.5])
def test_usage_fails_closed(invalid: object) -> None:
    budget = OutputTokenBudget(100)
    budget.limit(50)
    with pytest.raises(RuntimeError, match="usage"):
        budget.consume(invalid)


def test_returned_tokens_cannot_exceed_per_call_allowance() -> None:
    budget = OutputTokenBudget(100)
    budget.limit(10)
    with pytest.raises(RuntimeError, match="more output"):
        budget.consume(11)
    assert budget.used == 0


def test_absent_budget_preserves_legacy() -> None:
    budget = OutputTokenBudget(None)
    assert budget.limit(None) is None
    budget.consume(None)
    assert not budget.exhausted
