# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = (
    Path(__file__).resolve().parents[3] / "tools/super_rl/gym_overlays/judge_retry.py"
)
spec = importlib.util.spec_from_file_location("judge_retry", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class JudgeError(Exception):
    pass


class Judge:
    def __init__(self, outcomes, attempts=4):
        self.config = SimpleNamespace(name="test_judge", judge_max_attempts=attempts)
        self.outcomes = iter(outcomes)
        self.calls = []

    @module.retry_judge_errors(JudgeError)
    async def decision(self, question, candidate, *, budget):
        self.calls.append((question, candidate, budget))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    async def sleep(_delay):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", sleep)


def test_retry_reuses_arguments_and_stops_at_first_valid_negative():
    judge = Judge([JudgeError("missing verdict"), False, True])
    assert not asyncio.run(judge.decision("question", "original answer", budget=8192))
    assert judge.calls == [("question", "original answer", 8192)] * 2


def test_exhaustion_is_bounded_and_preserves_failure():
    error = JudgeError("max_output_tokens")
    judge = Judge([error] * 4)
    with pytest.raises(JudgeError) as caught:
        asyncio.run(judge.decision("q", "a", budget=8192))
    assert caught.value is error
    assert len(judge.calls) == 4


def test_non_judge_bug_is_not_retried():
    judge = Judge([ValueError("bad configuration")])
    with pytest.raises(ValueError, match="bad configuration"):
        asyncio.run(judge.decision("q", "a", budget=8192))
    assert len(judge.calls) == 1


@pytest.mark.parametrize("attempts", [0, 9, True, 1.5])
def test_invalid_budget_fails_before_any_call(attempts):
    judge = Judge([], attempts=attempts)
    with pytest.raises(ValueError, match="judge_max_attempts"):
        asyncio.run(judge.decision("q", "a", budget=8192))
    assert not judge.calls


def test_single_attempt_preserves_legacy_transport_failure():
    judge = Judge([JudgeError("connection reset")], attempts=1)
    with pytest.raises(JudgeError, match="connection reset"):
        asyncio.run(judge.decision("q", "a", budget=8192))
    assert len(judge.calls) == 1
