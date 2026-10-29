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
"""Batch-level straggler cutoff on the synchronous NeMo-Gym rollout path."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import nemo_rl.environments.nemo_gym as gym_mod
import nemo_rl.experience.rollouts as rollouts_mod
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.nemo_gym import (
    STRAGGLER_CUT_KEY,
    NemoGym,
    StragglerCutoffConfig,
    StragglerCutoffEvent,
    _build_gym_actor_config,
    _iter_gym_results_with_straggler_cutoff,
    _parse_straggler_cutoff,
    _straggler_cut_placeholder,
    _straggler_cutoff_decision,
    _straggler_cutoff_wait_s,
)
from nemo_rl.experience.failures import GymTransportError
from nemo_rl.experience.rollouts import (
    _postprocess_single_nemo_gym_group,
    run_async_nemo_gym_rollout,
)
from nemo_rl.utils.timer import Timer

NemoGymClass = NemoGym.__ray_metadata__.modified_class


# ── config ────────────────────────────────────────────────────────────────


class TestParse:
    @pytest.mark.parametrize("raw", [None, {}, {"enabled": False, "grace_s": 5}])
    def test_off_unless_enabled(self, raw):
        assert _parse_straggler_cutoff(raw) is None

    def test_defaults(self):
        assert _parse_straggler_cutoff({"enabled": True}) == StragglerCutoffConfig()

    def test_values(self):
        cutoff = _parse_straggler_cutoff(
            {
                "enabled": True,
                "done_fraction": 0.95,
                "min_elapsed_s": 1800,
                "grace_s": 300,
                "max_wall_s": 2700,
            }
        )
        assert cutoff == StragglerCutoffConfig(
            done_fraction=0.95, grace_s=300.0, min_elapsed_s=1800.0, max_wall_s=2700.0
        )

    @pytest.mark.parametrize(
        "raw, match",
        [
            ({"enabled": True, "grace": 5}, "Unknown"),
            ({"enabled": True, "done_fraction": 0.0}, "done_fraction"),
            ({"enabled": True, "done_fraction": 1.5}, "done_fraction"),
            ({"enabled": True, "grace_s": -1}, "grace_s"),
            ({"enabled": True, "min_elapsed_s": -1}, "min_elapsed_s"),
            ({"enabled": True, "max_wall_s": 0}, "max_wall_s"),
            ([1, 2], "mapping"),
        ],
    )
    def test_rejects_bad_blocks(self, raw, match):
        with pytest.raises(ValueError, match=match):
            _parse_straggler_cutoff(raw)

    @pytest.mark.parametrize(
        "key", ["done_fraction", "grace_s", "min_elapsed_s", "max_wall_s"]
    )
    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), -float("inf"), "nan", "-inf"]
    )
    def test_rejects_non_finite_values(self, key, value):
        # NaN passes every range check (each comparison is false) and would leave
        # a cutoff that never fires; an infinite timer never fires either.
        with pytest.raises(ValueError, match=f"{key} must be finite"):
            _parse_straggler_cutoff({"enabled": True, key: value})

    def test_rejects_non_numeric_values(self):
        with pytest.raises(ValueError, match="grace_s must be a number"):
            _parse_straggler_cutoff({"enabled": True, "grace_s": "soon"})

    def test_actor_config_keeps_the_block_out_of_gyms_global_config(self):
        cfg = _build_gym_actor_config(
            {"straggler_cutoff": {"enabled": True, "grace_s": 60}, "some_server": {}},
            base_urls=["http://x"],
            model_name="m",
            enable_router_replay=False,
            use_fastokens=False,
        )
        assert cfg["straggler_cutoff"] == {"enabled": True, "grace_s": 60}
        assert "straggler_cutoff" not in cfg["initial_global_config_dict"]

    def test_actor_config_validates_on_the_driver(self):
        with pytest.raises(ValueError, match="done_fraction"):
            _build_gym_actor_config(
                {"straggler_cutoff": {"enabled": True, "done_fraction": 2}},
                base_urls=["http://x"],
                model_name="m",
                enable_router_replay=False,
                use_fastokens=False,
            )

    def test_actor_config_rejects_non_finite_timers_on_the_driver(self):
        with pytest.raises(ValueError, match="grace_s must be finite"):
            _build_gym_actor_config(
                {"straggler_cutoff": {"enabled": True, "grace_s": float("nan")}},
                base_urls=["http://x"],
                model_name="m",
                enable_router_replay=False,
                use_fastokens=False,
            )


# ── decision ──────────────────────────────────────────────────────────────


CUTOFF = StragglerCutoffConfig(
    done_fraction=0.9, grace_s=300.0, min_elapsed_s=600.0, max_wall_s=3000.0
)


@pytest.mark.parametrize(
    "elapsed_s, done, armed_at_s, expected",
    [
        # Not enough rows back: nothing happens.
        (700.0, 80, None, (None, None)),
        # Enough rows back but min_elapsed_s has not passed: not armed yet.
        (500.0, 95, None, (None, None)),
        # Enough rows back after min_elapsed_s: arms, grace starts.
        (700.0, 95, None, (700.0, None)),
        # Armed, grace not over.
        (900.0, 95, 700.0, (700.0, None)),
        # Armed, grace over.
        (1000.0, 95, 700.0, (700.0, "grace")),
        # The hard cap fires regardless of done fraction.
        (3000.0, 10, None, (None, "max_wall")),
    ],
)
def test_decision(elapsed_s, done, armed_at_s, expected):
    assert (
        _straggler_cutoff_decision(
            CUTOFF, elapsed_s=elapsed_s, done=done, total=100, armed_at_s=armed_at_s
        )
        == expected
    )


@pytest.mark.parametrize(
    "elapsed_s, done, armed_at_s, expected",
    [
        (100.0, 10, None, 2900.0),  # only the hard cap is ahead
        (100.0, 95, None, 500.0),  # waiting for min_elapsed_s
        (800.0, 95, 700.0, 200.0),  # waiting for the grace to end
        (3100.0, 10, None, 0.0),  # overdue
    ],
)
def test_wait_s(elapsed_s, done, armed_at_s, expected):
    assert _straggler_cutoff_wait_s(
        CUTOFF, elapsed_s=elapsed_s, done=done, total=100, armed_at_s=armed_at_s
    ) == pytest.approx(expected)


def test_wait_s_without_a_deadline_waits_for_the_next_completion():
    cutoff = StragglerCutoffConfig(done_fraction=0.9, grace_s=1.0)
    assert (
        _straggler_cutoff_wait_s(
            cutoff, elapsed_s=5.0, done=1, total=10, armed_at_s=None
        )
        is None
    )


# ── cancelling stream ─────────────────────────────────────────────────────


class _FakeCompletions:
    """Stands in for Gym's run_examples return value: completion-order and closable."""

    def __init__(self, rows: list[dict], delays: dict[int, float | None]):
        self.cancelled: list[int] = []
        self.closed = False
        self._claimed: set[int] = set()

        async def run(row):
            delay = delays.get(row["_rowidx"])
            try:
                if delay is None:
                    await asyncio.Event().wait()  # never finishes on its own
                elif isinstance(delay, Exception):
                    raise delay
                else:
                    await asyncio.sleep(delay)
            except asyncio.CancelledError:
                self.cancelled.append(row["_rowidx"])
                raise
            return row, {"reward": 1.0, "row": row["_rowidx"]}

        self._tasks = [asyncio.ensure_future(run(row)) for row in rows]
        self._completions = asyncio.as_completed(self._tasks)

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        return self._next_completed(next(self._completions))

    async def _next_completed(self, completion):
        row, result = await completion
        if self.closed:
            raise asyncio.CancelledError
        self._claimed.add(row["_rowidx"])
        return row, result

    async def aclose_and_drain(self):
        if self.closed:
            return []
        self.closed = True
        completed = [
            task
            for index, task in enumerate(self._tasks)
            if task.done() and index not in self._claimed
        ]
        await self.aclose()
        return completed

    async def aclose(self):
        self.closed = True
        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


def _rows(n: int) -> list[dict]:
    return [{"_rowidx": i, "agent_ref": {"name": "agent"}} for i in range(n)]


async def _collect(rows, delays, cutoff):
    completions = _FakeCompletions(rows, delays)
    stats: dict[str, float] = {}
    out = [
        (row["_rowidx"], result, bool(was_cut))
        async for row, result, was_cut in _iter_gym_results_with_straggler_cutoff(
            completions, rows, cutoff, Timer(), "timing/test", stats
        )
    ]
    return out, stats, completions


def test_cuts_the_tail_after_the_grace():
    rows = _rows(4)
    cutoff = StragglerCutoffConfig(done_fraction=0.5, grace_s=0.05)
    out, stats, completions = asyncio.run(
        _collect(rows, {0: 0.0, 1: 0.01, 2: None, 3: None}, cutoff)
    )

    assert [(idx, was_cut) for idx, _, was_cut in out] == [
        (0, False),
        (1, False),
        (2, True),
        (3, True),
    ]
    assert [result for _, result, was_cut in out if was_cut] == [None, None]
    assert completions.closed
    assert sorted(completions.cancelled) == [2, 3]
    assert stats["straggler_cutoff/num_cut"] == 2.0
    assert stats["straggler_cutoff/armed"] == 1.0
    assert stats["straggler_cutoff/done_fraction_at_cut"] == 0.5
    assert stats["straggler_cutoff/cut_by_max_wall"] == 0.0
    assert stats["straggler_cutoff/cut_at_s"] >= stats["straggler_cutoff/armed_at_s"]


def test_no_cut_when_everything_finishes():
    rows = _rows(3)
    cutoff = StragglerCutoffConfig(done_fraction=0.5, grace_s=5.0)
    out, stats, completions = asyncio.run(
        _collect(rows, {0: 0.0, 1: 0.01, 2: 0.02}, cutoff)
    )

    assert sorted(idx for idx, _, _ in out) == [0, 1, 2]
    assert not any(was_cut for _, _, was_cut in out)
    assert completions.closed
    assert stats["straggler_cutoff/num_cut"] == 0.0


def test_min_elapsed_holds_the_cutoff_back():
    rows = _rows(2)
    # Row 1 finishes after 0.1s; the cutoff could arm right after row 0 but must
    # not before min_elapsed_s, and the grace then outlasts row 1.
    cutoff = StragglerCutoffConfig(done_fraction=0.5, grace_s=0.5, min_elapsed_s=0.05)
    out, stats, _ = asyncio.run(_collect(rows, {0: 0.0, 1: 0.1}, cutoff))

    assert not any(was_cut for _, _, was_cut in out)
    assert stats["straggler_cutoff/armed_at_s"] >= 0.05


def test_max_wall_cuts_before_the_done_fraction_is_reached():
    rows = _rows(4)
    cutoff = StragglerCutoffConfig(done_fraction=0.9, grace_s=100.0, max_wall_s=0.05)
    out, stats, completions = asyncio.run(
        _collect(rows, {0: 0.0, 1: None, 2: None, 3: None}, cutoff)
    )

    assert [idx for idx, _, was_cut in out if was_cut] == [1, 2, 3]
    assert sorted(completions.cancelled) == [1, 2, 3]
    assert stats["straggler_cutoff/cut_by_max_wall"] == 1.0
    assert stats["straggler_cutoff/armed"] == 0.0


def test_a_failed_rollout_still_raises_typed():
    class _HttpError(Exception):
        status = 503

    rows = _rows(2)
    cutoff = StragglerCutoffConfig(done_fraction=0.5, grace_s=5.0)

    async def collect():
        completions = _FakeCompletions(rows, {0: _HttpError("down"), 1: None})
        try:
            with pytest.raises(GymTransportError, match="HTTP 503"):
                async for _ in _iter_gym_results_with_straggler_cutoff(
                    completions, rows, cutoff, Timer(), "timing/test", {}
                ):
                    pytest.fail("a transport failure must not become a cut row")
            # Assert before asyncio.run's shutdown can hide leaked tasks.
            assert completions.closed
            assert completions.cancelled == [1]
            assert all(task.done() for task in completions._tasks)
        finally:
            await completions.aclose()

    asyncio.run(collect())


@pytest.mark.parametrize("fails", [False, True])
def test_cutoff_harvests_an_already_claimed_pending_waiter(monkeypatch, fails):
    class HttpError(Exception):
        status = 503

    clock = [0.0]
    monkeypatch.setattr(gym_mod, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gym_mod, "_straggler_cutoff_wait_s", lambda *args, **kwargs: 0)
    rows = _rows(3)

    class ClaimedWaiter:
        def __init__(self):
            self.release = asyncio.Event()
            self.claimed = False
            self.closed = False
            self.requests = 0

        def __next__(self):
            self.requests += 1
            return self.completion()

        async def completion(self):
            self.claimed = True
            clock[0] = 2.0
            await self.release.wait()
            if fails:
                raise HttpError("claimed failure")
            return rows[0], {"reward": 1}

        async def aclose_and_drain(self):
            assert self.claimed
            self.closed = True
            self.release.set()
            ready = asyncio.get_running_loop().create_future()
            ready.set_result((rows[1], {"reward": 1}))
            return [ready]

        async def aclose(self):
            self.closed = True
            self.release.set()

    async def collect():
        completions = ClaimedWaiter()
        stream = _iter_gym_results_with_straggler_cutoff(
            completions,
            rows,
            StragglerCutoffConfig(done_fraction=1, grace_s=10, max_wall_s=1),
            Timer(),
            "timing/race",
            {},
        )
        if fails:
            with pytest.raises(GymTransportError, match="HTTP 503"):
                await anext(stream)
        else:
            out = [
                (row["_rowidx"], event is not None) async for row, _, event in stream
            ]
            assert out == [(0, False), (1, False), (2, True)]
        assert completions.requests == 1
        assert completions.closed
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(collect())


@pytest.mark.parametrize("exit_path", ["early_close", "cancellation"])
def test_cutoff_iterator_closes_on_non_cutoff_exit(exit_path):
    async def collect():
        rows = _rows(2)
        completions = _FakeCompletions(rows, {0: 0.0, 1: None})
        stream = _iter_gym_results_with_straggler_cutoff(
            completions, rows, StragglerCutoffConfig(), Timer(), "timing/test", {}
        )
        try:
            await anext(stream)
            if exit_path == "cancellation":
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                await stream.aclose()
            assert completions.closed
            assert completions.cancelled == [1]
            assert all(task.done() for task in completions._tasks)
            assert asyncio.all_tasks() == {asyncio.current_task()}
        finally:
            await stream.aclose()
            await completions.aclose()

    asyncio.run(collect())


# ── placeholder ───────────────────────────────────────────────────────────


PLACEHOLDER_EVENT = StragglerCutoffEvent(
    deadline="max_wall_s",
    deadline_s=1.0,
    elapsed_s=1.1,
    armed_at_s=None,
    config=StragglerCutoffConfig(max_wall_s=1.0),
)


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, template_fails: bool = False):
        self.template_fails = template_fails
        self.templated: list[Any] = []

    def apply_chat_template(self, messages, tokenize=True):
        if self.template_fails:
            raise ValueError("template rejects these messages")
        self.templated.append(messages)
        return [10 + len(message["content"]) for message in messages]

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_placeholder_is_a_text_only_one_token_sample():
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "fix it"},
                        {"type": "input_image", "image_url": "file:///x.png"},
                    ],
                },
                {"type": "function_call", "name": "f", "arguments": "{}"},
            ]
        }
    }
    tokenizer = _FakeTokenizer()
    result = _straggler_cut_placeholder(row, tokenizer, PLACEHOLDER_EVENT)

    assert tokenizer.templated == [
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix it"}]
    ]
    user, assistant = result["message_log"]
    assert user["token_ids"].tolist() == [13, 16]
    assert assistant["token_ids"].tolist() == [2]
    assert assistant["generation_logprobs"].tolist() == [0.0]
    assert result["input_message_log"] == [user]
    assert result["full_result"]["reward"] == 0.0
    assert result["full_result"][STRAGGLER_CUT_KEY] is True


def test_placeholder_falls_back_to_plain_text_and_never_empty():
    row = {"responses_create_params": {"input": "hi"}}
    result = _straggler_cut_placeholder(
        row, _FakeTokenizer(template_fails=True), PLACEHOLDER_EVENT
    )
    assert result["message_log"][0]["token_ids"].tolist() == [ord("h"), ord("i")]

    empty = _straggler_cut_placeholder({}, _FakeTokenizer(), PLACEHOLDER_EVENT)
    assert empty["message_log"][0]["token_ids"].tolist() == [2]


# ── actor ─────────────────────────────────────────────────────────────────


class _FakeRolloutHelper:
    def __init__(
        self,
        delays: dict[int, float | None],
        closable: bool = True,
        drainable: bool = True,
    ):
        self.delays = delays
        self.closable = closable
        self.drainable = drainable
        self.completions: _FakeCompletions | None = None

    def run_examples(self, examples, head_server_config):
        self.completions = _FakeCompletions(examples, self.delays)
        if self.closable:
            if not self.drainable:
                return _CloseOnlyCompletions(self.completions)
            return self.completions
        return iter(list(self.completions))


class _CloseOnlyCompletions:
    def __init__(self, completions):
        self.completions = completions

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.completions)

    async def aclose(self):
        await self.completions.aclose()


def _actor(cutoff: dict | None, rollout_helper: _FakeRolloutHelper) -> NemoGymClass:
    env = NemoGymClass(
        {
            "model_name": "m",
            "base_urls": [],
            "initial_global_config_dict": {},
            "straggler_cutoff": cutoff,
        }
    )
    env.rh = object()
    env._tokenizer = _FakeTokenizer()
    env.head_server_config = "head"
    env.rch = rollout_helper
    env._postprocess_nemo_gym_to_nemo_rl_result = lambda row, result, *_a, **_k: {
        "message_log": [],
        "full_result": result,
    }
    return env


def _run(env, rows, allow):
    async def collect():
        return [
            item
            async for item in env.run_rollouts(
                rows, "timing/test", allow_straggler_cutoff=allow
            )
        ]

    return asyncio.run(collect())


ENABLED = {"enabled": True, "done_fraction": 0.5, "grace_s": 0.05}


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize(
    "exit_path", ["early_close", "postprocess_error", "cancellation"]
)
def test_actor_closes_completions_on_every_exit(allow, exit_path):
    async def collect():
        helper = _FakeRolloutHelper({0: 0.0, 1: None})
        env = _actor(ENABLED, helper)
        if exit_path == "postprocess_error":

            def fail(*args, **kwargs):
                raise ValueError("bad response")

            env._postprocess_nemo_gym_to_nemo_rl_result = fail
        stream = env.run_rollouts(_rows(2), "timing/test", allow_straggler_cutoff=allow)
        try:
            if exit_path == "postprocess_error":
                with pytest.raises(ValueError, match="bad response"):
                    await anext(stream)
            else:
                await anext(stream)
                if exit_path == "cancellation":
                    pending = asyncio.create_task(anext(stream))
                    await asyncio.sleep(0)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                else:
                    await stream.aclose()
            assert helper.completions.closed
            assert helper.completions.cancelled == [1]
            assert all(task.done() for task in helper.completions._tasks)
            assert asyncio.all_tasks() == {asyncio.current_task()}
        finally:
            await stream.aclose()
            await helper.completions.aclose()

    asyncio.run(collect())


@pytest.mark.parametrize("deadline", ["done_fraction_grace", "max_wall_s"])
def test_cut_rows_record_trainer_deadline_separately_from_failures(deadline):
    cutoff = dict(ENABLED)
    if deadline == "max_wall_s":
        cutoff.update(done_fraction=1.0, max_wall_s=0.02)
    helper = _FakeRolloutHelper({0: 0.0, 1: None})
    streamed = _run(_actor(cutoff, helper), _rows(2), allow=True)
    finished = streamed[0][2]["full_result"]
    cut = streamed[1][2]["full_result"]
    assert "failure_kind" not in finished
    assert cut["_nrl_straggler_cut"] is True
    assert not any(key.startswith("_ng_") for key in cut)
    assert cut["failure_kind"] == "trainer_cutoff"
    event = cut["_nrl_straggler_cutoff"]
    assert event["deadline"] == deadline
    assert event["elapsed_s"] == streamed[-1][3]["straggler_cutoff/cut_at_s"]
    assert event["elapsed_s"] >= event["deadline_s"]
    assert deadline in cut["failure_reason"]
    if deadline == "max_wall_s":
        assert event["deadline_s"] == cutoff["max_wall_s"]
        assert event["armed_at_s"] is None
    else:
        assert event["deadline_s"] == event["armed_at_s"] + cutoff["grace_s"]
        assert event["config"]["done_fraction"] == cutoff["done_fraction"]
        assert event["config"]["grace_s"] == cutoff["grace_s"]


def test_actor_closes_completions_when_resolved_identity_is_invalid():
    async def collect():
        helper = _FakeRolloutHelper({0: None})
        env = _actor(ENABLED, helper)
        stream = env.run_rollouts(
            [{"_rowidx": 0}], "timing/test", allow_straggler_cutoff=True
        )
        try:
            with pytest.raises(RuntimeError, match="agent_ref"):
                await anext(stream)
            assert helper.completions.closed
            assert all(task.done() for task in helper.completions._tasks)
        finally:
            await stream.aclose()
            await helper.completions.aclose()

    asyncio.run(collect())


@pytest.mark.parametrize("kind", ["cancelled", "agent_timeout", "verifier_error"])
def test_genuine_gym_failure_kind_is_preserved(kind):
    helper = _FakeRolloutHelper({0: 0.0, 1: None})
    env = _actor(ENABLED, helper)

    def postprocess(row, result, *args, **kwargs):
        return {"message_log": [], "full_result": {"reward": 0.0, "failure_kind": kind}}

    env._postprocess_nemo_gym_to_nemo_rl_result = postprocess
    streamed = _run(env, _rows(2), allow=True)
    assert streamed[0][2]["full_result"] == {"reward": 0.0, "failure_kind": kind}
    assert streamed[1][2]["full_result"]["failure_kind"] == "trainer_cutoff"


def test_run_rollouts_cuts_and_reports_on_the_last_row():
    helper = _FakeRolloutHelper({0: 0.0, 1: 0.0, 2: None})
    streamed = _run(_actor(ENABLED, helper), _rows(3), allow=True)

    assert [rowidx for rowidx, *_ in streamed] == [0, 1, 2]
    cut = streamed[-1][2]
    assert cut["full_result"][STRAGGLER_CUT_KEY] is True
    assert cut["message_log"][1]["token_ids"].tolist() == [2]
    assert streamed[-1][1] == {"name": "agent"}
    assert all(item[3] is None for item in streamed[:-1])
    assert streamed[-1][3]["straggler_cutoff/num_cut"] == 1.0
    assert helper.completions.cancelled == [2]


def test_run_rollouts_without_permission_never_cuts():
    helper = _FakeRolloutHelper({0: 0.0, 1: 0.0, 2: 0.2})
    streamed = _run(_actor(ENABLED, helper), _rows(3), allow=False)

    assert not any(STRAGGLER_CUT_KEY in item[2]["full_result"] for item in streamed)
    assert "straggler_cutoff/num_cut" not in streamed[-1][3]
    assert helper.completions.cancelled == []


def test_run_rollouts_disabled_never_cuts():
    helper = _FakeRolloutHelper({0: 0.0, 1: 0.2})
    streamed = _run(_actor(None, helper), _rows(2), allow=True)
    assert not any(STRAGGLER_CUT_KEY in item[2]["full_result"] for item in streamed)


@pytest.mark.parametrize("closable", [False, True])
def test_run_rollouts_on_a_gym_without_drain_runs_uncut(capsys, closable):
    helper = _FakeRolloutHelper({0: 0.0, 1: 0.2}, closable=closable, drainable=False)
    env = _actor(ENABLED, helper)
    streamed = _run(env, _rows(2), allow=True)
    _run(env, _rows(2), allow=True)

    assert not any(STRAGGLER_CUT_KEY in item[2]["full_result"] for item in streamed)
    assert capsys.readouterr().err.count("running without the cutoff") == 1


def test_run_rollouts_in_receipt_mode_runs_uncut(capsys):
    helper = _FakeRolloutHelper({0: 0.0, 1: 0.2})
    env = _actor(ENABLED, helper)
    env._token_capture_enabled = True

    async def receipt(row, result):
        return {"message_log": [], "full_result": result}

    env._postprocess_receipt_mode = receipt
    streamed = _run(env, _rows(2), allow=True)

    # A cancelled rollout has no receipt for the finalizer to rebuild, so the
    # cutoff stays off in receipt mode and the batch runs to completion.
    assert not any(STRAGGLER_CUT_KEY in item[2]["full_result"] for item in streamed)
    assert helper.completions.cancelled == []
    assert "receipt mode" in capsys.readouterr().err


# ── driver ────────────────────────────────────────────────────────────────


GENERATION_CONFIG = {
    "max_new_tokens": 16,
    "stop_strings": [],
    "stop_token_ids": [],
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": None,
}


def _driver_rows(n: int) -> list[dict]:
    return [
        {
            "task_source": "agent",
            "agent_ref": {"name": "agent"},
            "responses_create_params": {"input": "hi"},
        }
        for _ in range(n)
    ]


def test_allow_straggler_cutoff_requires_returns_entire_batch():
    """A per-group stream could not tell a whole-batch cutoff where its batch ends."""

    async def collect():
        return [
            result
            async for result in run_async_nemo_gym_rollout(
                policy_generation=SimpleNamespace(
                    cfg={"max_total_sequence_length": 128}
                ),
                input_batch=BatchedDataDict({"extra_env_info": _driver_rows(2)}),
                tokenizer=None,
                task_to_env={},
                generation_config=GENERATION_CONFIG,
                num_generations=2,
                log_full_result_tables=False,
                allow_straggler_cutoff=True,
            )
        ]

    with pytest.raises(
        ValueError, match="allow_straggler_cutoff requires returns_entire_batch"
    ):
        asyncio.run(collect())


class _ReadyRef:
    """Ray hands back a reference per streamed row; the driver awaits it."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def _resolve():
            return self.value

        return _resolve().__await__()


class _AsyncOnlyStream:
    def __init__(self, values):
        self.values = iter(values)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.values)
        except StopIteration as error:
            raise StopAsyncIteration from error


def _finished_result(rowidx: int) -> dict:
    return {
        "full_result": {"reward": 1.0},
        "input_message_log": [{"role": "user", "token_ids": [rowidx, 7]}],
        "message_log": [
            {"role": "user", "token_ids": [rowidx, 7]},
            {"role": "assistant", "token_ids": [3], "generation_logprobs": [0.0]},
        ],
    }


def test_driver_gives_cut_rows_no_media_back(monkeypatch):
    """With deduplicated media, a finished row gets its tensors back; a placeholder does not.

    The placeholder's prompt is text only, so re-attaching pixel values would
    hand the policy image tensors with no image placeholder tokens.
    """
    rows = _driver_rows(2)
    media = [PackedTensor(torch.tensor([[float(i)]]), dim_to_pack=0) for i in range(2)]
    input_batch = BatchedDataDict(
        {
            "extra_env_info": rows,
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "prompt",
                        "token_ids": torch.tensor([1, 2]),
                        "pixel_values": media[i],
                    }
                ]
                for i in range(2)
            ],
            "loss_multiplier": torch.ones(2),
        }
    )

    class _RunRollouts:
        def options(self, *, num_returns):
            assert num_returns == "streaming"
            return self

        def remote(self, rows, timer_prefix, deduplicate_multimodal_data, **kwargs):
            assert deduplicate_multimodal_data is True
            assert kwargs == {"allow_straggler_cutoff": True}
            cut = _straggler_cut_placeholder(
                rows[1], _FakeTokenizer(), PLACEHOLDER_EVENT
            )
            return _AsyncOnlyStream(
                [
                    _ReadyRef((0, {"name": "agent"}, _finished_result(0), None)),
                    _ReadyRef(
                        (1, {"name": "agent"}, cut, {"straggler_cutoff/num_cut": 1.0})
                    ),
                ]
            )

    captured: dict[str, Any] = {}

    def _postprocess_group(**kwargs):
        captured["results"] = kwargs["results"]
        return rollouts_mod.NemoGymRolloutResult(
            input_ids=torch.empty(0),
            final_batch=kwargs["input_batch"],
            rollout_metrics={},
            task_index=None,
        )

    monkeypatch.setattr(
        rollouts_mod, "_postprocess_single_nemo_gym_group", _postprocess_group
    )
    env = SimpleNamespace(run_rollouts=_RunRollouts())

    async def collect():
        return [
            result
            async for result in run_async_nemo_gym_rollout(
                policy_generation=SimpleNamespace(
                    cfg={"max_total_sequence_length": 128}
                ),
                input_batch=input_batch,
                tokenizer=None,
                task_to_env={"nemo_gym": env},
                generation_config=GENERATION_CONFIG,
                num_generations=2,
                log_full_result_tables=False,
                deduplicate_multimodal_data=True,
                returns_entire_batch=True,
                allow_straggler_cutoff=True,
            )
        ]

    (rollout_result,) = asyncio.run(collect())
    finished, cut = captured["results"]
    assert cut["full_result"][STRAGGLER_CUT_KEY] is True
    assert finished["message_log"][0]["pixel_values"] is media[0]
    assert finished["input_message_log"][0]["pixel_values"] is media[0]
    for message in cut["message_log"] + cut["input_message_log"]:
        assert "pixel_values" not in message
    # The single actor's stats pass through the roll-up unchanged.
    assert rollout_result.rollout_metrics["straggler_cutoff/num_cut"] == 1.0


# ── batch ─────────────────────────────────────────────────────────────────


class _FakeGeneration:
    cfg = {"max_total_sequence_length": 100}


def _finished(reward: float) -> dict:
    answer = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "42"}],
    }
    return {
        "full_result": {"reward": reward, "response": {"output": [answer]}, "turns": 3},
        "message_log": [
            {"role": "user", "token_ids": torch.tensor([3, 4])},
            {"role": "assistant", "token_ids": torch.tensor([1, 2])},
        ],
        "input_message_log": [{"role": "user", "token_ids": torch.tensor([3, 4])}],
    }


@pytest.mark.parametrize("mask_env_flagged_samples", [True, False])
def test_cut_rows_never_train_and_stay_out_of_agent_metrics(mask_env_flagged_samples):
    results = [
        _finished(1.0),
        _straggler_cut_placeholder({}, _FakeTokenizer(), PLACEHOLDER_EVENT),
        _finished(0.0),
    ]
    rollout_result = _postprocess_single_nemo_gym_group(
        nemo_gym_rows=[{"agent_ref": {"name": "agent"}} for _ in results],
        results=results,
        timer=Timer(),
        timer_prefix="timing/test",
        policy_generation=_FakeGeneration(),
        input_batch=BatchedDataDict({"loss_multiplier": torch.ones(3)}),
        tokenizer=_FakeTokenizer(),
        log_full_result_tables=False,
        mask_env_flagged_samples=mask_env_flagged_samples,
    )

    final_batch = rollout_result.final_batch
    assert final_batch["loss_multiplier"].tolist() == [1.0, 0.0, 1.0]
    if mask_env_flagged_samples:
        assert final_batch["mask_sample"].tolist() == [False, True, False]
    assert final_batch["total_reward"].tolist() == [1.0, 0.0, 0.0]
    # Per-agent full-result and token means cover the two real rollouts only.
    assert rollout_result.rollout_metrics["agent/turns/mean"] == 3.0
    assert rollout_result.rollout_metrics["agent/reward/mean"] == 0.5
    assert rollout_result.rollout_metrics["agent/gen_tokens_per_sample/mean"] == 2.0


def test_cut_rows_take_no_part_in_reward_penalties():
    results = [
        _finished(1.0),
        _straggler_cut_placeholder({}, _FakeTokenizer(), PLACEHOLDER_EVENT),
        _finished(0.0),
    ]
    rollout_result = _postprocess_single_nemo_gym_group(
        nemo_gym_rows=[{"agent_ref": {"name": "agent"}} for _ in results],
        results=results,
        timer=Timer(),
        timer_prefix="timing/test",
        policy_generation=_FakeGeneration(),
        input_batch=BatchedDataDict({"loss_multiplier": torch.ones(3)}),
        tokenizer=_FakeTokenizer(),
        log_full_result_tables=False,
        reward_penalty_config={"penalize_empty_final_answer": True},
    )

    # The placeholder's response.output is empty, but it is not a model output
    # that answered nothing: only the two real rollouts are checked, and both
    # answered, so the rate is 0 over the scored rows rather than 1/3.
    assert rollout_result.rollout_metrics["empty_final_answer_rate"] == 0.0
    assert rollout_result.final_batch["total_reward"].tolist() == [1.0, 0.0, 0.0]
    assert rollout_result.final_batch["loss_multiplier"].tolist() == [1.0, 0.0, 1.0]
