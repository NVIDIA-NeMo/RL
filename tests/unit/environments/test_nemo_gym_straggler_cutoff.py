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
from typing import Any

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.nemo_gym import (
    STRAGGLER_CUT_KEY,
    NemoGym,
    StragglerCutoffConfig,
    _build_gym_actor_config,
    _iter_gym_results_with_straggler_cutoff,
    _parse_straggler_cutoff,
    _straggler_cut_placeholder,
    _straggler_cutoff_decision,
    _straggler_cutoff_wait_s,
)
from nemo_rl.experience.failures import GymTransportError
from nemo_rl.experience.rollouts import _postprocess_single_nemo_gym_group
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
        return next(self._completions)

    async def aclose(self):
        self.closed = True
        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def _rows(n: int) -> list[dict]:
    return [{"_rowidx": i, "agent_ref": {"name": "agent"}} for i in range(n)]


async def _collect(rows, delays, cutoff):
    completions = _FakeCompletions(rows, delays)
    stats: dict[str, float] = {}
    out = [
        (row["_rowidx"], result, was_cut)
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
    assert not completions.closed
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
    with pytest.raises(GymTransportError, match="HTTP 503"):
        asyncio.run(_collect(rows, {0: _HttpError("down"), 1: None}, cutoff))


# ── placeholder ───────────────────────────────────────────────────────────


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
    result = _straggler_cut_placeholder(row, tokenizer)

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
    result = _straggler_cut_placeholder(row, _FakeTokenizer(template_fails=True))
    assert result["message_log"][0]["token_ids"].tolist() == [ord("h"), ord("i")]

    empty = _straggler_cut_placeholder({}, _FakeTokenizer())
    assert empty["message_log"][0]["token_ids"].tolist() == [2]


# ── actor ─────────────────────────────────────────────────────────────────


class _FakeRolloutHelper:
    def __init__(self, delays: dict[int, float | None], closable: bool = True):
        self.delays = delays
        self.closable = closable
        self.completions: _FakeCompletions | None = None

    def run_examples(self, examples, head_server_config):
        self.completions = _FakeCompletions(examples, self.delays)
        if self.closable:
            return self.completions
        return iter(list(self.completions))


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


def test_run_rollouts_on_a_gym_without_aclose_runs_uncut(capsys):
    helper = _FakeRolloutHelper({0: 0.0, 1: 0.2}, closable=False)
    env = _actor(ENABLED, helper)
    streamed = _run(env, _rows(2), allow=True)
    _run(env, _rows(2), allow=True)

    assert not any(STRAGGLER_CUT_KEY in item[2]["full_result"] for item in streamed)
    assert capsys.readouterr().err.count("running without the cutoff") == 1


# ── batch ─────────────────────────────────────────────────────────────────


class _FakeGeneration:
    cfg = {"max_total_sequence_length": 100}


def _finished(reward: float) -> dict:
    return {
        "full_result": {"reward": reward, "response": {"output": []}, "turns": 3},
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
        _straggler_cut_placeholder({}, _FakeTokenizer()),
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
    # Per-agent full-result means cover the two real rollouts only.
    assert rollout_result.rollout_metrics["agent/turns/mean"] == 3.0
    assert rollout_result.rollout_metrics["agent/reward/mean"] == 0.5
