# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Routing of NeMo-Gym rollout rows across sharded actors.

These cover the pure dispatch layer -- bucketing, the K-stream merge, and the
metrics roll-up -- with fake actors, so they need neither Ray nor a GPU.
"""

import asyncio

import pytest

from nemo_rl.environments.nemo_gym import NemoGymShardSet, as_nemo_gym_shard_set
from nemo_rl.environments.nemo_gym_shards import ShardSetupError
from nemo_rl.experience.rollouts import (
    _bucket_nemo_gym_rows_by_shard,
    _merge_nemo_gym_shard_streams,
    _merge_nemo_gym_timing_metrics,
)


class _FakeStream:
    """One actor's streaming ``run_rollouts`` return value."""

    def __init__(self, rows, shard_name, fail=False, fail_on_await=False):
        self._rows = rows
        self._shard_name = shard_name
        self._fail = fail
        self._fail_on_await = fail_on_await

    async def __aiter__(self):
        for position, row in enumerate(self._rows):
            if self._fail:
                raise RuntimeError(f"{self._shard_name} died")
            timing = (
                {"timing/rollout/await_results": 1.0}
                if position == len(self._rows) - 1
                else None
            )
            # Ray hands back a future per row; the caller awaits it separately.
            future = asyncio.get_running_loop().create_future()
            if self._fail_on_await:
                future.set_exception(RuntimeError(f"{self._shard_name} actor died"))
            else:
                future.set_result((row["_rowidx"], {"row": row["_rowidx"]}, timing))
            yield future


class _FakeRunRollouts:
    def __init__(self, handle):
        self._handle = handle

    def options(self, **kwargs):
        assert kwargs == {"num_returns": "streaming"}
        return self

    def remote(
        self,
        rows,
        tokenizer,
        timer_prefix,
        deduplicate_multimodal_data=False,
    ):
        self._handle.calls.append((rows, timer_prefix))
        return _FakeStream(
            rows,
            self._handle.shard_name,
            fail=self._handle.fail,
            fail_on_await=self._handle.fail_on_await,
        )


class _FakeActor:
    def __init__(self, shard_name, fail=False, fail_on_await=False):
        self.shard_name = shard_name
        self.fail = fail
        self.fail_on_await = fail_on_await
        self.calls = []
        self.run_rollouts = _FakeRunRollouts(self)


def _rows(agent_names, num_generations):
    """One group per agent name, rows stamped with a batch-global ``_rowidx``."""
    rows = []
    for group_index, agent_name in enumerate(agent_names):
        for offset in range(num_generations):
            rows.append(
                {
                    "agent_ref": {"name": agent_name},
                    "_rowidx": group_index * num_generations + offset,
                }
            )
    return rows


def _shard_set(handles, agent_to_shard):
    return NemoGymShardSet(
        handles=handles,
        agent_to_shard=agent_to_shard,
        placement_group=object(),
    )


def test_unsharded_dispatch_keeps_one_bucket_holding_the_whole_batch():
    actor = _FakeActor("only")
    shard_set = as_nemo_gym_shard_set(actor)
    rows = _rows(["alpha", "beta"], num_generations=2)

    buckets = _bucket_nemo_gym_rows_by_shard(rows, shard_set, num_generations=2)

    assert len(buckets) == 1
    _, handle, bucket_rows = buckets[0]
    assert handle is actor
    assert bucket_rows == rows


def test_groups_follow_their_agent_to_its_shard():
    left, right = _FakeActor("left"), _FakeActor("right")
    shard_set = _shard_set(
        {"left": [left], "right": [right]},
        {"alpha": "left", "beta": "right"},
    )
    rows = _rows(["alpha", "beta", "alpha"], num_generations=2)

    buckets = _bucket_nemo_gym_rows_by_shard(rows, shard_set, num_generations=2)

    by_shard = {name: rows for name, _, rows in buckets}
    # Whole groups, and the batch-global row indices survive bucketing -- the
    # accumulator downstream indexes into the full batch.
    assert [row["_rowidx"] for row in by_shard["left"]] == [0, 1, 4, 5]
    assert [row["_rowidx"] for row in by_shard["right"]] == [2, 3]


def test_replicas_of_one_shard_take_turns_by_group():
    first, second = _FakeActor("busy"), _FakeActor("busy")
    shard_set = _shard_set({"busy": [first, second]}, {"alpha": "busy"})
    rows = _rows(["alpha"] * 4, num_generations=2)

    buckets = _bucket_nemo_gym_rows_by_shard(rows, shard_set, num_generations=2)

    assert len(buckets) == 2
    dispatched = {id(handle): rows for _, handle, rows in buckets}
    assert [row["_rowidx"] for row in dispatched[id(first)]] == [0, 1, 4, 5]
    assert [row["_rowidx"] for row in dispatched[id(second)]] == [2, 3, 6, 7]


def test_a_group_that_mixes_agents_across_shards_is_rejected():
    left, right = _FakeActor("left"), _FakeActor("right")
    shard_set = _shard_set(
        {"left": [left], "right": [right]},
        {"alpha": "left", "beta": "right"},
    )
    rows = _rows(["alpha"], num_generations=2)
    rows[1]["agent_ref"]["name"] = "beta"

    with pytest.raises(ValueError, match="mixes agents"):
        _bucket_nemo_gym_rows_by_shard(rows, shard_set, num_generations=2)


def test_a_group_that_mixes_agents_on_one_shard_is_rejected():
    shard_set = _shard_set(
        {"shared": [_FakeActor("shared")]},
        {"alpha": "shared", "beta": "shared"},
    )
    rows = _rows(["alpha"], num_generations=2)
    rows[1]["agent_ref"]["name"] = "beta"

    with pytest.raises(ValueError, match="mixes agents"):
        _bucket_nemo_gym_rows_by_shard(rows, shard_set, num_generations=2)


def test_a_row_naming_an_unhosted_agent_names_the_agents_that_are_hosted():
    shard_set = _shard_set({"left": [_FakeActor("left")]}, {"alpha": "left"})
    rows = _rows(["ghost"], num_generations=2)

    with pytest.raises(ShardSetupError, match="No NeMo-Gym shard hosts agent 'ghost'"):
        _bucket_nemo_gym_rows_by_shard(rows, shard_set, num_generations=2)


async def _drain(buckets):
    return [
        (rowidx, timing, shard_name)
        async for rowidx, _, timing, shard_name in _merge_nemo_gym_shard_streams(
            buckets, tokenizer=object(), timer_prefix="timing/rollout"
        )
    ]


def test_the_merge_drains_every_shard_and_tags_each_row_with_its_own():
    left, right = _FakeActor("left"), _FakeActor("right")
    buckets = [
        ("left", left, _rows(["alpha"], num_generations=2)),
        ("right", right, _rows(["beta"], num_generations=3)),
    ]

    drained = asyncio.run(_drain(buckets))

    assert sorted(rowidx for rowidx, _, _ in drained) == [0, 0, 1, 1, 2]
    assert {shard for _, _, shard in drained} == {"left", "right"}
    # Timing arrives once per shard, on that shard's last row.
    assert sum(timing is not None for _, timing, _ in drained) == 2


def test_a_failing_shard_names_itself_rather_than_surfacing_a_bare_ray_error():
    healthy, broken = _FakeActor("healthy"), _FakeActor("broken", fail=True)
    buckets = [
        ("healthy", healthy, _rows(["alpha"], num_generations=2)),
        ("broken", broken, _rows(["beta"], num_generations=2)),
    ]

    with pytest.raises(RuntimeError, match="shard 'broken' failed"):
        asyncio.run(_drain(buckets))


def test_a_ray_failure_raised_while_awaiting_a_row_names_the_shard():
    buckets = [
        (
            "busy/1",
            _FakeActor("busy", fail_on_await=True),
            _rows(["alpha"], num_generations=2),
        )
    ]

    with pytest.raises(RuntimeError, match="shard 'busy/1' failed"):
        asyncio.run(_drain(buckets))


def test_one_shard_reports_exactly_what_the_actor_reported():
    reported = {
        "timing/rollout/await_results": 4.0,
        "timing/rollout/postprocess_results_pct": 12.5,
    }

    merged = _merge_nemo_gym_timing_metrics({"only": reported}, "timing/rollout")

    assert merged == reported


def test_several_shards_keep_their_own_numbers_and_roll_up_to_the_slowest():
    merged = _merge_nemo_gym_timing_metrics(
        {
            "left": {"timing/rollout/await_results": 4.0},
            "right": {"timing/rollout/await_results": 9.0},
        },
        "timing/rollout",
    )

    assert merged["timing/rollout/shard/left/await_results"] == 4.0
    assert merged["timing/rollout/shard/right/await_results"] == 9.0
    # The step waits for the slowest shard, not for their sum.
    assert merged["timing/rollout/await_results"] == 9.0


def test_no_shard_reported_timing_yields_no_metrics():
    assert _merge_nemo_gym_timing_metrics({}, "timing/rollout") == {}
