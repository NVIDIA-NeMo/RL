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
"""The advantage stage with positive_length_reweight's per-turn scopes.

Drives the SingleController's real _advantage_stage (in process and through a sharded pool) on
rows carrying the assistant_turn_id / thinking_mask / action_mask columns the RolloutReassembler
publishes, and checks the order: estimator -> per-turn reweight -> message penalties -> clip.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
    reweight_positive_text_per_turn,
)
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    AdvantageComputer,
    AdvantageStageConfig,
)
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ACTION_MASK,
    ASSISTANT_TURN_ID,
    GROUP_ID_TAG,
    INVALID_TOOL_CALL_MASK,
    MALFORMED_THINKING_MASK,
    THINKING_MASK,
)
from nemo_rl.utils.timer import Timer
from tests.unit.single_controller.test_advantage_stage import _InlineActor, _RowStore

GROUPS, SIZE, SEQ = 4, 4, 40


def _row_columns(turns, rng):
    """[(thinking, tool)] per turn -> mask / turn id / thinking / action of length SEQ."""
    mask = torch.zeros(SEQ)
    turn_id = torch.full((SEQ,), -1, dtype=torch.int32)
    thinking = torch.zeros(SEQ, dtype=torch.bool)
    action = torch.zeros(SEQ, dtype=torch.bool)
    pos = 1  # position 0 is the prompt
    for t, (n_think, n_tool) in enumerate(turns):
        if t > 0:
            pos += 1  # tool output
        for k in range(n_think + n_tool + 1):
            mask[pos] = 1.0
            turn_id[pos] = t
            thinking[pos] = k < n_think
            action[pos] = k >= n_think
            pos += 1
    assert pos <= SEQ
    return mask, turn_id, thinking, action


def _rows():
    gen = torch.Generator().manual_seed(1)
    rows = {}
    for g in range(GROUPS):
        for m in range(SIZE):
            n_turns = int(torch.randint(1, 4, (), generator=gen))
            turns = [
                (
                    int(torch.randint(1, 8, (), generator=gen)),
                    int(torch.randint(0, 3, (), generator=gen)),
                )
                for _ in range(n_turns)
            ]
            mask, turn_id, thinking, action = _row_columns(turns, gen)
            invalid = torch.zeros(SEQ, dtype=torch.bool)
            if g == 0 and m == 0:
                invalid[1:3] = (
                    True  # an invalid tool call step in a positive-heavy group
                )
            rows[f"s{g * SIZE + m}"] = {
                "total_reward": torch.tensor(float(m % 2 == 0 or (g == 1 and m == 1))),
                "token_mask": mask,
                "sample_mask": torch.tensor(1.0),
                "mask_sample": torch.tensor(False),
                "truncated": torch.tensor(False),
                "generation_logprobs": -torch.rand(SEQ, generator=gen),
                "input_ids": torch.arange(SEQ),
                "input_lengths": torch.tensor(SEQ),
                "prompt_ids_for_adv": torch.tensor([g, m]),
                ASSISTANT_TURN_ID: turn_id,
                THINKING_MASK: thinking,
                ACTION_MASK: action,
                INVALID_TOOL_CALL_MASK: invalid,
                MALFORMED_THINKING_MASK: torch.zeros(SEQ, dtype=torch.bool),
            }
    return rows


def _meta():
    n = GROUPS * SIZE
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"s{i}" for i in range(n)],
        fields=["total_reward"],
        tags=[
            {"weight_version": 0, GROUP_ID_TAG: f"group-{i // SIZE}"} for i in range(n)
        ],
    )


def _controller(store, *, num_actors, reweight, penalties, clip_high=None):
    algo = GRPOConfig(
        num_generations_per_prompt=SIZE,
        adv_estimator=AdvEstimatorConfig(positive_length_reweight=reweight),
        seq_logprob_error_threshold=None,
        invalid_tool_call_advantage=-2.0 if penalties else None,
        advantage_clip_high=clip_high,
    )
    estimator = GRPOAdvantageEstimator(algo.adv_estimator, None)
    config = AdvantageStageConfig(
        advantage=AdvantageConfig(),
        algo=algo,
        is_ppo=False,
        policy_logprobs_required=False,
        reference_logprobs_required=False,
        teacher_logprobs_required=False,
        message_level_advantage_penalties_enabled=penalties,
        shardable=True,
        train_data_dump_dir=None,
        per_turn_length_reweight=(
            algo.adv_estimator.positive_length_reweight
            if algo.adv_estimator.positive_length_reweight.per_turn
            else None
        ),
    )
    ctrl = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    ctrl._advantage_estimator = estimator
    ctrl._advantage_stage_config = config
    ctrl._advantage_computer = AdvantageComputer(
        store, config=config, advantage_estimator=estimator
    )
    ctrl._train_data_dump = None
    ctrl._timer = Timer()
    ctrl._train_data_dump_rows = 0
    ctrl._train_steps = 0
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._advantage_actors = [
        _InlineActor(
            AdvantageComputer(
                store, config=config, advantage_estimator=estimator, shard_id=str(i)
            )
        )
        for i in range(num_actors)
    ]
    ctrl._available_advantage_actors = asyncio.Queue()
    for actor in ctrl._advantage_actors:
        ctrl._available_advantage_actors.put_nowait(actor)
    ctrl._opd_stat_sum = ctrl._opd_stat_sumsq = 0.0
    ctrl._opd_stat_count = 0
    ctrl._opd_gap_sum = 0.0
    ctrl._length_reweight_stats = None
    ctrl._step_log_dict = {
        "reward_partials": [],
        "advantage_partials": [],
        "num_mask_sample_filtered": [],
        "seq_logprob_error_metrics": [],
    }
    return ctrl


def _stage(reweight, *, num_actors=0, penalties=False, clip_high=None):
    store = _RowStore(_rows())
    ctrl = _controller(
        store,
        num_actors=num_actors,
        reweight=reweight,
        penalties=penalties,
        clip_high=clip_high,
    )
    asyncio.run(ctrl._advantage_stage(_meta()))
    adv = torch.stack([store.rows[f"s{i}"]["advantages"] for i in range(GROUPS * SIZE)])
    return ctrl, adv


def _batch():
    rows = _rows()
    ids = [f"s{i}" for i in range(GROUPS * SIZE)]
    stack = lambda key: torch.stack([rows[s][key] for s in ids])  # noqa: E731
    return {
        "rewards": stack("total_reward"),
        "mask": stack("token_mask"),
        "turn": stack(ASSISTANT_TURN_ID),
        "thinking": stack(THINKING_MASK),
        "action": stack(ACTION_MASK),
        "invalid": stack(INVALID_TOOL_CALL_MASK),
        "groups": (torch.arange(GROUPS * SIZE) // SIZE).unsqueeze(-1),
    }


PER_TURN = dict(enabled=True, scope="prose_per_turn")


def test_input_fields_include_the_turn_columns_only_for_per_turn_scopes():
    algo = GRPOConfig(
        num_generations_per_prompt=SIZE,
        adv_estimator=AdvEstimatorConfig(positive_length_reweight=PER_TURN),
    )
    cfg = AdvantageStageConfig(
        advantage=AdvantageConfig(),
        algo=algo,
        is_ppo=False,
        policy_logprobs_required=False,
        reference_logprobs_required=False,
        teacher_logprobs_required=False,
        message_level_advantage_penalties_enabled=False,
        shardable=True,
        train_data_dump_dir=None,
        per_turn_length_reweight=algo.adv_estimator.positive_length_reweight,
    )
    fields = cfg.input_fields()
    assert {ASSISTANT_TURN_ID, THINKING_MASK, ACTION_MASK} <= set(fields)
    assert not {ASSISTANT_TURN_ID, THINKING_MASK, ACTION_MASK} & set(
        AdvantageStageConfig(
            **{**cfg.__dict__, "per_turn_length_reweight": None}
        ).input_fields()
    )


def test_stage_applies_the_per_turn_reweight_and_logs_its_stats():
    ctrl, adv = _stage(PER_TURN)
    b = _batch()
    plain = GRPOAdvantageEstimator(AdvEstimatorConfig(), None).compute_advantage(
        prompt_ids=b["groups"], rewards=b["rewards"], mask=b["mask"]
    )
    expected, stats = reweight_positive_text_per_turn(
        plain,
        prompt_ids=b["groups"],
        rewards=b["rewards"],
        mask=b["mask"],
        assistant_turn_id=b["turn"],
        thinking_mask=b["thinking"],
        action_mask=b["action"],
        valid=torch.ones(GROUPS * SIZE, dtype=torch.bool),
        config=AdvEstimatorConfig(
            positive_length_reweight=PER_TURN
        ).positive_length_reweight,
    )
    trained = b["mask"].bool()
    torch.testing.assert_close(adv[trained], expected[trained])
    assert stats.groups_reweighted > 0
    assert ctrl._length_reweight_stats is not None
    assert ctrl._length_reweight_stats.to_metrics() == pytest.approx(stats.to_metrics())
    # Something actually moved.
    assert not torch.equal(adv[trained], plain[trained])


def test_penalties_and_clip_run_after_the_reweight():
    _, adv = _stage(PER_TURN, penalties=True, clip_high=1.2)
    b = _batch()
    flagged = b["invalid"] & b["mask"].bool()
    assert flagged.any()
    assert torch.all(adv[flagged] == -2.0)  # the penalty wins over the reweight
    assert (
        adv[b["mask"].bool()].max() <= 1.2 + 1e-6
    )  # the clip saw the reweighted values


def test_sharded_pool_matches_in_process():
    _, local = _stage(PER_TURN, num_actors=0)
    ctrl, pooled = _stage(PER_TURN, num_actors=2)
    assert sorted(len(a.calls) for a in ctrl._advantage_actors) == [1, 1]
    torch.testing.assert_close(pooled, local)


def test_trajectory_scope_never_reads_the_turn_columns():
    store = _RowStore(_rows())
    for row in store.rows.values():
        for key in (ASSISTANT_TURN_ID, THINKING_MASK, ACTION_MASK):
            del row[key]  # a KeyError here would mean the stage fetched them
    ctrl = _controller(
        store, num_actors=0, reweight=dict(enabled=True), penalties=False
    )
    asyncio.run(ctrl._advantage_stage(_meta()))
    assert ctrl._advantage_stage_config.per_turn_length_reweight is None
