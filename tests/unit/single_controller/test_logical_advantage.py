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

"""Exercise logical-owner GRPO and PPO through the actual SingleController stage."""

import asyncio
import math
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GAEConfig,
    GeneralizedAdvantageEstimator,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.ppo import PPOConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    AdvantageComputer,
    AdvantageStageConfig,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    TokenCaptureConfig,
    validate_forest_objective,
)
from nemo_rl.algorithms.single_controller_utils.debug_writer import DebugWriterConfig
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    DP_VALUE_TRAIN_FIELDS,
    GROUP_ID_TAG,
    ROUTER_EXCLUDED_ROWS,
)
from tests.unit.single_controller.test_advantage_stage import _InlineActor, _RowStore


class _DataPlane:
    """Store real stage inputs/outputs; no estimator or controller replacement."""

    def __init__(self, meta: KVBatchMeta, data: TensorDict):
        self.meta, self.data, self.puts = meta, data, []

    def get_samples(self, *, sample_ids, partition_id, select_fields):
        assert sample_ids == self.meta.sample_ids
        assert partition_id == self.meta.partition_id
        return self.data.select(*select_fields)

    def put_samples(self, *, sample_ids, partition_id, fields):
        assert sample_ids == self.meta.sample_ids
        assert partition_id == self.meta.partition_id
        self.puts.append(fields)
        self.data.update(fields)


def _batch(
    groups: list[tuple[str, list[tuple[float, int]]]], *, padding: bool = False
) -> tuple[KVBatchMeta, TensorDict]:
    tags, ids, rewards = [], [], []
    for group, owners in groups:
        for slot, (reward, count) in enumerate(owners):
            for segment in range(count):
                ids.append(f"{group}_g{slot}_s{segment}")
                rewards.append(reward)
                tags.append(
                    dict(
                        dispatch_group_id=group,
                        logical_rollout_id=f"{group}_g{slot}",
                        logical_slot=slot,
                        logical_group_size=len(owners),
                        segment_index=segment,
                        segment_count=count,
                        is_execution_padding=False,
                    )
                )
    if padding:
        ids.append(f"{groups[0][0]}_pad0")
        rewards.append(1000.0)  # Must never contribute to any owner's baseline.
        tags.append(
            dict(
                dispatch_group_id=groups[0][0],
                logical_rollout_id=None,
                logical_slot=None,
                logical_group_size=len(groups[0][1]),
                segment_index=None,
                segment_count=0,
                is_execution_padding=True,
            )
        )
    n = len(ids)
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.tensor([[17, 42]] * n),
            "total_reward": torch.tensor(rewards),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0]] * n),
            "sample_mask": torch.ones(n),
            "mask_sample": torch.zeros(n, dtype=torch.bool),
            "truncated": torch.zeros(n, dtype=torch.bool),
        },
        batch_size=[n],
    )
    if padding:
        data["token_mask"][-1] = 0
        data["sample_mask"][-1] = 0
    for tag in tags:
        tag[GROUP_ID_TAG] = tag["dispatch_group_id"]
    return KVBatchMeta("train", "train", ids, fields=list(data.keys()), tags=tags), data


@pytest.mark.parametrize("policy", ["returned_history", "all_captured"])
def test_forest_selection_config(policy):
    TokenCaptureConfig(forest_training=True, selection_policy=policy)
    with pytest.raises(ValueError, match="requires forest_training"):
        TokenCaptureConfig(selection_policy="all_captured")
    with pytest.raises(ValueError, match="selection_policy"):
        TokenCaptureConfig(forest_training=True, selection_policy="guess")


@pytest.mark.parametrize(
    "option",
    [
        {"forest_penalty_scope": "all_segments"},
        {"exclude_padding_from_router_stats": True},
        {"forest_padding_scope": "step"},
    ],
)
def test_forest_options_require_forest_training(option):
    assert TokenCaptureConfig(forest_training=True, **option)
    with pytest.raises(ValueError, match=f"{next(iter(option))}.*forest_training"):
        TokenCaptureConfig(**option)


def test_max_rows_per_rollout_requires_continued_chains():
    assert TokenCaptureConfig(
        forest_training=True,
        selection_policy="continued_chains",
        max_rows_per_rollout=16,
    )
    with pytest.raises(ValueError, match="max_rows_per_rollout.*forest_training"):
        TokenCaptureConfig(max_rows_per_rollout=16)
    with pytest.raises(ValueError, match="max_rows_per_rollout.*continued_chains"):
        TokenCaptureConfig(forest_training=True, max_rows_per_rollout=16)


def test_forest_pad_source_requires_step_padding():
    assert TokenCaptureConfig(
        forest_training=True, forest_padding_scope="step", forest_pad_source="cyclic"
    )
    with pytest.raises(ValueError, match="forest_padding_scope=step"):
        TokenCaptureConfig(forest_training=True, forest_pad_source="cyclic")


def _controller(
    meta, data, *, grpo=None, loss=None, policy_logprobs_required: bool = False
):
    settings = dict(
        num_generations_per_prompt=2,
        masked_reward_policy="exclude",
        adv_estimator=AdvEstimatorConfig(
            normalize_rewards=False, use_leave_one_out_baseline=False
        ),
    )
    if any("logical_group_size" in tag for tag in meta.tags or []):
        settings["num_generations_per_prompt"] = next(
            tag["logical_group_size"]
            for tag in meta.tags
            if "logical_group_size" in tag
        )
    else:
        settings["num_generations_per_prompt"] = 3
    config = GRPOConfig(**(settings | (grpo or {})))
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = _DataPlane(meta, data)
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = GRPOAdvantageEstimator(
        config.adv_estimator, ClippedPGLossConfig()
    )
    ctrl._algo_cfg = config
    ctrl._master_config = SimpleNamespace(loss_fn=ClippedPGLossConfig(**(loss or {})))
    ctrl._policy_logprobs_required = policy_logprobs_required
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._message_level_advantage_penalties_enabled = (
        config.invalid_tool_call_advantage is not None
        or config.malformed_thinking_advantage is not None
    )
    ctrl._step_log_dict = defaultdict(list)
    ctrl._advantage_stage_config = AdvantageStageConfig(
        advantage=ctrl._advantage_cfg,
        algo=config,
        is_ppo=False,
        policy_logprobs_required=policy_logprobs_required,
        reference_logprobs_required=False,
        teacher_logprobs_required=False,
        opd_full_enabled=False,
        message_level_advantage_penalties_enabled=ctrl._message_level_advantage_penalties_enabled,
        shardable=True,
        train_data_dump_dir=None,
    )
    ctrl._advantage_computer = AdvantageComputer(
        ctrl._dp_client,
        config=ctrl._advantage_stage_config,
        advantage_estimator=ctrl._advantage_estimator,
    )
    ctrl._advantage_actors = []
    ctrl._train_data_dump = None
    ctrl._train_data_dump_rows = 0
    ctrl._opd_stat_sum = ctrl._opd_stat_sumsq = ctrl._opd_gap_sum = 0.0
    ctrl._opd_stat_count = 0
    return ctrl


@pytest.mark.parametrize("threshold", [None, 2.0])
@pytest.mark.parametrize("bad_rows", [[], [1], [0, 1, 2, 3]])
@pytest.mark.parametrize("reorder", [False, True])
def test_logprob_error_masks_whole_owner(
    threshold: float | None, bad_rows: list[int], reorder: bool
) -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1), (3.0, 1)])], padding=True)
    data["prev_logprobs"] = torch.zeros(5, 3)
    data["generation_logprobs"] = torch.zeros(5, 3)
    data["generation_logprobs"][bad_rows, 2] = -5.0
    # Neither replayed context nor ownerless padding may trigger filtering.
    data["token_mask"][0, 1] = 0
    data["generation_logprobs"][0, 1] = -1000.0
    data["generation_logprobs"][-1] = -1000.0
    if reorder:
        order = [4, 2, 1, 3, 0]
        meta, data = meta.subset(order), data[order]
    ctrl = _controller(
        meta,
        data,
        grpo={"seq_logprob_error_threshold": threshold},
        policy_logprobs_required=True,
    )
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    rejected = set() if threshold is None else ({0} if bad_rows == [1] else set())
    if threshold is not None and len(bad_rows) == 4:
        rejected = {0, 1, 2}
    retained = [slot for slot in range(3) if slot not in rejected]
    rewards = torch.tensor([0.0, 1.0, 3.0])
    baseline = rewards[retained].mean() if retained else 0.0
    for row, tag in enumerate(meta.tags):
        slot = tag["logical_slot"]
        keep = slot is not None and slot in retained
        assert data["sample_mask"][row].item() == float(keep)
        expected = float(rewards[slot] - baseline) if keep else 0.0
        torch.testing.assert_close(data["advantages"][row], torch.full((3,), expected))
    assert valid == bool(retained)
    assert ctrl._step_log_dict["reward_partials"][0].weight == len(retained)


@pytest.mark.parametrize("lengths", [(100,), (99, 1), (40, 40, 20), (20,) * 5])
@pytest.mark.parametrize("reorder", [False, True])
def test_logprob_filtering_pools_owned_tokens_independently_of_segmentation(
    lengths: tuple[int, ...], reorder: bool
) -> None:
    meta, data = _batch(
        [("a", [(0.0, len(lengths)), (1.0, 2), (3.0, 1)])], padding=True
    )
    width = max(lengths) + 2
    counts = [*lengths, 1, 1, 1, 0]
    data["token_mask"] = torch.zeros(len(counts), width)
    data["prev_logprobs"] = torch.zeros(len(counts), width)
    data["generation_logprobs"] = torch.zeros(len(counts), width)
    for row, count in enumerate(counts):
        data["token_mask"][row, 2 : count + 2] = 1
    # Prompts/replayed prefixes and a borrowed padding layout cannot add error.
    data["generation_logprobs"][:, :2] = -1000
    data["generation_logprobs"][-1] = torch.nan
    data["token_mask"][-1] = 1
    # Owner 0: 99 matching tokens and one 1.2-error token, pooled mean 1.002.
    data["generation_logprobs"][len(lengths) - 1, lengths[-1] + 1] = -math.log(1.2)
    # Owner 1: pooled mean 1.1; both of its rows must be filtered.
    data["generation_logprobs"][len(lengths), 2] = -math.log(1.2)
    if reorder:
        order = list(reversed(range(len(counts))))
        meta, data = meta.subset(order), data[order]
    ctrl = _controller(
        meta,
        data,
        grpo={"seq_logprob_error_threshold": 1.05},
        policy_logprobs_required=True,
    )
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    for row, tag in enumerate(meta.tags):
        slot = tag["logical_slot"]
        keep = slot in (0, 2)
        assert data["sample_mask"][row].item() == float(keep)
        expected = {0: -1.5, 2: 1.5}.get(slot, 0.0)
        torch.testing.assert_close(
            data["advantages"][row], torch.full((width,), expected)
        )
    metrics = ctrl._step_log_dict["seq_logprob_error_metrics"][0]
    assert metrics["num_masked_seqs_by_logprob_error"] == 2
    assert math.isfinite(metrics["max_seq_mult_prob_error"])
    if lengths == (99, 1):
        # Diagnostics retain the short segment's error even though its owner passes.
        assert metrics["max_seq_mult_prob_error_after_mask"] == pytest.approx(1.2)
    assert ctrl._step_log_dict["reward_partials"][0].weight == 2


@pytest.mark.parametrize("empty_rows", [[0], [1], [0, 1]])
@pytest.mark.parametrize("policy_logprobs_required", [False, True])
def test_tokenless_segments_do_not_remove_an_owner_with_training_tokens(
    empty_rows: list[int], policy_logprobs_required: bool
) -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1), (3.0, 1)])], padding=True)
    data["token_mask"][empty_rows] = 0
    data["prev_logprobs"] = torch.zeros(5, 3)
    data["generation_logprobs"] = torch.zeros(5, 3)
    ctrl = _controller(meta, data, policy_logprobs_required=policy_logprobs_required)
    keep_owner_zero = len(empty_rows) < 2
    baseline = 4 / 3 if keep_owner_zero else 2.0
    for _ in range(2):
        # Reusing already filtered physical rows must preserve logical validity.
        _, valid = asyncio.run(ctrl._advantage_stage(meta))
        assert valid
        for row, tag in enumerate(meta.tags):
            slot = tag["logical_slot"]
            keep = slot is not None and row not in empty_rows
            assert data["sample_mask"][row].item() == float(keep)
            expected = ([0.0, 1.0, 3.0][slot] - baseline) if keep else 0.0
            torch.testing.assert_close(
                data["advantages"][row], torch.full((3,), expected)
            )
        assert ctrl._step_log_dict["reward_partials"][-1].weight == (
            3 if keep_owner_zero else 2
        )


@pytest.mark.parametrize("bounds", [(-0.2, 0.3), (0.2, 0.4), (-0.4, -0.2)])
@pytest.mark.parametrize("reorder", [False, True])
def test_clips_owner_advantages_once_without_activating_padding(bounds, reorder):
    meta, data = _batch([("a", [(0.0, 3), (1.0, 1)])], padding=True)
    if reorder:
        order = [4, 3, 1, 0, 2]
        meta, data = meta.subset(order), data[order]
    grpo = dict(advantage_clip_low=bounds[0], advantage_clip_high=bounds[1])
    validate_forest_objective(GRPOConfig(**grpo), ClippedPGLossConfig())
    ctrl = _controller(meta, data, grpo=grpo)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    expected = torch.tensor([-0.5, 0.5]).clamp(*bounds)
    for i, tag in enumerate(meta.tags):
        value = 0.0 if tag["is_execution_padding"] else expected[tag["logical_slot"]]
        torch.testing.assert_close(
            data["advantages"][i], torch.full((3,), float(value))
        )
    assert ctrl._step_log_dict["reward_partials"][0].weight == 2


def test_clipping_does_not_reactivate_filtered_logical_owners():
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    data["mask_sample"][0] = True
    ctrl = _controller(meta, data, grpo={"advantage_clip_low": 0.2})
    asyncio.run(ctrl._advantage_stage(meta))
    assert data["advantages"][[0, 1, 3]].count_nonzero() == 0
    assert data["sample_mask"].tolist() == [0, 0, 1, 0]


@pytest.mark.parametrize("reorder", [False, True])
def test_deduplicates_unequal_segments_without_pooling_identical_prompts_across_groups(
    reorder,
):
    meta, data = _batch(
        [("a", [(0.0, 2), (0.0, 1)]), ("b", [(1.0, 3), (1.0, 1)])], padding=True
    )
    if reorder:
        order = [4, 1, 6, 0, 7, 5, 3, 2]
        meta = meta.subset(order)
        data = data[order]
    ctrl = _controller(meta, data)
    result, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid and "advantages" in result.fields
    assert len(ctrl._dp_client.puts) == 1
    reward_partial = ctrl._step_log_dict["reward_partials"][0]
    assert reward_partial.weighted_total == 2
    assert reward_partial.weight == 4
    assert data["advantages"].count_nonzero() == 0


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize(
    "flagged_rows,expected_count",
    [((), 0), ((0,), 1), ((2,), 1), ((0, 1, 2), 1), ((0, 3), 2)],
)
def test_filter_metric_counts_owners_not_segments_or_padding(
    padding: bool, flagged_rows: tuple[int, ...], expected_count: int
) -> None:
    meta, data = _batch([("a", [(0.0, 3), (1.0, 1)])], padding=padding)
    for row in flagged_rows:
        data["mask_sample"][row] = True
    if padding:
        # Match the finalizer's environment-mask field on execution padding.
        data["mask_sample"][-1] = True
    ctrl = _controller(meta, data)
    asyncio.run(ctrl._advantage_stage(meta))
    assert ctrl._step_log_dict["num_mask_sample_filtered"] == [expected_count]
    # Per-row reason counts would count segments and padding, so none are kept.
    assert ctrl._step_log_dict["sample_mask_metrics"] == []
    assert (
        data["sample_mask"][:3].tolist()
        == [float(not any(row < 3 for row in flagged_rows))] * 3
    )
    assert data["sample_mask"][3].item() == float(3 not in flagged_rows)
    if padding:
        assert data["sample_mask"][-1].item() == 0


@pytest.mark.parametrize("mask_owner", [False, True])
def test_logical_advantages_match_through_sharded_stage(mask_owner: bool) -> None:
    meta, data = _batch([("a", [(0.0, 3), (2.0, 1)]), ("b", [(1.0, 1), (5.0, 4)])])
    if mask_owner:
        data["mask_sample"][-1] = True
    outputs = []
    for actor_count in (0, 2):
        ctrl = _controller(meta, data)
        store = _RowStore(
            {
                sid: {name: data[name][i].clone() for name in data.keys()}
                for i, sid in enumerate(meta.sample_ids)
            }
        )
        computer = AdvantageComputer(
            store,
            config=ctrl._advantage_stage_config,
            advantage_estimator=ctrl._advantage_estimator,
        )
        ctrl._advantage_computer = computer
        ctrl._advantage_actors = [_InlineActor(computer) for _ in range(actor_count)]
        ctrl._available_advantage_actors = asyncio.Queue()
        for actor in ctrl._advantage_actors:
            ctrl._available_advantage_actors.put_nowait(actor)
        _, valid = asyncio.run(ctrl._advantage_stage(meta))
        assert valid
        assert sum(ctrl._step_log_dict["num_mask_sample_filtered"]) == int(mask_owner)
        if actor_count:
            assert sorted(
                n for actor in ctrl._advantage_actors for n in actor.calls
            ) == [4, 5]
        partials = ctrl._step_log_dict["reward_partials"]
        assert sum(p.weight for p in partials) == (3 if mask_owner else 4)
        assert sum(p.weighted_total for p in partials) == (3 if mask_owner else 8)
        outputs.append(
            store.get_samples(
                sample_ids=meta.sample_ids, select_fields=["advantages", "sample_mask"]
            )
        )
    for field in ("advantages", "sample_mask"):
        torch.testing.assert_close(outputs[0][field], outputs[1][field], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0]["advantages"][:3], torch.full((3, 3), -1.0))
    torch.testing.assert_close(outputs[0]["advantages"][3], torch.ones(3))
    if mask_owner:
        assert outputs[0]["sample_mask"][5:].count_nonzero() == 0


def test_dispatch_identity_must_match_sharding_identity() -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])])
    for tag in meta.tags:
        tag[GROUP_ID_TAG] = "foreign"
    ctrl = _controller(meta, data)
    with pytest.raises(ValueError, match="Forest dispatch identity"):
        asyncio.run(ctrl._advantage_stage(meta))
    assert not ctrl._dp_client.puts


@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("leave_one_out", [False, True])
def test_sibling_rollouts_with_distinct_prompts_share_baseline(
    reorder: bool, leave_one_out: bool
) -> None:
    meta, data = _batch([("a", [(0.0, 3), (1.0, 2)])], padding=True)
    data["prompt_ids_for_adv"][3:5] = torch.tensor([42, 17])
    if reorder:
        order = [4, 1, 5, 0, 3, 2]
        meta, data = meta.subset(order), data[order]
    ctrl = _controller(
        meta,
        data,
        grpo={
            "adv_estimator": {
                "normalize_rewards": False,
                "use_leave_one_out_baseline": leave_one_out,
            },
        },
    )
    asyncio.run(ctrl._advantage_stage(meta))
    magnitude = 1.0 if leave_one_out else 0.5
    for row, tag in enumerate(meta.tags):
        expected = (
            0.0
            if tag["is_execution_padding"]
            else (-magnitude if tag["logical_slot"] == 0 else magnitude)
        )
        torch.testing.assert_close(data["advantages"][row], torch.full((3,), expected))


@pytest.mark.nemo_gym
def test_real_capture_finalizer_rows_feed_sc_at_actual_segment_lengths(
    monkeypatch, tmp_path
):
    # Keep optional Gym integration out of ordinary controller test imports.
    pytest.importorskip("nemo_gym", reason="requires the nemo_gym extra")
    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler
    from tests.unit.experience.test_logical_owner_finalization import (
        PublicationDataPlane,
        capture_segment,
        decide_capture_input,
        finalize,
        gym_harness,
    )

    data_plane = PublicationDataPlane()
    source = TQTokenSource(data_plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(data_plane, staging_partition="staged"),
        fetch_prefix=lambda keys: source.fetch_prefix_chains(keys).expanded,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    try:
        owners = [
            [
                capture_segment(harness, "group_g0", child=True),
                capture_segment(harness, "group_g0"),
            ],
            [capture_segment(harness, "group_g1")],
        ]
        finalizer = RolloutReassembler(
            data_plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
        )
        result = finalize(finalizer, owners)
        meta = result.meta
        assert meta is not None and not result.dropped
        assert meta.sequence_lengths == [6, 3, 3]
        data = data_plane.get_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=meta.fields,
        )
        ctrl = _controller(meta, data)
        _, valid = asyncio.run(ctrl._advantage_stage(meta))
        assert valid and data["advantages"].is_nested
        for row, expected, length in zip(
            data["advantages"].unbind(), [-0.5, -0.5, 0.5], meta.sequence_lengths
        ):
            torch.testing.assert_close(row, torch.full((length,), expected))
        assert data["sample_mask"].tolist() == [1, 1, 1]
        for token_mask in data["token_mask"].unbind():
            assert token_mask[:2].count_nonzero() == 0
            assert token_mask[2:].count_nonzero() > 0
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())


def test_sibling_prompt_lengths_do_not_split_the_sampling_group():
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])])
    data["prompt_ids_for_adv"] = torch.nested.as_nested_tensor(
        [torch.tensor([17, 42])] * 2 + [torch.tensor([17, 42, 0])],
        layout=torch.jagged,
    )
    ctrl = _controller(meta, data)
    asyncio.run(ctrl._advantage_stage(meta))
    torch.testing.assert_close(
        data["advantages"][:, 0], torch.tensor([-0.5, -0.5, 0.5])
    )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("leave_one_out", [False, True])
def test_preserves_existing_grpo_normalization_and_leave_one_out(
    normalize, leave_one_out
):
    meta, data = _batch([("a", [(0.0, 2), (1.0, 3), (3.0, 1)])])
    ctrl = _controller(
        meta,
        data,
        grpo={
            "adv_estimator": {
                "normalize_rewards": normalize,
                "use_leave_one_out_baseline": leave_one_out,
            }
        },
    )
    expected = ctrl._advantage_estimator.compute_advantage(
        prompt_ids=torch.tensor([[17, 42]] * 3),
        rewards=torch.tensor([0.0, 1.0, 3.0]),
        mask=torch.ones(3, 3),
        valid_mask=torch.ones(3),
    )
    asyncio.run(ctrl._advantage_stage(meta))
    torch.testing.assert_close(data["advantages"], expected[[0, 0, 1, 1, 1, 2]])


@pytest.mark.parametrize("filtering", [False, True])
def test_overlong_filter_flag_is_respected_for_all_owner_segments(filtering):
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])])
    data["truncated"][1] = True
    ctrl = _controller(meta, data, grpo={"overlong_filtering": filtering})
    asyncio.run(ctrl._advantage_stage(meta))
    assert data["sample_mask"].tolist() == ([0, 0, 1] if filtering else [1, 1, 1])


@pytest.mark.parametrize("filter_field", ["sample_mask", "mask_sample", "truncated"])
@pytest.mark.parametrize("ordinary", [False, True])
def test_masked_owner_is_excluded_from_baseline_and_loss(filter_field, ordinary):
    expected = [-1.0, 1.0]
    counts = [1, 1, 1] if ordinary else [2, 1, 2]
    meta, data = _batch([("a", list(zip([0.0, 2.0, 10.0], counts)))])
    if ordinary:
        meta.tags = [{GROUP_ID_TAG: "a"} for _ in meta.sample_ids]
    data[filter_field][-1] = 0 if filter_field == "sample_mask" else 1
    ctrl = _controller(meta, data, grpo={"overlong_filtering": True})
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    assert ctrl._step_log_dict["num_mask_sample_filtered"] == [
        int(filter_field == "mask_sample")
    ]
    torch.testing.assert_close(data["advantages"][0], torch.full((3,), expected[0]))
    torch.testing.assert_close(
        data["advantages"][counts[0]], torch.full((3,), expected[1])
    )
    assert data["sample_mask"][-counts[-1] :].count_nonzero() == 0
    assert (data["token_mask"] * data["sample_mask"].unsqueeze(-1))[
        -counts[-1] :
    ].count_nonzero() == 0


@pytest.mark.parametrize("policy", ["include", "exclude"])
@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize(
    "filter_field", ["mask_sample", "truncated", "logprob_error", "placeholder"]
)
def test_masked_reward_policy_counts_each_rollout_once(policy, ordinary, filter_field):
    counts = [1, 1, 1] if ordinary else [2, 1, 3]
    meta, data = _batch([("a", list(zip([0.0, 2.0, 10.0], counts)))], padding=True)
    if ordinary:
        meta = meta.subset(range(3))
        data = data[:3]
        meta.tags = [{GROUP_ID_TAG: "a"} for _ in meta.sample_ids]
    last = sum(counts) - 1
    if filter_field == "placeholder":
        data["sample_mask"][sum(counts[:2]) : last + 1] = 0
        data["token_mask"][sum(counts[:2]) : last + 1] = 0
    elif filter_field != "logprob_error":
        data[filter_field][last] = True
    data["prev_logprobs"] = torch.zeros_like(data["token_mask"])
    data["generation_logprobs"] = torch.zeros_like(data["token_mask"])
    if filter_field == "logprob_error":
        data["generation_logprobs"][last, 2] = -5
    ctrl = _controller(
        meta,
        data,
        grpo={
            "masked_reward_policy": policy,
            "overlong_filtering": True,
            "seq_logprob_error_threshold": 1.05,
        },
        policy_logprobs_required=True,
    )
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    baseline = 4.0 if policy == "include" and filter_field != "placeholder" else 1.0
    torch.testing.assert_close(data["advantages"][0], torch.full((3,), -baseline))
    torch.testing.assert_close(
        data["advantages"][counts[0]], torch.full((3,), 2 - baseline)
    )
    assert data["sample_mask"][sum(counts[:2]) :].count_nonzero() == 0
    effective_advantages = data["advantages"] * data["sample_mask"].unsqueeze(-1)
    assert effective_advantages[sum(counts[:2]) :].count_nonzero() == 0


def test_all_invalid_owners_take_existing_no_training_path():
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    data["mask_sample"][:] = True
    ctrl = _controller(meta, data)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert not valid
    assert data["advantages"].count_nonzero() == 0
    assert data["sample_mask"].count_nonzero() == 0


def test_failed_owner_single_placeholder_is_excluded_from_baseline_and_loss():
    meta, data = _batch([("a", [(0.0, 2), (2.0, 1), (10.0, 1)])])
    data["sample_mask"][-1] = 0
    data["token_mask"][-1] = 0
    ctrl = _controller(meta, data)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    assert data["advantages"][0, 1].item() == -1.0
    assert data["sample_mask"][-1].item() == 0
    assert data["token_mask"][-1].count_nonzero() == 0


@pytest.mark.parametrize(
    "bad_case",
    [
        "missing_tag",
        "mixed",
        "missing_owner",
        "missing_segment",
        "duplicate_segment",
        "wrong_owner",
        "wrong_row_id",
        "wrong_group_size",
        "nonfinite",
        "reward_disagreement",
        "prompt_disagreement",
        "prompt_length_disagreement",
        "padding_owner",
        "padding_mask",
        "nonbinary_mask",
        "duplicate_row_id",
        "boolean_segment",
        "boolean_group_size",
        "padding_only",
    ],
)
def test_rejects_incomplete_or_inconsistent_inputs_before_writing(bad_case):
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    if bad_case == "missing_tag":
        del meta.tags[0]["segment_count"]
    elif bad_case == "mixed":
        meta.tags[0] = {GROUP_ID_TAG: "a"}
    elif bad_case in {"missing_owner", "missing_segment"}:
        keep = [0, 1, 3] if bad_case == "missing_owner" else [0, 2, 3]
        meta, data = meta.subset(keep), data[keep]
    elif bad_case == "duplicate_segment":
        meta.tags[1]["segment_index"] = 0
    elif bad_case == "wrong_owner":
        meta.tags[1]["logical_rollout_id"] = "foreign_g0"
    elif bad_case == "wrong_row_id":
        meta.sample_ids[0] = "foreign_g0_s0"
    elif bad_case == "wrong_group_size":
        meta.tags[1]["logical_group_size"] = 3
    elif bad_case == "nonfinite":
        data["total_reward"][0] = float("nan")
    elif bad_case == "reward_disagreement":
        data["total_reward"][1] = 99
    elif bad_case == "prompt_disagreement":
        data["prompt_ids_for_adv"][1, 0] = 99
    elif bad_case == "prompt_length_disagreement":
        data["prompt_ids_for_adv"] = torch.nested.as_nested_tensor(
            [
                torch.tensor([17, 42]),
                torch.tensor([17, 42, 0]),
                torch.tensor([17, 42]),
                torch.tensor([17, 42]),
            ],
            layout=torch.jagged,
        )
    elif bad_case == "padding_owner":
        meta.tags[-1]["logical_rollout_id"] = "a_g0"
    elif bad_case == "padding_mask":
        data["sample_mask"][-1] = 1
    elif bad_case == "nonbinary_mask":
        data["sample_mask"][0] = 0.5
    elif bad_case == "duplicate_row_id":
        meta.sample_ids[1] = meta.sample_ids[0]
    elif bad_case == "boolean_segment":
        meta.tags[0]["segment_index"] = False
    elif bad_case == "boolean_group_size":
        meta.tags[1]["logical_group_size"] = True
    elif bad_case == "padding_only":
        meta, data = meta.subset([3]), data[[3]]
    ctrl = _controller(meta, data)
    with pytest.raises(ValueError, match="Forest"):
        asyncio.run(ctrl._advantage_stage(meta))
    assert not ctrl._dp_client.puts


@pytest.mark.parametrize(
    "grpo,loss",
    [
        ({"adv_estimator": {"name": "gdpo"}}, {}),
        ({"use_dynamic_sampling": True}, {}),
        ({"reward_scaling": {"enabled": True}}, {}),
        ({"reward_shaping": {"enabled": True}}, {}),
        ({"calculate_advantages_on_gpu": True}, {}),
        ({"invalid_tool_call_advantage": -5.0}, {}),
        ({"malformed_thinking_advantage": 0.0}, {}),
        ({}, {"token_level_loss": False}),
        ({}, {"sequence_level_importance_ratios": True}),
        ({}, {"truncated_importance_sampling_type": "seq-mask-tis"}),
        ({}, {"use_kl_in_reward": True}),
        ({}, {"positive_example_nll_weight": 1}),
    ],
)
def test_unsupported_forest_estimator_or_objective_is_rejected(grpo, loss):
    setting = ("grpo." + next(iter(grpo))) if grpo else ("loss_fn." + next(iter(loss)))
    with pytest.raises(ValueError, match=setting):
        validate_forest_objective(GRPOConfig(**grpo), ClippedPGLossConfig(**loss))
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])])
    ctrl = _controller(meta, data, grpo=grpo, loss=loss)
    with pytest.raises(ValueError, match="Forest"):
        asyncio.run(ctrl._advantage_stage(meta))
    assert not ctrl._dp_client.puts


def _ppo_controller(meta, data, **gae):
    """The GRPO harness above, switched to PPO with per-row GAE."""
    ctrl = _controller(meta, data)
    config = PPOConfig(
        num_generations_per_prompt=ctrl._algo_cfg.num_generations_per_prompt,
        adv_estimator=GAEConfig(gae_lambda=1.0, gae_gamma=1.0, **gae),
    )
    ctrl._algo_cfg = config
    ctrl._is_ppo = True
    ctrl._advantage_estimator = GeneralizedAdvantageEstimator(
        config.adv_estimator, ClippedPGLossConfig()
    )
    ctrl._advantage_stage_config = AdvantageStageConfig(
        advantage=ctrl._advantage_cfg,
        algo=config,
        is_ppo=True,
        policy_logprobs_required=False,
        reference_logprobs_required=False,
        teacher_logprobs_required=False,
        message_level_advantage_penalties_enabled=False,
        opd_full_enabled=False,
        shardable=False,
        train_data_dump_dir=None,
    )
    ctrl._advantage_computer = AdvantageComputer(
        ctrl._dp_client,
        config=ctrl._advantage_stage_config,
        advantage_estimator=ctrl._advantage_estimator,
    )
    return ctrl


def _with_values(data: TensorDict, seed: int = 0) -> TensorDict:
    generator = torch.Generator().manual_seed(seed)
    data["values"] = torch.rand(data.batch_size[0], 3, generator=generator)
    return data


@pytest.mark.parametrize("reorder", [False, True])
def test_forest_ppo_runs_gae_per_row_and_masks_whole_owner(reorder: bool) -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1), (3.0, 2)])], padding=True)
    data = _with_values(data)
    # A flagged segment filters every row of its owner (slot 2).
    data["mask_sample"][3] = True
    if reorder:
        order = [5, 3, 0, 4, 2, 1]
        meta, data = meta.subset(order), data[order]
    inputs = data.clone()
    ctrl = _ppo_controller(meta, data)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    keep = torch.tensor([float(tag["logical_slot"] in (0, 1)) for tag in meta.tags])
    torch.testing.assert_close(data["sample_mask"], keep)
    expected, returns = ctrl._advantage_estimator.compute_advantage(
        prompt_ids=None,
        rewards=inputs["total_reward"].float(),
        mask=inputs["token_mask"] * keep.unsqueeze(-1),
        values=inputs["values"],
    )
    torch.testing.assert_close(data["advantages"], expected)
    torch.testing.assert_close(data["returns"], returns)
    # Segments of one owner keep their own values, so their advantages differ.
    first = [i for i, tag in enumerate(meta.tags) if tag["logical_slot"] == 0]
    assert not torch.equal(data["advantages"][first[0]], data["advantages"][first[1]])
    padding = [i for i, tag in enumerate(meta.tags) if tag["is_execution_padding"]]
    assert data["advantages"][padding].count_nonzero() == 0
    # Owner-level reward metrics: two retained owners, padding never votes.
    assert ctrl._step_log_dict["reward_partials"][0].weight == 2


def test_forest_ppo_normalize_over_all_rows_excludes_execution_padding() -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    data = _with_values(data, seed=1)
    data["mask_sample"][2] = True  # slot 1 trains nothing but still whitens.
    # Live padding tokens and a huge reward would dominate any statistic.
    data["token_mask"][-1] = torch.tensor([0.0, 1.0, 1.0])
    inputs = data.clone()
    results = {}
    for normalize_over in ("kept_rows", "all_rows"):
        data = inputs.clone()
        ctrl = _ppo_controller(meta, data, normalize_over=normalize_over)
        asyncio.run(ctrl._advantage_stage(meta))
        results[normalize_over] = data["advantages"].clone()
    keep = torch.tensor([1.0, 1.0, 0.0, 0.0])
    real = torch.tensor([1.0, 1.0, 1.0, 0.0]).unsqueeze(-1)
    estimator = ctrl._advantage_estimator
    expected, _ = estimator.compute_advantage(
        prompt_ids=None,
        rewards=inputs["total_reward"].float(),
        mask=inputs["token_mask"] * real,
        values=inputs["values"],
    )
    expected = expected * inputs["token_mask"] * keep.unsqueeze(-1)
    torch.testing.assert_close(results["all_rows"], expected)
    assert not torch.allclose(results["all_rows"], results["kept_rows"])
    assert results["all_rows"][2:].count_nonzero() == 0


def test_ppo_keeps_gae_advantages_for_weighted_rows() -> None:
    """Dataset loss weights reach GAE through the mask; the stage must not reapply them."""
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.tensor([[17, 42]] * 2),
            "total_reward": torch.tensor([2.0, 6.0]),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0]] * 2),
            "sample_mask": torch.tensor([0.5, 1.0]),
            "mask_sample": torch.zeros(2, dtype=torch.bool),
            "truncated": torch.zeros(2, dtype=torch.bool),
            "values": torch.zeros(2, 3),
        },
        batch_size=[2],
    )
    meta = KVBatchMeta(
        "train",
        "train",
        ["r0", "r1"],
        fields=list(data.keys()),
        tags=[{GROUP_ID_TAG: "a"}, {GROUP_ID_TAG: "a"}],
    )
    inputs = data.clone()
    ctrl = _ppo_controller(meta, data, normalize_advantages=False)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    expected, returns = ctrl._advantage_estimator.compute_advantage(
        prompt_ids=None,
        rewards=inputs["total_reward"],
        mask=inputs["token_mask"] * inputs["sample_mask"].unsqueeze(-1),
        values=inputs["values"],
    )
    torch.testing.assert_close(data["advantages"], expected)
    torch.testing.assert_close(data["returns"], returns)


def test_gae_whitening_padding_requires_all_rows() -> None:
    assert GAEConfig(normalize_over="all_rows", whiten_execution_padding=True)
    with pytest.raises(ValueError, match="normalize_over=all_rows"):
        GAEConfig(whiten_execution_padding=True)


def test_gae_all_rows_requires_whitening() -> None:
    with pytest.raises(ValueError, match="normalize_advantages=true"):
        GAEConfig(normalize_over="all_rows", normalize_advantages=False)


@pytest.mark.parametrize("whiten_padding", [False, True])
def test_raw_reward_rejects_gae_population_options(whiten_padding) -> None:
    with pytest.raises(ValueError, match="require name=gae"):
        GAEConfig(
            name="raw_reward",
            normalize_over="all_rows",
            whiten_execution_padding=whiten_padding,
        )
    assert GAEConfig(name="raw_reward")
    assert GAEConfig(normalize_over="all_rows", whiten_execution_padding=whiten_padding)


def _step_padded(source: str, multiple: int = 4):
    """Five real rows in two groups, padded once for the whole batch."""
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)]), ("b", [(2.0, 1), (3.0, 1)])])
    data = _with_values(data, seed=2)
    data[ROUTER_EXCLUDED_ROWS] = torch.zeros(meta.size, dtype=torch.bool)
    meta = KVBatchMeta(
        "train", "train", meta.sample_ids, fields=list(data.keys()), tags=meta.tags
    )
    store = _RowStore(
        {
            sid: {name: data[name][row].clone() for name in data.keys()}
            for row, sid in enumerate(meta.sample_ids)
        }
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = store
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._forest_step_row_multiple = multiple
    ctrl._master_config = SimpleNamespace(
        token_capture=TokenCaptureConfig(
            forest_training=True, forest_padding_scope="step", forest_pad_source=source
        )
    )
    padded, pad_meta = asyncio.run(ctrl._pad_forest_step_rows(meta))
    return meta, padded, pad_meta, store


@pytest.mark.parametrize("source,copied", [("row0", [0, 0, 0]), ("cyclic", [0, 1, 2])])
def test_forest_step_padding_copies_rows_once_per_batch(source, copied) -> None:
    meta, padded, pad_meta, store = _step_padded(source)
    assert padded.size == 8 and pad_meta.size == 3
    assert padded.sample_ids == meta.sample_ids + pad_meta.sample_ids
    for pad_id, row in zip(pad_meta.sample_ids, copied, strict=True):
        pad, real = store.rows[pad_id], store.rows[meta.sample_ids[row]]
        torch.testing.assert_close(pad["token_mask"], real["token_mask"])
        torch.testing.assert_close(pad["values"], real["values"])
        assert pad["sample_mask"] == 0 and pad[ROUTER_EXCLUDED_ROWS]
    for tag, row in zip(pad_meta.tags, copied, strict=True):
        assert tag["is_execution_padding"] and tag["uses_borrowed_input"]
        assert tag["dispatch_group_id"] == meta.tags[row]["dispatch_group_id"]
        assert tag["logical_rollout_id"] is None and tag["segment_count"] == 0
    # A batch already at the multiple is left alone.
    _, unpadded, none, _ = _step_padded(source, multiple=5)
    assert none is None and unpadded.size == 5


@pytest.mark.parametrize("whiten_padding", [False, True])
def test_forest_step_padding_whitening_counts_padding_copies(whiten_padding) -> None:
    _, padded, _, store = _step_padded("row0")
    data = store.get_samples(sample_ids=padded.sample_ids, select_fields=padded.fields)
    inputs = data.clone()
    ctrl = _ppo_controller(
        padded,
        data,
        normalize_over="all_rows",
        whiten_execution_padding=whiten_padding,
    )
    _, valid = asyncio.run(ctrl._advantage_stage(padded))
    assert valid
    real = torch.tensor([1.0] * 5 + [0.0] * 3)
    # With whiten_execution_padding the row-0 copies join whitening; padding
    # never trains.
    expected, _ = ctrl._advantage_estimator.compute_advantage(
        prompt_ids=None,
        rewards=inputs["total_reward"].float(),
        mask=inputs["token_mask"]
        * (torch.ones(8) if whiten_padding else real).unsqueeze(-1),
        values=inputs["values"],
    )
    expected = expected * inputs["token_mask"] * real.unsqueeze(-1)
    torch.testing.assert_close(data["advantages"], expected)
    assert data["advantages"][5:].count_nonzero() == 0


def test_forest_objective_accepts_ppo_gae_only() -> None:
    validate_forest_objective(PPOConfig(), ClippedPGLossConfig())
    with pytest.raises(ValueError, match="ppo.adv_estimator.name"):
        validate_forest_objective(
            PPOConfig(adv_estimator={"name": "raw_reward"}), ClippedPGLossConfig()
        )
    with pytest.raises(ValueError, match="ppo.use_dynamic_sampling"):
        validate_forest_objective(
            PPOConfig(use_dynamic_sampling=True), ClippedPGLossConfig()
        )


@pytest.mark.parametrize("exclude_router_rows", [False, True])
@pytest.mark.parametrize("forest", [False, True])
def test_critic_takes_every_forest_row_as_one_step(
    forest: bool, exclude_router_rows: bool
) -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    if not forest:
        tags = [{GROUP_ID_TAG: "a"} for _ in meta.sample_ids]
        meta = KVBatchMeta(
            "train", "train", meta.sample_ids, fields=list(data.keys()), tags=tags
        )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._value = MagicMock()
    ctrl._value.train_from_meta.return_value = {}
    ctrl._value_loss_fn = object()
    ctrl._algo_cfg = SimpleNamespace(value_training_mode="whole_batch")
    ctrl._router_excluded_fields = (
        (ROUTER_EXCLUDED_ROWS,) if forest and exclude_router_rows else ()
    )
    asyncio.run(ctrl._value_train_epochs([meta], num_epochs=2))
    calls = ctrl._value.train_from_meta.call_args_list
    assert len(calls) == 2
    assert {call.kwargs["gbs"] for call in calls} == {meta.size if forest else None}
    # Critics fetch the rows their routers must not count only when opted in.
    assert {call.kwargs["train_fields"] for call in calls} == {
        (*DP_VALUE_TRAIN_FIELDS, ROUTER_EXCLUDED_ROWS)
        if forest and exclude_router_rows
        else DP_VALUE_TRAIN_FIELDS
    }


@pytest.mark.parametrize("filtering", [False, True])
def test_forest_ppo_overlong_filter_masks_whole_owner(filtering: bool) -> None:
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    data = _with_values(data, seed=3)
    data["truncated"][1] = True  # Only slot 0's second segment.
    inputs = data.clone()
    ctrl = _ppo_controller(meta, data)
    # The stage config holds this same PPOConfig object.
    ctrl._algo_cfg.overlong_filtering = filtering
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    keep = torch.tensor([0.0, 0.0, 1.0, 0.0] if filtering else [1.0, 1.0, 1.0, 0.0])
    torch.testing.assert_close(data["sample_mask"], keep)
    mask = inputs["token_mask"] * keep.unsqueeze(-1)
    expected, returns = ctrl._advantage_estimator.compute_advantage(
        prompt_ids=None,
        rewards=inputs["total_reward"].float(),
        mask=mask,
        values=inputs["values"],
    )
    torch.testing.assert_close(data["advantages"], expected * mask)
    torch.testing.assert_close(data["returns"], returns)
    if filtering:
        assert data["advantages"][:2].count_nonzero() == 0
        assert ctrl._step_log_dict["reward_partials"][0].weight == 1


@pytest.mark.parametrize("failure", ["mask_sample", "placeholder"])
def test_forest_ppo_all_invalid_owners_report_no_training_tokens(failure) -> None:
    """The stage writes zeros; the pump then refuses the step (no optimizer step)."""
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    data = _with_values(data)
    if failure == "mask_sample":
        data["mask_sample"][:] = True
    else:
        # Every owner is a failed-owner placeholder: no loss rows, no tokens.
        data["sample_mask"][:] = 0
        data["token_mask"][:] = 0
    ctrl = _ppo_controller(meta, data)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert not valid
    assert data["advantages"].count_nonzero() == 0
    assert data["returns"].count_nonzero() == 0
    assert data["sample_mask"].count_nonzero() == 0
    assert ctrl._step_log_dict["reward_partials"][0].weight == 0


def test_gae_advantage_stage_is_never_sharded() -> None:
    """GAE whitens over the whole batch, so a pool runs it as one call."""
    master_config = SimpleNamespace(
        grpo=None,
        ppo=PPOConfig(),
        loss_fn=ClippedPGLossConfig(),
        on_policy_distillation=None,
        policy={"generation": {"top_k": None, "top_p": 1.0}},
        async_rl=SimpleNamespace(
            log_full_train_data=False, debug_writer=DebugWriterConfig()
        ),
        token_capture=TokenCaptureConfig(forest_training=True),
    )
    stage_config = AdvantageStageConfig.from_master_config(master_config)
    assert stage_config.is_ppo and not stage_config.shardable
    meta, data = _batch([("a", [(0.0, 3), (2.0, 1)]), ("b", [(1.0, 1), (5.0, 4)])])
    ctrl = _ppo_controller(meta, _with_values(data))
    actors = [_InlineActor(ctrl._advantage_computer) for _ in range(2)]
    ctrl._advantage_actors = actors
    ctrl._available_advantage_actors = asyncio.Queue()
    for actor in actors:
        ctrl._available_advantage_actors.put_nowait(actor)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    # Two prompt groups and two actors, yet one call over every row.
    assert sorted(n for actor in actors for n in actor.calls) == [meta.size]
