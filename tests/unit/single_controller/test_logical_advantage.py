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

"""Exercise logical-owner GRPO through the actual SingleController stage."""

import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
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
from nemo_rl.algorithms.single_controller_utils.debug_writer import JsonDebugWriter
from nemo_rl.algorithms.single_controller_utils.logical_advantage import (
    build_logical_owner_batch,
    rollout_loss_step_metrics,
    rollout_token_mean_weights,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import GROUP_ID_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import Timer
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


def _controller(
    meta,
    data,
    *,
    grpo=None,
    loss=None,
    policy_logprobs_required: bool = False,
    forest_loss_normalization: str = "global_token_mean",
    train_data_dump_dir: str | None = None,
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
        train_data_dump_dir=train_data_dump_dir,
        forest_loss_normalization=forest_loss_normalization,
    )
    ctrl._advantage_computer = AdvantageComputer(
        ctrl._dp_client,
        config=ctrl._advantage_stage_config,
        advantage_estimator=ctrl._advantage_estimator,
    )
    ctrl._advantage_actors = []
    ctrl._train_data_dump = (
        None if train_data_dump_dir is None else JsonDebugWriter(train_data_dump_dir)
    )
    ctrl._train_data_dump_receipts = []
    ctrl._train_steps = 0
    ctrl._timer = Timer()
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


TOKEN, ROLLOUT = "global_token_mean", "rollout_token_mean"


def _ragged_batch(
    groups: list[tuple[str, list[tuple[float, int]]]], lengths: list[int]
) -> tuple[KVBatchMeta, TensorDict]:
    """``_batch`` with padding, where row ``i`` trains ``lengths[i]`` tokens."""
    meta, data = _batch(groups, padding=True)
    data["token_mask"] = torch.zeros(meta.size, max(lengths) + 1)
    for row, length in enumerate(lengths):
        data["token_mask"][row, 1 : length + 1] = 1
    return meta, data


def _owner_mass(meta: KVBatchMeta, data: TensorDict) -> dict[str | None, float]:
    """Loss-mask mass per logical rollout, as the workers count it."""
    row_mass = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum(-1)
    mass: dict[str | None, float] = defaultdict(float)
    for row, tag in enumerate(meta.tags):
        mass[tag["logical_rollout_id"]] += row_mass[row].item()
    return dict(mass)


def test_rollout_normalization_requires_forest_training():
    TokenCaptureConfig(
        forest_training=True, forest_loss_normalization="rollout_token_mean"
    )
    with pytest.raises(ValueError, match="requires forest_training"):
        TokenCaptureConfig(forest_loss_normalization="rollout_token_mean")


@pytest.mark.parametrize("reorder", [False, True])
def test_rollout_normalization_gives_each_valid_rollout_unit_mass(reorder: bool):
    groups = [("a", [(0.0, 3), (1.0, 1), (3.0, 2)])]
    meta, data = _ragged_batch(groups, [2, 5, 1, 4, 3, 3, 0])
    # One flagged segment masks its whole rollout.
    data["mask_sample"][4] = True
    if reorder:
        order = [6, 2, 4, 0, 5, 1, 3]
        meta, data = meta.subset(order), data[order]
    token_data = data.clone()
    asyncio.run(_controller(meta, token_data)._advantage_stage(meta))
    ctrl = _controller(meta, data, forest_loss_normalization=ROLLOUT)
    # A restored unfinished step sends already-weighted rows through again.
    for _ in range(2):
        _, valid = asyncio.run(ctrl._advantage_stage(meta))
        assert valid
        mass = _owner_mass(meta, data)
        assert mass["a_g0"] == pytest.approx(1.0)
        assert mass["a_g1"] == pytest.approx(1.0)
        assert mass["a_g2"] == 0 and mass[None] == 0
        torch.testing.assert_close(
            data["advantages"], token_data["advantages"], rtol=0, atol=0
        )
    # Token-mode masks are binary: zero for a_g2 and padding.
    owner_tokens = {"a_g0": 8.0, "a_g1": 4.0, "a_g2": 6.0, None: 1.0}
    for row, tag in enumerate(meta.tags):
        expected = (
            token_data["sample_mask"][row] / owner_tokens[tag["logical_rollout_id"]]
        )
        assert data["sample_mask"][row].item() == pytest.approx(expected.item())


@pytest.mark.parametrize("segments", [1, 3])
@pytest.mark.parametrize("rollout", [False, True])
def test_extra_segments_do_not_grow_a_rollouts_weight(segments: int, rollout: bool):
    meta, data = _ragged_batch(
        [("a", [(0.0, segments), (1.0, 1)])], [2] * segments + [3, 0]
    )
    ctrl = _controller(
        meta, data, forest_loss_normalization=ROLLOUT if rollout else TOKEN
    )
    asyncio.run(ctrl._advantage_stage(meta))
    mass = _owner_mass(meta, data)
    share = mass["a_g0"] / (mass["a_g0"] + mass["a_g1"])
    # Token normalization weights the branched rollout by its 2 * segments
    # tokens; rollout normalization gives both rollouts half the step.
    expected = 0.5 if rollout else 2 * segments / (2 * segments + 3)
    assert share == pytest.approx(expected)


def test_rollout_token_mean_weights_are_one_over_rollout_tokens():
    meta, data = _ragged_batch([("a", [(0.0, 3), (1.0, 1)])], [2, 5, 1, 0, 0])
    sample_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    owners = build_logical_owner_batch(
        meta,
        prompt_ids=data["prompt_ids_for_adv"],
        rewards=data["total_reward"],
        sample_mask=sample_mask,
        expected_group_size=2,
    )

    weights = rollout_token_mean_weights(owners, data["token_mask"], sample_mask)

    # a_g0 trains 2 + 5 + 1 tokens; a_g1 trains none and the last row is padding.
    torch.testing.assert_close(weights, torch.tensor([1 / 8, 1 / 8, 1 / 8, 0, 0]))
    assert weights.dtype == torch.float32 and not weights.requires_grad


@pytest.mark.parametrize("resume_mode", [TOKEN, ROLLOUT])
def test_unfinished_rollout_step_resumes_under_either_mode(resume_mode: str):
    groups = [("a", [(0.0, 3), (1.0, 1), (3.0, 2)])]
    meta, data = _ragged_batch(groups, [2, 5, 1, 4, 3, 3, 0])
    data["mask_sample"][4] = True
    fresh = data.clone()
    asyncio.run(
        _controller(
            meta, fresh, forest_loss_normalization=resume_mode
        )._advantage_stage(meta)
    )
    # The checkpoint holds rows that an unfinished rollout step weighted.
    asyncio.run(
        _controller(meta, data, forest_loss_normalization=ROLLOUT)._advantage_stage(
            meta
        )
    )

    asyncio.run(
        _controller(meta, data, forest_loss_normalization=resume_mode)._advantage_stage(
            meta
        )
    )

    torch.testing.assert_close(data["sample_mask"], fresh["sample_mask"])
    torch.testing.assert_close(data["advantages"], fresh["advantages"])


@pytest.mark.parametrize("mode", [TOKEN, ROLLOUT])
def test_resume_rejects_weights_this_stage_did_not_write(mode: str):
    meta, data = _ragged_batch([("a", [(0.0, 3), (1.0, 1)])], [2, 5, 1, 4, 0])
    ctrl = _controller(meta, data, forest_loss_normalization=ROLLOUT)
    asyncio.run(ctrl._advantage_stage(meta))
    # 1/8 is a_g0's weight, not a_g1's.
    data["sample_mask"][3] = 1 / 8
    ctrl = _controller(meta, data, forest_loss_normalization=mode)
    with pytest.raises(ValueError, match="binary-masked"):
        asyncio.run(ctrl._advantage_stage(meta))


@pytest.mark.parametrize("mode", [TOKEN, ROLLOUT])
def test_rollout_mode_reports_row_token_and_rollout_counts(mode: str):
    groups = [("a", [(0.0, 3), (1.0, 1), (3.0, 2)])]
    meta, data = _ragged_batch(groups, [2, 5, 1, 4, 3, 3, 0])
    data["mask_sample"][4] = True
    ctrl = _controller(meta, data, forest_loss_normalization=mode)

    asyncio.run(ctrl._advantage_stage(meta))

    if mode == TOKEN:
        assert not ctrl._step_log_dict["rollout_loss_counts"]
        return
    # a_g0 has three rows and 8 tokens, a_g1 one row and 4 tokens; a_g2 is
    # masked and padding trains nothing.
    assert rollout_loss_step_metrics(ctrl._step_log_dict["rollout_loss_counts"]) == {
        "global_valid_seqs": 4.0,
        "global_valid_toks": 12.0,
        "global_valid_rollouts": 2.0,
    }


def test_rollout_mode_dump_keeps_binary_participation_and_records_weights(
    tmp_path: Path,
):
    meta, data = _ragged_batch([("a", [(0.0, 3), (1.0, 1)])], [2, 5, 1, 4, 0])
    # The dump also records the token ids, lengths and sampler logprobs.
    data["input_ids"] = torch.zeros(data["token_mask"].shape, dtype=torch.long)
    data["input_lengths"] = torch.full((meta.size,), data["token_mask"].shape[1])
    data["generation_logprobs"] = torch.zeros(data["token_mask"].shape)
    ctrl = _controller(
        meta,
        data,
        forest_loss_normalization=ROLLOUT,
        train_data_dump_dir=str(tmp_path),
    )

    asyncio.run(ctrl._advantage_stage(meta))
    ctrl._train_data_dump.finish_step(0, ctrl._train_data_dump_receipts)

    rows = {
        row["sample_id"][0]: row
        for row in map(
            json.loads, (tmp_path / "train_data_step1.jsonl").read_text().splitlines()
        )
    }
    by_row = [rows[sample_id] for sample_id in meta.sample_ids]
    assert [row["sample_loss_mask"] for row in by_row] == [[1], [1], [1], [1], [0]]
    assert [row["rollout_loss_weight"][0] for row in by_row] == pytest.approx(
        [1 / 8, 1 / 8, 1 / 8, 1 / 4, 0.0]
    )
    # Training still reads the weighted mask.
    torch.testing.assert_close(
        data["sample_mask"], torch.tensor([1 / 8, 1 / 8, 1 / 8, 1 / 4, 0.0])
    )


def _rollout_normalized_training_batch() -> tuple[
    KVBatchMeta, BatchedDataDict, torch.Tensor
]:
    """Two groups of ragged, branched rollouts after the real advantage stage.

    Returns the metadata, the training batch and on-policy-adjacent current
    log-probabilities whose ratios stay inside the PPO clip range.
    """
    groups = [
        ("a", [(0.0, 3), (1.0, 1), (4.0, 2)]),
        ("b", [(2.0, 1), (0.0, 2), (5.0, 2)]),
    ]
    lengths = [3, 1, 4, 5, 2, 2, 1, 3, 4, 2, 6, 0]
    meta, data = _ragged_batch(groups, lengths)
    data["mask_sample"][10] = True
    asyncio.run(
        _controller(meta, data, forest_loss_normalization=ROLLOUT)._advantage_stage(
            meta
        )
    )
    generator = torch.Generator().manual_seed(7)
    shape = data["token_mask"].shape
    prev = -1.0 - torch.rand(shape, generator=generator, dtype=torch.float64)
    batch = BatchedDataDict(
        {
            "input_ids": torch.arange(shape[0] * shape[1]).reshape(shape),
            "input_lengths": torch.full((shape[0],), shape[1]),
            "token_mask": data["token_mask"].double(),
            "sample_mask": data["sample_mask"].double(),
            "advantages": data["advantages"].double(),
            "prev_logprobs": prev,
            "generation_logprobs": prev,
            "row_index": torch.arange(shape[0]),
        }
    )
    offset = 0.1 * (2 * torch.rand(shape, generator=generator) - 1)
    return meta, batch, (prev + offset.double())[:, 1:]


def _pg_loss_and_metrics(
    logprobs: torch.Tensor, data: BatchedDataDict, normalizer: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    return ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0))(
        logprobs,
        data,
        global_valid_seqs=data["sample_mask"].sum(),
        global_valid_toks=normalizer,
    )


def _pg_loss(
    logprobs: torch.Tensor, data: BatchedDataDict, normalizer: torch.Tensor
) -> torch.Tensor:
    return _pg_loss_and_metrics(logprobs, data, normalizer)[0]


def test_rollout_normalized_loss_averages_tokens_then_rollouts():
    meta, batch, curr = _rollout_normalized_training_batch()
    curr.requires_grad_()
    normalizer = (batch["token_mask"][:, 1:] * batch["sample_mask"][:, None]).sum()
    # Five valid rollouts; b_g2 is masked and padding owns nothing.
    torch.testing.assert_close(normalizer, torch.tensor(5.0, dtype=torch.float64))
    loss, metrics = _pg_loss_and_metrics(curr, batch, normalizer)
    loss.backward()
    # A row count, not the weight sum: every row except b_g2's and padding.
    assert metrics["num_valid_samples"] == 9

    ratio = (curr - batch["prev_logprobs"][:, 1:]).exp().detach()
    advantages = batch["advantages"][:, 1:]
    trained = batch["token_mask"][:, 1:].bool()
    owners: dict[str, list[int]] = defaultdict(list)
    for row, tag in enumerate(meta.tags):
        if tag["logical_rollout_id"] not in (None, "b_g2"):
            owners[tag["logical_rollout_id"]].append(row)
    expected_loss = torch.zeros((), dtype=torch.float64)
    expected_grad = torch.zeros_like(curr)
    for rows in owners.values():
        tokens = trained[rows]
        surrogate = -(advantages[rows] * ratio[rows])[tokens]
        expected_loss += surrogate.mean() / len(owners)
        grad = expected_grad[rows]
        grad[tokens] = surrogate / (len(owners) * tokens.sum())
        expected_grad[rows] = grad
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(curr.grad, expected_grad)


@pytest.mark.parametrize(
    "dp_size,micro_batch_size,packing",
    [(1, 12, False), (2, 1, False), (2, 3, False), (3, 2, False), (2, None, True)],
)
def test_rollout_normalization_is_invariant_to_dp_and_microbatch_split(
    dp_size: int, micro_batch_size: int | None, packing: bool
):
    _, batch, curr = _rollout_normalized_training_batch()
    reference_curr = curr.clone().requires_grad_()
    normalizer = (batch["token_mask"][:, 1:] * batch["sample_mask"][:, None]).sum()
    reference = _pg_loss(reference_curr, batch, normalizer)
    reference.backward()

    if packing:
        shards, _ = batch.shard_by_batch_size(
            dp_size,
            batch_size=batch.size,
            sequence_packing_args={
                "algorithm": "modified_first_fit_decreasing",
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": 1,
                "max_tokens_per_microbatch": 14,
            },
        )
        microbatches = [
            mb
            for shard in shards
            for mb in shard.make_microbatch_iterator_for_packable_sequences()
        ]
        assert any(mb.size > 1 for mb in microbatches)
    else:
        microbatches = [
            mb
            for shard in batch.shard_by_batch_size(dp_size, batch_size=batch.size)
            for mb in shard.make_microbatch_iterator(micro_batch_size)
        ]
    assert sorted(i for mb in microbatches for i in mb["row_index"].tolist()) == list(
        range(batch.size)
    )
    # The split workers sum unnormalized microbatch losses and divide once by
    # the all-reduced local loss-mask mass.
    split_curr = curr.clone().requires_grad_()
    placeholder = torch.tensor(1.0, dtype=torch.float64)
    split_loss = torch.zeros((), dtype=torch.float64)
    local_mass = torch.zeros((), dtype=torch.float64)
    for mb in microbatches:
        local_mass += (mb["token_mask"][:, 1:] * mb["sample_mask"][:, None]).sum()
        mb_loss = _pg_loss(split_curr[mb["row_index"]], mb, placeholder)
        mb_loss.backward()
        split_loss += mb_loss.detach()
    torch.testing.assert_close(local_mass, normalizer)
    torch.testing.assert_close(split_loss / local_mass, reference)
    torch.testing.assert_close(split_curr.grad / local_mass, reference_curr.grad)
