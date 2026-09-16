"""Exercise logical-owner GRPO through the actual SingleController stage."""

import asyncio
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.data_plane import KVBatchMeta


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
    return KVBatchMeta("train", "train", ids, fields=list(data.keys()), tags=tags), data


def _controller(meta, data, *, grpo=None, loss=None):
    settings = dict(
        num_generations_per_prompt=2,
        adv_estimator=AdvEstimatorConfig(
            normalize_rewards=False, use_leave_one_out_baseline=False
        ),
    )
    if meta.tags:
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
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._message_level_advantage_penalties_enabled = (
        config.invalid_tool_call_advantage is not None
        or config.malformed_thinking_advantage is not None
    )
    ctrl._step_log_dict = defaultdict(list)
    return ctrl


@pytest.mark.parametrize("population", ["valid_owners", "all_owners"])
@pytest.mark.parametrize("reorder", [False, True])
def test_deduplicates_unequal_segments_without_pooling_identical_prompts_across_groups(
    population, reorder
):
    meta, data = _batch(
        [("a", [(0.0, 2), (0.0, 1)]), ("b", [(1.0, 3), (1.0, 1)])], padding=True
    )
    if reorder:
        order = [4, 1, 6, 0, 7, 5, 3, 2]
        meta = meta.subset(order)
        data = data[order]
    ctrl = _controller(meta, data, grpo={"baseline_population": population})
    result, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid and "advantages" in result.fields
    assert len(ctrl._dp_client.puts) == 1
    assert sorted(ctrl._step_log_dict["rewards"][0].tolist()) == [0, 0, 1, 1]
    assert ctrl._step_log_dict["sample_masks"][0].tolist() == [1, 1, 1, 1]
    assert data["advantages"].count_nonzero() == 0


@pytest.mark.parametrize("population", ["valid_owners", "all_owners"])
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("leave_one_out", [False, True])
def test_sibling_rollouts_with_distinct_prompts_share_baseline(
    population: str, reorder: bool, leave_one_out: bool
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
            "baseline_population": population,
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
        gym_harness,
    )

    data_plane = PublicationDataPlane()
    source = TQTokenSource(data_plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(data_plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
    )
    try:
        owners = [
            [
                capture_segment(harness, "a_g0_s0", child=True),
                capture_segment(harness, "a_g0_s1"),
            ],
            [capture_segment(harness, "a_g1_s0")],
        ]
        finalizer = RolloutReassembler(
            data_plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
        )
        result = finalizer.finalize_group(
            "a",
            ["a_g0", "a_g1"],
            [None, None],
            [0.0, 1.0],
            mask_sample=[False, False],
            fallback_weight_version=7,
            prompt_idx=99,
            logical_segments=owners,
        )
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


@pytest.mark.parametrize(
    "population,expected", [("valid_owners", [-1.0, 1.0]), ("all_owners", [-4.0, -2.0])]
)
@pytest.mark.parametrize("filter_field", ["sample_mask", "mask_sample", "truncated"])
@pytest.mark.parametrize("ordinary", [False, True])
def test_masked_owner_participation_never_revives_its_segments(
    population, expected, filter_field, ordinary
):
    counts = [1, 1, 1] if ordinary else [2, 1, 2]
    meta, data = _batch([("a", list(zip([0.0, 2.0, 10.0], counts)))])
    if ordinary:
        meta.tags = None
    data[filter_field][-1] = 0 if filter_field == "sample_mask" else 1
    ctrl = _controller(
        meta, data, grpo={"baseline_population": population, "overlong_filtering": True}
    )
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    torch.testing.assert_close(data["advantages"][0], torch.full((3,), expected[0]))
    torch.testing.assert_close(
        data["advantages"][counts[0]], torch.full((3,), expected[1])
    )
    assert data["sample_mask"][-counts[-1] :].count_nonzero() == 0
    assert (data["token_mask"] * data["sample_mask"].unsqueeze(-1))[
        -counts[-1] :
    ].count_nonzero() == 0


def test_all_invalid_owners_take_existing_no_training_path():
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])], padding=True)
    data["mask_sample"][:] = True
    ctrl = _controller(meta, data, grpo={"baseline_population": "all_owners"})
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert not valid
    assert data["advantages"].count_nonzero() == 0
    assert data["sample_mask"].count_nonzero() == 0


@pytest.mark.parametrize(
    "population,first_advantage", [("valid_owners", -1.0), ("all_owners", -4.0)]
)
def test_failed_owner_single_placeholder_stays_loss_masked(population, first_advantage):
    meta, data = _batch([("a", [(0.0, 2), (2.0, 1), (10.0, 1)])])
    data["sample_mask"][-1] = 0
    data["token_mask"][-1] = 0
    ctrl = _controller(meta, data, grpo={"baseline_population": population})
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    assert data["advantages"][0, 1].item() == first_advantage
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
        meta.tags[0] = {}
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
    with pytest.raises(ValueError, match="CC"):
        asyncio.run(ctrl._advantage_stage(meta))
    assert not ctrl._dp_client.puts


@pytest.mark.parametrize(
    "grpo,loss",
    [
        ({"adv_estimator": {"name": "gdpo"}}, {}),
        ({"advantage_clip_low": -1}, {}),
        ({"seq_logprob_error_threshold": 2}, {}),
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
def test_unsupported_cc_estimator_or_objective_is_rejected(grpo, loss):
    meta, data = _batch([("a", [(0.0, 2), (1.0, 1)])])
    ctrl = _controller(meta, data, grpo=grpo, loss=loss)
    with pytest.raises(ValueError, match="CC"):
        asyncio.run(ctrl._advantage_stage(meta))
    assert not ctrl._dp_client.puts


def test_baseline_population_default_and_invalid_value():
    assert GRPOConfig().baseline_population == "valid_owners"
    with pytest.raises(ValidationError):
        GRPOConfig(baseline_population="segments")
