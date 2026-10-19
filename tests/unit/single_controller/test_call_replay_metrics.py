# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call splitting preserves full-context metrics and keeps call spikes visible."""

import asyncio
import math
from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller_utils.call_replay_metrics import (
    compute_call_replay_metrics,
)
from nemo_rl.algorithms.single_controller_utils.utils import (
    _reduce_seq_logprob_error_metrics,
)
from nemo_rl.data_plane import KVBatchMeta
from tests.unit.single_controller.test_logical_advantage import _controller


def _case() -> tuple[KVBatchMeta, dict[str, torch.Tensor]]:
    tags = [
        {
            "dispatch_group_id": "group",
            "logical_rollout_id": "group_g0",
            "logical_slot": 0,
            "logical_group_size": 2,
            "segment_index": 0,
            "segment_count": 2,
            "context_segment_index": 0,
            "context_segment_count": 1,
            "call_index": 0,
            "call_count": 2,
            "is_execution_padding": False,
        },
        {
            "dispatch_group_id": "group",
            "logical_rollout_id": "group_g0",
            "logical_slot": 0,
            "logical_group_size": 2,
            "segment_index": 1,
            "segment_count": 2,
            "context_segment_index": 0,
            "context_segment_count": 1,
            "call_index": 1,
            "call_count": 2,
            "is_execution_padding": False,
        },
        {
            "dispatch_group_id": "group",
            "logical_rollout_id": "group_g1",
            "logical_slot": 1,
            "logical_group_size": 2,
            "segment_index": 0,
            "segment_count": 1,
            "context_segment_index": 0,
            "context_segment_count": 1,
            "call_index": 0,
            "call_count": 1,
            "is_execution_padding": False,
        },
        {
            "dispatch_group_id": "old",
            "logical_rollout_id": "old-owner",
            "is_execution_padding": False,
        },
        {"is_execution_padding": True},
    ]
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=[
            "group_g0_s0",
            "group_g0_s1",
            "group_g1_s0",
            "old-row",
            "group_pad0",
        ],
        tags=tags,
    )
    mask = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    policy = (
        -torch.tensor([math.log(8), 0.0, math.log(2), math.log(3), 0.0])[:, None]
        .expand(5, 6)
        .clone()
    )
    policy[-1] = float("nan")
    return meta, {
        "policy_logprobs": policy,
        "generation_logprobs": torch.zeros(5, 6),
        "token_mask": mask,
        "sample_mask": torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32),
    }


@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("precision", [torch.float32, torch.float64])
def test_exact_target_weighting_and_legacy_contexts_without_hiding_call_peaks(
    reorder: bool,
    precision: torch.dtype,
) -> None:
    meta, data = _case()
    data["policy_logprobs"] = data["policy_logprobs"].to(precision)
    data["generation_logprobs"] = data["generation_logprobs"].to(precision)
    if reorder:
        order = [4, 2, 1, 3, 0]
        meta = meta.subset(order)
        data = {key: value[order] for key, value in data.items()}
    before = {key: value.clone() for key, value in data.items()}
    result = compute_call_replay_metrics(meta, **data)
    assert result is not None and result.valid_context_count == 3
    metrics = result.metrics
    # Context0=(one target at8 + three targets at1)/4=2.75. The other
    # call context is2 and the retained legacy context is3.
    assert metrics["max_seq_mult_prob_error"] == pytest.approx(3.0)
    assert metrics["mean_seq_mult_prob_error"] == pytest.approx((2.75 + 2 + 3) / 3)
    assert metrics["min_seq_mult_prob_error"] == pytest.approx(2.0)
    assert metrics["max_call_mult_prob_error"] == pytest.approx(8.0)
    assert metrics["mean_call_mult_prob_error"] == pytest.approx((8 + 1 + 2) / 3)
    assert metrics["_num_valid_calls"] == 3
    assert metrics["tmpe_legacy_context_row_count"] == 1
    assert (
        metrics["max_seq_mult_prob_error_after_mask"]
        == metrics["max_seq_mult_prob_error"]
    )
    for key in data:
        torch.testing.assert_close(data[key], before[key], equal_nan=True)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_call",
        "duplicate_call",
        "different_count",
        "missing_context",
        "partial_metadata",
        "mixed_owner",
        "padding_targets",
    ],
)
def test_corrupt_or_incomplete_publication_is_not_reported_as_healthy(
    corruption: str,
) -> None:
    meta, data = _case()
    meta.tags = deepcopy(meta.tags)
    if corruption == "missing_call":
        meta = meta.subset([0, 2, 3, 4])
        data = {key: value[[0, 2, 3, 4]] for key, value in data.items()}
    elif corruption == "duplicate_call":
        meta.tags[1]["call_index"] = 0
    elif corruption == "different_count":
        meta.tags[1]["call_count"] = 3
    elif corruption == "missing_context":
        meta.tags[0]["context_segment_count"] = meta.tags[1][
            "context_segment_count"
        ] = 2
    elif corruption == "partial_metadata":
        del meta.tags[1]["call_count"]
    elif corruption == "mixed_owner":
        meta.tags[3].update(dispatch_group_id="group", logical_rollout_id="group_g0")
    else:
        data["sample_mask"][-1] = 1
        data["token_mask"][-1, 1] = 1
    with pytest.raises(ValueError):
        compute_call_replay_metrics(meta, **data)


@pytest.mark.parametrize("masked", [False, True])
def test_nonfinite_valid_target_remains_visible_and_masked_padding_does_not_poison(
    masked: bool,
) -> None:
    meta, data = _case()
    data["policy_logprobs"][0, 1] = float("inf")
    if masked:
        data["sample_mask"][0] = 0
    result = compute_call_replay_metrics(meta, **data)
    assert result is not None
    assert math.isfinite(result.metrics["max_seq_mult_prob_error"]) == masked
    assert math.isfinite(result.metrics["max_call_mult_prob_error"]) == masked


def test_legacy_metadata_keeps_the_existing_metrics_path() -> None:
    meta, data = _case()
    meta = meta.subset([3, 4])
    data = {key: value[[3, 4]] for key, value in data.items()}
    assert compute_call_replay_metrics(meta, **data) is None


def test_streaming_reduction_uses_context_counts_and_preserves_call_maximum() -> None:
    meta, data = _case()
    result = compute_call_replay_metrics(meta, **data)
    record = {
        **result.metrics,
        "_num_valid_seqs_before": 3.0,
        "_num_valid_seqs_after": 3.0,
        "num_masked_seqs_by_logprob_error": 0,
        "masked_correct_pct": 0.0,
    }
    second = {
        **record,
        "_num_valid_calls": 1.0,
        "max_call_mult_prob_error": 2.0,
        "mean_call_mult_prob_error": 2.0,
        "min_call_mult_prob_error": 2.0,
        "tmpe_legacy_context_row_count": 0.0,
    }
    reduced = _reduce_seq_logprob_error_metrics([record, second])
    assert reduced["raw_tmpe_evaluated_seq_count"] == 6
    assert reduced["raw_tmpe_evaluated_call_count"] == 4
    assert reduced["max_call_mult_prob_error"] == pytest.approx(8)
    assert reduced["mean_call_mult_prob_error"] == pytest.approx((8 + 1 + 2 + 2) / 4)
    assert reduced["tmpe_legacy_context_row_count"] == 1


def test_controller_reports_original_context_metrics_and_keeps_all_reward_votes_and_targets() -> (
    None
):
    meta, data = _case()
    meta = meta.subset([0, 1, 2])
    data = {key: value[:3] for key, value in data.items()}
    tensor_data = TensorDict(
        {
            "prompt_ids_for_adv": torch.ones(3, 2, dtype=torch.long),
            "total_reward": torch.tensor([0.0, 0.0, 1.0]),
            "token_mask": data["token_mask"],
            "sample_mask": data["sample_mask"],
            "prev_logprobs": data["policy_logprobs"],
            "generation_logprobs": data["generation_logprobs"],
            "mask_sample": torch.zeros(3, dtype=torch.bool),
            "truncated": torch.zeros(3, dtype=torch.bool),
        },
        batch_size=[3],
    )
    meta.fields = list(tensor_data.keys())
    controller = _controller(meta, tensor_data)
    controller._policy_logprobs_required = True
    _, valid = asyncio.run(controller._advantage_stage(meta))
    assert valid
    record = controller._step_log_dict["seq_logprob_error_metrics"][0]
    assert record["max_seq_mult_prob_error"] == pytest.approx(2.75)
    assert record["mean_seq_mult_prob_error"] == pytest.approx((2.75 + 2) / 2)
    assert record["max_call_mult_prob_error"] == pytest.approx(8)
    assert record["_num_valid_seqs_before"] == record["_num_valid_seqs_after"] == 2
    assert record["num_masked_seqs_by_logprob_error"] == 0
    assert tensor_data["sample_mask"].tolist() == [1, 1, 1]
    assert tensor_data["token_mask"].sum() == 6
    assert controller._step_log_dict["rewards"][0].tolist() == [0, 1]
