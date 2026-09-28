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
"""The advantage stage's boundary: what it fetches and what it hands back."""

from dataclasses import fields

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    AdvantageOutcome,
    AdvantageRequest,
    AdvantageStageConfig,
)
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.algorithms.single_controller_utils.utils import (
    AdvantagePartial,
    RewardPartial,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.utils.rpc_guard import assert_metadata_only


def _config(**overrides) -> AdvantageStageConfig:
    base = dict(
        advantage=AdvantageConfig(),
        algo=None,
        is_ppo=False,
        policy_logprobs_required=False,
        reference_logprobs_required=False,
        teacher_logprobs_required=False,
        message_level_advantage_penalties_enabled=False,
    )
    base.update(overrides)
    return AdvantageStageConfig(**base)


class TestInputFields:
    def test_only_the_always_present_columns_by_default(self) -> None:
        adv = AdvantageConfig()
        assert _config().input_fields() == [
            adv.prompt_ids_field,
            adv.reward_field,
            adv.token_mask_field,
            adv.sample_mask_field,
            adv.mask_sample_field,
            adv.truncated_field,
        ]

    def test_policy_logprobs_pull_both_logprob_columns(self) -> None:
        adv = AdvantageConfig()
        fields = _config(policy_logprobs_required=True).input_fields()
        assert adv.policy_logprobs_field in fields
        assert adv.generation_logprobs_field in fields

    def test_ppo_pulls_values_and_penalties_pull_their_masks(self) -> None:
        adv = AdvantageConfig()
        fields = _config(
            is_ppo=True,
            message_level_advantage_penalties_enabled=True,
        ).input_fields()
        assert adv.values_field in fields
        assert adv.invalid_tool_call_mask_field in fields
        assert adv.malformed_thinking_mask_field in fields

    def test_repeated_batch_fields_do_not_duplicate_the_reward_column(self) -> None:
        adv = AdvantageConfig(repeated_batch_fields=["total_reward"])
        fields = _config(advantage=adv).input_fields()
        assert fields.count("total_reward") == 1


class TestRpcBoundaryStaysMetadataOnly:
    """The actor only helps if nothing cohort-sized crosses to it."""

    def test_request_carries_no_payload(self) -> None:
        request = AdvantageRequest(
            meta=KVBatchMeta(
                partition_id="rollout_data",
                task_name="train",
                sample_ids=["sample-0"],
                fields=["total_reward"],
            ),
        )
        assert_metadata_only(request)

    def test_outcome_carries_no_payload(self) -> None:
        outcome = AdvantageOutcome(
            meta=KVBatchMeta(
                partition_id="rollout_data",
                task_name="train",
                sample_ids=["sample-0"],
                fields=["advantages"],
            ),
            has_valid_training_tokens=True,
            num_mask_sample_filtered=1,
            reward_partial=RewardPartial.from_rows(torch.tensor([1.0, 3.0])),
            advantage_partial=AdvantagePartial.from_values(torch.tensor([0.5, 1.5])),
            seq_logprob_error_metrics={"max_seq_mult_prob_error": 1.25},
            opd_stat_sum=2.0,
            opd_stat_sumsq=2.5,
            opd_stat_count=2,
        )
        assert_metadata_only(outcome)

    def test_a_tensor_on_the_outcome_is_rejected(self) -> None:
        outcome = AdvantageOutcome(
            meta=KVBatchMeta(
                partition_id="rollout_data",
                task_name="train",
                sample_ids=["sample-0"],
                fields=["advantages"],
            ),
            has_valid_training_tokens=True,
            num_mask_sample_filtered=0,
            reward_partial=RewardPartial.from_rows(torch.tensor([1.0])),
            # The regression this guard exists for: handing back the tensor
            # instead of its reduction.
            advantage_partial=torch.zeros(4),  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="torch.Tensor"):
            assert_metadata_only(outcome)


def test_rpc_dataclass_fields_are_classified() -> None:
    """A new field on either RPC dataclass must be a deliberate choice.

    assert_metadata_only cannot tell a heavy list[int] of token ids from a short
    list of metadata, so FORBIDDEN_RPC_KEYS is maintained by hand. Pinning the
    inventory makes a new field fail here until someone decides whether it is
    light enough to cross the wire.
    """
    assert {f.name for f in fields(AdvantageRequest)} == {"meta"}
    assert {f.name for f in fields(AdvantageOutcome)} == {
        "meta",
        "has_valid_training_tokens",
        "num_mask_sample_filtered",
        # Already reduced to a handful of floats, not the reward tensors.
        "reward_partial",
        "advantage_partial",
        "seq_logprob_error_metrics",
        # OPD's moments, as this call's contribution rather than a total.
        "opd_stat_sum",
        "opd_stat_sumsq",
        "opd_stat_count",
    }
