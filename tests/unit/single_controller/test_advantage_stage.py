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

from collections.abc import Sequence
from dataclasses import fields

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    SHARD_INVARIANT_ESTIMATORS,
    AdvantageOutcome,
    AdvantageRequest,
    AdvantageStageConfig,
    group_index_column,
    row_group_ids,
    split_meta_by_prompt_group,
)
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.algorithms.single_controller_utils.utils import (
    AdvantagePartial,
    RewardPartial,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import GROUP_ID_TAG
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
        shardable=True,
    )
    base.update(overrides)
    return AdvantageStageConfig(**base)


class TestInputFields:
    def test_only_the_always_present_columns_by_default(self) -> None:
        adv = AdvantageConfig()
        assert _config().input_fields() == [
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


def _meta_for_group_sizes(sizes: Sequence[int]) -> KVBatchMeta:
    """Rows for one contiguous prompt group per entry in ``sizes``.

    Tags carry no ``dataset_source``, matching a dataset whose rows have no
    ``dataset`` key -- the case that made an even earlier, tag-keyed splitter
    decline every time.
    """
    sample_ids: list[str] = []
    tags: list[dict[str, object]] = []
    for group, size in enumerate(sizes):
        for generation in range(size):
            sample_ids.append(f"group-{group}_g{generation}")
            tags.append({"weight_version": 1, GROUP_ID_TAG: f"group-{group}"})
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids,
        fields=["total_reward"],
        sequence_lengths=[8] * len(sample_ids),
        tags=tags,
    )


def _grouped_meta(num_groups: int, per_group: int) -> KVBatchMeta:
    """Rows for ``num_groups`` contiguous prompt groups of equal size."""
    return _meta_for_group_sizes([per_group] * num_groups)


def _shard_group_ids(shard: KVBatchMeta) -> list[str]:
    return [tag[GROUP_ID_TAG] for tag in shard.tags]


class TestSplitMetaByPromptGroup:
    def test_shards_cover_every_row_in_original_order(self):
        meta = _grouped_meta(num_groups=8, per_group=4)
        shards = split_meta_by_prompt_group(meta, 4)
        assert shards is not None and len(shards) == 4
        rejoined = [sid for shard in shards for sid in shard.sample_ids]
        assert rejoined == meta.sample_ids
        # Per-sample sidecars travel with the rows they describe.
        assert [t for shard in shards for t in shard.tags] == meta.tags
        assert [n for s in shards for n in s.sequence_lengths] == meta.sequence_lengths

    def test_every_shard_holds_whole_groups(self):
        meta = _grouped_meta(num_groups=8, per_group=4)
        shards = split_meta_by_prompt_group(meta, 3)
        assert shards is not None
        assert all(len(shard.sample_ids) % 4 == 0 for shard in shards)

    def test_splits_without_any_dataset_source_tag(self):
        """Regression: an earlier tag-keyed version declined here and measured
        nothing."""
        meta = _grouped_meta(num_groups=2048, per_group=16)
        shards = split_meta_by_prompt_group(meta, 8)
        assert shards is not None
        assert [len(shard.sample_ids) for shard in shards] == [4096] * 8

    def test_never_returns_more_shards_than_requested(self):
        meta = _grouped_meta(num_groups=7, per_group=2)
        shards = split_meta_by_prompt_group(meta, 4)
        assert shards is not None and len(shards) <= 4
        assert [s for sh in shards for s in sh.sample_ids] == meta.sample_ids

    def test_groups_of_unequal_size_are_never_cut(self):
        """The group-size version assumed every group had
        num_generations_per_prompt rows, so a feature like Never Give Up that
        produces variable-size groups was cut silently."""
        meta = _meta_for_group_sizes([4, 12, 4])
        shards = split_meta_by_prompt_group(meta, 3)
        assert shards is not None
        assert [len(shard.sample_ids) for shard in shards] == [4, 12, 4]
        assert [_shard_group_ids(s)[0] for s in shards] == [
            "group-0",
            "group-1",
            "group-2",
        ]
        assert all(len(set(_shard_group_ids(s))) == 1 for s in shards)

    def test_unequal_groups_pack_whole_into_fewer_shards(self):
        meta = _meta_for_group_sizes([4, 12, 4, 8, 4, 4])
        shards = split_meta_by_prompt_group(meta, 3)
        assert shards is not None and len(shards) <= 3
        assert [s for sh in shards for s in sh.sample_ids] == meta.sample_ids
        # Every group's rows land wholly inside one shard.
        placements = {
            group_id: i
            for i, shard in enumerate(shards)
            for group_id in _shard_group_ids(shard)
        }
        for i, shard in enumerate(shards):
            assert all(placements[g] == i for g in _shard_group_ids(shard))

    def test_declines_when_groups_are_fewer_than_two(self):
        assert split_meta_by_prompt_group(_grouped_meta(1, 16), 8) is None

    def test_declines_without_a_pool_to_spread_across(self):
        meta = _grouped_meta(num_groups=8, per_group=4)
        assert split_meta_by_prompt_group(meta, 1) is None
        assert split_meta_by_prompt_group(meta, 0) is None

    def test_declines_when_groups_are_interleaved(self):
        """An interleaved layout has no contiguous whole-group cut, so it must
        decline rather than reorder rows behind the caller's back."""
        meta = _grouped_meta(num_groups=2, per_group=2)
        interleaved = meta.tags[:]
        interleaved[1], interleaved[2] = interleaved[2], interleaved[1]
        meta = KVBatchMeta(
            partition_id=meta.partition_id,
            task_name=meta.task_name,
            sample_ids=meta.sample_ids,
            fields=meta.fields,
            sequence_lengths=meta.sequence_lengths,
            tags=interleaved,
        )
        assert split_meta_by_prompt_group(meta, 2) is None

    def test_a_partial_group_still_shards_the_groups_it_has(self):
        """The group-size version declined on any row count that was not a
        whole multiple, because it could not tell a trimmed group from a
        different layout. Real boundaries make the first groups shardable."""
        meta = _grouped_meta(num_groups=8, per_group=4).slice(0, 30)
        shards = split_meta_by_prompt_group(meta, 4)
        assert shards is not None
        assert [s for sh in shards for s in sh.sample_ids] == meta.sample_ids


class TestGroupIdIsMandatory:
    """A missing key must raise: falling back to prompt tokens is the bug."""

    def test_untagged_meta_raises(self):
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["sample-0"],
        )
        with pytest.raises(ValueError, match="carry no tags"):
            row_group_ids(meta)

    def test_a_row_without_the_tag_raises_and_names_it(self):
        meta = _grouped_meta(num_groups=2, per_group=2)
        stripped = [dict(tag) for tag in meta.tags]
        del stripped[2][GROUP_ID_TAG]
        meta = KVBatchMeta(
            partition_id=meta.partition_id,
            task_name=meta.task_name,
            sample_ids=meta.sample_ids,
            tags=stripped,
        )
        with pytest.raises(ValueError, match="group-1_g0"):
            row_group_ids(meta)


def _baseline(prompt_key: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    baseline, _, _ = calculate_baseline_and_std_per_prompt(
        prompt_key, rewards, torch.ones_like(rewards)
    )
    return baseline


class TestGroupIdIsTheBaselineKey:
    """Two groups can share prompt text; keying on tokens merges them.

    DAPO-Math-17k stores each of its prompts 100 times, so a 512-prompt chunk
    carries roughly seven same-prompt pairs.
    """

    # Groups 0 and 2 are different groups that happen to carry identical
    # prompt tokens. Their reward means differ, so merging them is visible.
    GROUP_SIZE = 4
    TOKENS = {0: [1, 2, 3], 1: [4, 5, 6], 2: [1, 2, 3], 3: [7, 8, 9]}
    # Groups 0 and 2 have different reward means, so a merged baseline is not
    # the same number as either group's own.
    REWARDS = {
        0: [1.0, 0.0, 0.0, 0.0],
        1: [1.0, 1.0, 0.0, 0.0],
        2: [1.0, 1.0, 1.0, 0.0],
        3: [0.0, 0.0, 0.0, 1.0],
    }

    @property
    def meta(self) -> KVBatchMeta:
        return _meta_for_group_sizes([self.GROUP_SIZE] * len(self.TOKENS))

    @property
    def token_key(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.TOKENS[g]
                for g in sorted(self.TOKENS)
                for _ in range(self.GROUP_SIZE)
            ]
        )

    @property
    def rewards(self) -> torch.Tensor:
        return torch.tensor([r for g in sorted(self.REWARDS) for r in self.REWARDS[g]])

    def test_the_token_key_merges_groups_that_share_prompt_text(self) -> None:
        # The same batch keyed so that groups 0 and 2 are deliberately one
        # group. If the token key matches this, it merged them.
        merged = group_index_column(
            [
                f"group-{0 if g == 2 else g}"
                for g in sorted(self.TOKENS)
                for _ in range(self.GROUP_SIZE)
            ]
        )
        assert torch.equal(
            _baseline(self.token_key, self.rewards),
            _baseline(merged, self.rewards),
        )

    def test_the_group_id_key_gives_each_group_its_own_baseline(self) -> None:
        group_key = group_index_column(row_group_ids(self.meta))
        per_group = torch.cat(
            [
                _baseline(
                    group_index_column([f"group-{g}"] * self.GROUP_SIZE),
                    torch.tensor(self.REWARDS[g]),
                )
                for g in sorted(self.REWARDS)
            ]
        )
        assert torch.equal(_baseline(group_key, self.rewards), per_group)
        # And that is not what the token key produced.
        assert not torch.equal(
            _baseline(self.token_key, self.rewards),
            _baseline(group_key, self.rewards),
        )

    def _shardwise(self, key_for) -> torch.Tensor:
        """Baselines computed one shard at a time, rejoined in row order."""
        shards = split_meta_by_prompt_group(self.meta, 2)
        assert shards is not None
        rewards = self.rewards
        pieces, start = [], 0
        for shard in shards:
            stop = start + len(shard.sample_ids)
            pieces.append(_baseline(key_for(shard, start, stop), rewards[start:stop]))
            start = stop
        return torch.cat(pieces)

    def test_sharding_is_exact_under_the_group_id_key(self) -> None:
        """What SHARD_INVARIANT_ESTIMATORS claims, as an equality."""
        whole = _baseline(group_index_column(row_group_ids(self.meta)), self.rewards)
        shardwise = self._shardwise(
            lambda shard, start, stop: group_index_column(row_group_ids(shard))
        )
        assert torch.equal(shardwise, whole)

    def test_sharding_was_not_exact_under_the_token_key(self) -> None:
        """Groups 0 and 2 fall in different shards, so the merge the token key
        performs on the whole batch cannot happen shard by shard -- which made
        grpo's advantages a function of num_advantage_workers."""
        token_key = self.token_key
        whole = _baseline(token_key, self.rewards)
        shardwise = self._shardwise(lambda shard, start, stop: token_key[start:stop])
        assert not torch.equal(shardwise, whole)


class TestShardInvariantEstimators:
    """Splitting a batch is only sound for estimators that do not reduce over it."""

    @pytest.mark.parametrize(
        "name",
        [
            # advantage_estimator.py: `advantages.std()` / `.mean()` over the
            # whole batch, unconditionally.
            "gdpo",
            # advantage_estimator.py: "global normalization across the batch",
            # unconditionally.
            "reinforce_plus_plus",
            # Both normalize over the batch whenever normalize_advantages is
            # set, and it defaults to True for both.
            "gae",
            "raw_reward",
        ],
    )
    def test_batch_normalizing_estimators_are_not_shardable(self, name: str) -> None:
        assert name not in SHARD_INVARIANT_ESTIMATORS

    def test_only_the_row_and_group_local_estimators_are_shardable(self) -> None:
        """Membership is opt-in, so a new estimator defaults to unshardable.

        grpo's baseline is per prompt group, which the split preserves now that
        the key is GROUP_ID_TAG rather than the prompt tokens two groups can
        share, and opd's advantage is a per-token teacher/student difference
        that reads no other row. Everything else has to be checked before it is
        added here.
        """
        assert SHARD_INVARIANT_ESTIMATORS == {"grpo", "opd"}


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
