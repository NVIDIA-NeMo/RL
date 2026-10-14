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
"""Stage-specific weights must change train assignment without changing logprobs."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.data.packing.shared_prefix_metadata import SHARED_PREFIX_PROMPT_LENGTHS
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS, LP_SEED_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import (
    SharedPrefixTrainingConfig,
    validate_shared_prefix_training_config,
)
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.tq_policy import TQPolicy


class TestStageWorkWeights(unittest.TestCase):
    def test_legacy_fallback_and_training_override(self):
        cfg = SharedPrefixTrainingConfig(mode="train", shard_work_weights=(1, 1))
        self.assertEqual(cfg.work_weights_for(stage="train"), (1, 1))
        self.assertEqual(cfg.work_weights_for(stage="logprobs"), (1, 1))
        cfg.training_shard_work_weights = (0, 1)
        self.assertEqual(cfg.work_weights_for(stage="train"), (0, 1))
        self.assertEqual(cfg.work_weights_for(stage="logprobs"), (1, 1))

    def test_training_override_without_common_weights(self):
        cfg = SharedPrefixTrainingConfig(
            mode="train", training_shard_work_weights=(0, 1)
        )
        self.assertEqual(cfg.work_weights_for(stage="train"), (0, 1))
        self.assertIsNone(cfg.work_weights_for(stage="logprobs"))

    def test_disabled_stages_do_not_resolve_weights(self):
        cfg = SharedPrefixTrainingConfig(mode="logprobs", shard_work_weights=(1, 1))
        self.assertIsNone(cfg.work_weights_for(stage="train"))
        self.assertEqual(cfg.work_weights_for(stage="logprobs"), (1, 1))
        for mode in ("disabled", "dense"):
            with self.subTest(mode=mode), self.assertRaises(ValidationError):
                SharedPrefixTrainingConfig(mode=mode, shard_work_weights=(1, 1))

    def test_invalid_overrides_fail_before_backend_setup(self):
        for weights in ((-1, 1), (0, 0)):
            with (
                self.subTest(weights=weights),
                self.assertRaisesRegex(ValueError, "positive sum"),
            ):
                validate_shared_prefix_training_config(
                    {
                        "shared_prefix_training": dict(
                            mode="train", training_shard_work_weights=weights
                        )
                    }
                )
        for mode in ("disabled", "dense", "logprobs"):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(ValueError, "requires shared train"),
            ):
                validate_shared_prefix_training_config(
                    {
                        "shared_prefix_training": dict(
                            mode=mode, training_shard_work_weights=(0, 1)
                        )
                    }
                )
        for weights in ((False, 1), (0, 1.0), (0, "1")):
            with self.subTest(weights=weights), self.assertRaises(ValidationError):
                SharedPrefixTrainingConfig(
                    mode="train", training_shard_work_weights=weights
                )

    def test_tq_backfills_metadata_for_training_only_override(self):
        cfg = SharedPrefixTrainingConfig(
            mode="train", training_shard_work_weights=(0, 1)
        )
        policy = SimpleNamespace(shared_prefix_training_config=cfg, dp_client=object())
        meta = SimpleNamespace(tags=None, sequence_lengths=None)
        self.assertIs(
            TQPolicy._with_shared_work_metadata(policy, meta, stage="logprobs"), meta
        )
        with self.assertRaisesRegex(ValueError, "requires sequence lengths"):
            TQPolicy._with_shared_work_metadata(policy, meta, stage="train")


def _meta() -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["p0_g0", "p0_g1"],
        sequence_lengths=[10, 12],
    )


def _tq_policy(config: SharedPrefixTrainingConfig | None) -> tuple[TQPolicy, MagicMock]:
    """Bare TQPolicy with the attributes the dispatch paths touch."""
    policy = object.__new__(TQPolicy)
    if config is not None:
        policy.shared_prefix_training_config = config
    policy.cfg = {"train_global_batch_size": 2, "train_micro_batch_size": 1}
    policy._router_replay_enabled = False
    policy._opd_full_field = None
    policy._opd_full_teacher_index_field = None
    policy.flops_tracker = None
    policy.dp_client = MagicMock()
    policy.worker_group = MagicMock()
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = 2
    return policy, policy.worker_group


def _dispatch(policy: TQPolicy, stage: str) -> tuple[MagicMock, MagicMock]:
    """Run one stage's dispatch and return the packing-args and sharder mocks."""
    meta = _meta()
    with (
        patch.object(TQPolicy, "_stamp_pad_seqlen"),
        patch.object(
            TQPolicy, "_packing_args", return_value=(None, None)
        ) as packing_args,
        patch(
            "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
            return_value=([meta, meta], None),
        ) as shard,
    ):
        if stage == "logprobs":
            policy.get_logprobs_from_meta(meta)
        else:
            policy.train_microbatches_from_meta(meta)
    return packing_args, shard


@pytest.mark.parametrize(
    "block,stage,budget,groups,weights",
    [
        (None, "logprobs", "logprob_mb_tokens", False, None),
        (None, "train", "train_mb_tokens", False, None),
        ({"mode": "dense"}, "logprobs", "logprob_mb_tokens", False, None),
        (
            {"mode": "logprobs", "shard_work_weights": (1, 1)},
            "logprobs",
            "logprob_mb_tokens",
            True,
            (1, 1),
        ),
        (
            {"mode": "logprobs", "shard_work_weights": (1, 1)},
            "train",
            "train_mb_tokens",
            False,
            None,
        ),
        (
            {
                "mode": "train",
                "shard_work_weights": (1, 1),
                "training_shard_work_weights": (0, 1),
            },
            "logprobs",
            "logprob_mb_tokens",
            True,
            (1, 1),
        ),
        (
            {
                "mode": "train",
                "shard_work_weights": (1, 1),
                "training_shard_work_weights": (0, 1),
            },
            "train",
            "train_mb_tokens",
            True,
            (0, 1),
        ),
        (
            {
                "mode": "train",
                "match_logprob_training_layout": True,
                "shard_work_weights": (1, 1),
                "training_shard_work_weights": (0, 1),
            },
            "logprobs",
            "train_mb_tokens",
            True,
            (0, 1),
        ),
    ],
)
def test_tq_dispatch_passes_stage_budget_groups_and_weights(
    block, stage, budget, groups, weights
):
    config = None if block is None else SharedPrefixTrainingConfig(**block)
    policy, _ = _tq_policy(config)
    with patch.object(
        TQPolicy, "_with_shared_work_metadata", side_effect=lambda meta, **_: meta
    ):
        packing_args, shard = _dispatch(policy, stage)

    packing_args.assert_called_once_with(budget)
    assert shard.call_args.kwargs["shared_prefix_groups"] is groups
    assert shard.call_args.kwargs["shared_prefix_work_weights"] == weights
    fields = shard.call_args.args[0].fields
    base_fields = LP_SEED_FIELDS if stage == "logprobs" else DP_TRAIN_FIELDS
    assert fields[: len(base_fields)] == list(base_fields)
    assert (SHARED_PREFIX_PROMPT_LENGTHS in fields) is groups


def test_tq_backfills_prompt_length_tags_from_the_stored_column():
    config = SharedPrefixTrainingConfig(
        mode="train", training_shard_work_weights=(0, 1)
    )
    policy, _ = _tq_policy(config)
    meta = _meta()
    with patch(
        "nemo_rl.models.policy.tq_policy.read_columns",
        return_value={SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor([4, 4])},
    ) as read_columns:
        backfilled = policy._with_shared_work_metadata(meta, stage="train")

    read_columns.assert_called_once_with(
        policy.dp_client, meta, [SHARED_PREFIX_PROMPT_LENGTHS]
    )
    assert [tag[SHARED_PREFIX_PROMPT_LENGTHS] for tag in backfilled.tags] == [4, 4]
    # Rows that already carry the tag are not read again.
    with patch("nemo_rl.models.policy.tq_policy.read_columns") as read_columns:
        assert (
            policy._with_shared_work_metadata(backfilled, stage="train") is backfilled
        )
    read_columns.assert_not_called()


def test_tq_stubs_without_init_default_to_disabled():
    policy, _ = _tq_policy(None)
    assert policy.shared_prefix_training_config.mode == "disabled"
    assert policy._with_shared_prefix_fields(LP_SEED_FIELDS, stage="logprobs") == (
        LP_SEED_FIELDS
    )


@pytest.mark.parametrize("mode", ["logprobs", "train"])
def test_sync_trainer_step_rejects_shared_execution(mode):
    policy, _ = _tq_policy(SharedPrefixTrainingConfig(mode=mode))
    with pytest.raises(NotImplementedError, match="single-controller trainer"):
        policy.prepare_step(num_samples=2, group_size=2)
    policy.dp_client.register_partition.assert_not_called()


@pytest.mark.parametrize(
    "mode,stage", [("logprobs", "logprobs"), ("train", "logprobs"), ("train", "train")]
)
def test_non_tq_policy_sharding_rejects_shared_execution(mode, stage):
    policy = object.__new__(Policy)
    policy.shared_prefix_training_config = SharedPrefixTrainingConfig(mode=mode)
    data = BatchedDataDict({"input_lengths": torch.tensor([4, 4])})
    with pytest.raises(NotImplementedError, match="single-controller TQPolicy"):
        if stage == "logprobs":
            policy._shard_for_logprob(data)
        else:
            policy._shard_for_train(data, batch_size=2)


@pytest.mark.parametrize("mode", ["logprobs", "train"])
def test_non_tq_policy_rejects_shared_execution_before_allocating_workers(mode):
    with patch(
        "nemo_rl.models.policy.lm_policy.validate_shared_prefix_training_config",
        return_value=SharedPrefixTrainingConfig(mode=mode),
    ):
        # Raises before the cluster or tokenizer is touched.
        with pytest.raises(NotImplementedError, match="single-controller TQPolicy"):
            Policy(cluster=None, config={}, tokenizer=None)
    assert TQPolicy._supports_shared_prefix_execution


if __name__ == "__main__":
    unittest.main(verbosity=2)
