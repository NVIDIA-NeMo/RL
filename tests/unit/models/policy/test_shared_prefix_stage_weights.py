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
from unittest.mock import patch

import torch
from pydantic import ValidationError

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

    def test_lm_policy_uses_stage_weights(self):
        cfg = SharedPrefixTrainingConfig(
            mode="train", shard_work_weights=(1, 1), training_shard_work_weights=(0, 1)
        )
        policy = SimpleNamespace(
            shared_prefix_training_config=cfg,
            data_parallel_size=2,
            cfg={"make_sequence_length_divisible_by": 1},
        )
        data = BatchedDataDict(
            {
                "input_lengths": torch.tensor([10, 10, 8, 8]),
                "shared_prefix_group_id": ["a", "a", "b", "b"],
                "shared_prefix_prompt_lengths": torch.tensor([8, 8, 1, 1]),
                "row_id": torch.arange(4),
            }
        )
        from nemo_rl.models.policy import lm_policy

        original = lm_policy.estimate_shared_prefix_row_work
        for stage, weights in [("train", (0, 1)), ("logprobs", (1, 1))]:
            with patch.object(
                lm_policy, "estimate_shared_prefix_row_work", wraps=original
            ) as estimator:
                shards, _ = Policy._shard_shared_prefix_data(
                    policy, data, bin_capacity=32, stage=stage
                )
                self.assertEqual(
                    estimator.call_args.kwargs["physical_weight"], weights[0]
                )
                self.assertEqual(
                    estimator.call_args.kwargs["expanded_weight"], weights[1]
                )
                self.assertEqual(len(shards), 2)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
