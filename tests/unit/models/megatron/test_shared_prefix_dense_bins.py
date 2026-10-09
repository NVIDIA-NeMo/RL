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
"""Coverage, causal reconstruction and normalization checks for dense-bin sharing."""

import unittest

import numpy as np
import pytest
import torch
from pydantic import ValidationError

# Module-level megatron imports would break COLLECTION on non-mcore CI shards
# (marks only deselect at run time); skip collection gracefully instead.
pytest.importorskip("megatron.core")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.rl.shared_prefix_dense_bins")

from megatron.rl.shared_prefix_alignment import materialize_alignment
from megatron.rl.shared_prefix_execution import SharedPrefixExecutionUnit

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.data import plan_shared_prefix_execution_units
from nemo_rl.models.megatron.shared_prefix_dense_bins import (
    plan_dense_training_bins,
    share_prefixes_in_dense_training_bins,
)
from nemo_rl.models.policy import (
    SharedPrefixTrainingConfig,
    validate_shared_prefix_training_config,
)

pytestmark = pytest.mark.mcore


def config(multiple=1):
    return {
        "make_sequence_length_divisible_by": multiple,
        "sequence_packing": {"algorithm": "modified_first_fit_decreasing"},
    }


def batch(tokens, prompts, groups):
    lengths = [len(row) for row in tokens]
    width = (max(lengths) + 15) // 16 * 16
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in tokens]
            ),
            "input_lengths": torch.tensor(lengths),
            "shared_prefix_prompt_lengths": torch.tensor(prompts),
            "shared_prefix_group_id": np.asarray(groups, dtype=object),
        }
    )


class TestSharedPrefixDenseBins(unittest.TestCase):
    def test_public_plan_rejects_empty_batch(self):
        data = BatchedDataDict(
            {
                "input_ids": torch.empty((0, 1), dtype=torch.long),
                "input_lengths": torch.empty(0, dtype=torch.long),
                "shared_prefix_prompt_lengths": torch.empty(0, dtype=torch.long),
                "shared_prefix_group_id": [],
                "_shared_prefix_execution_slot": [],
            }
        )
        with self.assertRaisesRegex(
            ValueError, "^shared-prefix train mode received an empty local batch$"
        ):
            plan_shared_prefix_execution_units(data, cfg=config(), bin_capacity=1)

    def test_mixed_roots_preserve_causal_rows_and_one_mtp_group(self):
        # Multi-token, unequal completions: predecessor maps are per token.
        data = batch(
            [[1, 2, 3, 8, 11], [4, 5, 6, 9, 12, 14], [1, 2, 3, 10]],
            [3, 3, 3],
            ["a", "b", "a"],
        )
        original = SharedPrefixExecutionUnit((0, 1, 2), None, 15)
        (unit,) = share_prefixes_in_dense_training_bins(
            data, [original], cfg=config(), bin_capacity=15
        )
        forest = unit.shared_layout
        assert forest is not None and forest.mtp_loss_group_root_counts == (2,)
        assert unit.physical_length == 12 and sorted(unit.row_indices) == [0, 1, 2]
        physical = data["input_ids"][
            list(forest.token_gather_rows), list(forest.token_gather_columns)
        ]
        for offset, root in forest.iter_roots():
            for index, row in enumerate(root.row_indices):
                start = offset + root.branch_starts[index]
                restored = torch.cat(
                    (
                        physical[offset : offset + root.prompt_length],
                        physical[start : start + root.completion_lengths[index]],
                    )
                )
                assert torch.equal(
                    restored, data["input_ids"][row, : data["input_lengths"][row]]
                )
                for token in range(root.completion_lengths[index]):
                    k = forest.completion_positions.index(start + token)
                    assert forest.predecessor_positions[k] == (
                        start + token - 1 if token else offset + root.prompt_length - 1
                    )

    def _test_prefix_or_group_mismatch_remains_dense(self, tokens, groups):
        data = batch(tokens, [3, 3], groups)
        original = SharedPrefixExecutionUnit((0, 1), None, 8)
        assert share_prefixes_in_dense_training_bins(
            data, [original], cfg=config(), bin_capacity=8
        ) == (original,)

    def test_prefix_or_group_mismatch_remains_dense_0(self):
        self._test_prefix_or_group_mismatch_remains_dense(
            [[1, 2, 3, 8], [1, 2, 4, 9]], ["a", "a"]
        )

    def test_prefix_or_group_mismatch_remains_dense_1(self):
        self._test_prefix_or_group_mismatch_remains_dense(
            [[1, 2, 3, 8], [1, 2, 3, 9]], ["a", "b"]
        )

    def _test_empty_prompt_or_completion_retains_every_row(self, prompts):
        data = batch([[1, 2, 3, 8], [1, 2, 3, 9]], prompts, ["a", "a"])
        original = SharedPrefixExecutionUnit((0, 1), None, 8)
        assert share_prefixes_in_dense_training_bins(
            data, [original], cfg=config(), bin_capacity=8
        ) == (original,)

    def test_empty_prompt_or_completion_retains_every_row_0(self):
        self._test_empty_prompt_or_completion_retains_every_row([0, 3])

    def test_empty_prompt_or_completion_retains_every_row_1(self):
        self._test_empty_prompt_or_completion_retains_every_row([4, 3])

    def test_alignment_precedes_sharing_and_preserves_expanded_budget(self):
        data = batch([[1, 2, 3, 8 + i] for i in range(6)], [3] * 6, ["a"] * 6)
        cfg = config(4)
        units = plan_dense_training_bins(data, cfg=cfg, bin_capacity=12)
        assert len(units) == 2
        aligned, _ = materialize_alignment(
            units, costs=[4] * 6, capacity=12, target_count=3
        )
        shared = share_prefixes_in_dense_training_bins(
            data, aligned, cfg=cfg, bin_capacity=12
        )
        assert len(shared) == 3
        assert sorted((row for unit in shared for row in unit.row_indices)) == list(
            range(6)
        )
        for original, unit in zip(aligned, shared, strict=True):
            assert sorted(unit.row_indices) == sorted(original.row_indices)
            assert (unit.physical_length + 3) // 4 * 4 <= original.physical_length <= 12
            if unit.shared_layout is not None:
                assert unit.shared_layout.mtp_loss_group_root_counts == (
                    len(unit.shared_layout.roots),
                )

    def test_invalid_coverage_and_capacity_fail_closed(self):
        data = batch([[1, 2, 3, 8], [1, 2, 3, 9]], [3, 3], ["a", "a"])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            share_prefixes_in_dense_training_bins(
                data,
                [SharedPrefixExecutionUnit((0, 0), None, 8)],
                cfg=config(),
                bin_capacity=8,
            )
        with self.assertRaisesRegex(ValueError, "expanded token budget"):
            plan_dense_training_bins(data, cfg=config(), bin_capacity=3)

    def test_many_siblings_keep_branch_limit_and_single_normalization_group(self):
        data = batch([[1, 2, 3, 8 + i] for i in range(17)], [3] * 17, ["a"] * 17)
        unit = SharedPrefixExecutionUnit(tuple(range(17)), None, 68)
        (shared,) = share_prefixes_in_dense_training_bins(
            data, [unit], cfg=config(), bin_capacity=68
        )
        assert shared.shared_layout is not None
        assert [len(root.row_indices) for root in shared.shared_layout.roots] == [16, 1]
        assert shared.shared_layout.mtp_loss_group_root_counts == (2,)

    def test_flag_is_strict_and_disabled_by_default(self):
        assert SharedPrefixTrainingConfig().training_dense_bins is False
        with self.assertRaises(ValidationError):
            SharedPrefixTrainingConfig(training_dense_bins="true")

    def _test_flag_requires_training_and_aligned_group_execution(self, field, value):
        shared = dict(
            mode="train",
            training_dense_bins=True,
            pack_groups=True,
            repack_groups=True,
            align_data_parallel=True,
        )
        shared[field] = value
        with self.assertRaisesRegex(ValueError, "training_dense_bins requires"):
            validate_shared_prefix_training_config({"shared_prefix_training": shared})

    def test_flag_requires_training_and_aligned_group_execution_0(self):
        self._test_flag_requires_training_and_aligned_group_execution(
            "mode", "logprobs"
        )

    def test_flag_requires_training_and_aligned_group_execution_1(self):
        self._test_flag_requires_training_and_aligned_group_execution(
            "pack_groups", False
        )

    def test_flag_requires_training_and_aligned_group_execution_2(self):
        self._test_flag_requires_training_and_aligned_group_execution(
            "repack_groups", False
        )

    def test_flag_requires_training_and_aligned_group_execution_3(self):
        self._test_flag_requires_training_and_aligned_group_execution(
            "align_data_parallel", False
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
