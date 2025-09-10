# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock

import numpy as np
import torch

from nemo_rl.algorithms.dpo import add_ref_logprobs_to_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding


class MockPolicy:
    def __init__(self, logprobs):
        self.logprobs = logprobs
        self.sharding_annotations = NamedSharding(
            layout=np.arange(2).reshape(
                1,  # PP
                -1,  # DP
                1,  # CP
                1,  # TP
            ),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

    def get_reference_policy_logprobs(self, batch, micro_batch_size):
        return {"reference_logprobs": self.logprobs}


def test_add_logprobs_to_batch():
    """Test that add_ref_logprobs_to_data correctly adds reference policy logprobs to batches."""
    # Create mock data
    batch_size = 8
    seq_len = 4
    vocab_size = 16

    # Create a mock batch
    mock_batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }

    # Create mock logprobs that will be returned by the policy
    mock_logprobs = torch.randn(batch_size, seq_len)

    # Create a mock dataloader that yields our mock batch
    mock_dataloader = MagicMock()
    mock_dataloader.__iter__.return_value = iter([BatchedDataDict(mock_batch)])

    # Create a mock policy that returns our mock logprobs
    mock_policy = MockPolicy(mock_logprobs)

    # Create a mock master config
    mock_master_config = {"policy": {"train_micro_batch_size": 1}}

    # Get the augmented batches
    augmented_batches = list(
        add_ref_logprobs_to_data(mock_dataloader, mock_policy, mock_master_config)
    )

    # Verify we got exactly one batch
    assert len(augmented_batches) == 1
    augmented_batch = augmented_batches[0]

    # Verify the original batch data is preserved
    assert torch.equal(augmented_batch["input_ids"], mock_batch["input_ids"])
    assert torch.equal(augmented_batch["attention_mask"], mock_batch["attention_mask"])

    # Verify the reference policy logprobs were added correctly
    assert "reference_policy_logprobs" in augmented_batch
    assert augmented_batch["reference_policy_logprobs"].shape == (batch_size, seq_len)

    # Verify the logprobs were rolled by -1 as expected
    expected_logprobs = torch.roll(mock_logprobs, -1, dims=-1)
    assert torch.equal(augmented_batch["reference_policy_logprobs"], expected_logprobs)
