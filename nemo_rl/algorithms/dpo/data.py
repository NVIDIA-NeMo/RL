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
"""DPO data transforms and preference batch processing.

This module provides functions for preparing preference pairs for DPO training,
including reference policy log probability computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.policy.interfaces import PolicyInterface


def prepare_preference_batch(
    batch: "BatchedDataDict",
    policy: "PolicyInterface",
    micro_batch_size: int,
    dp_size: int,
) -> "BatchedDataDict":
    """Prepare a preference batch with reference policy log probabilities.

    This function adds reference policy log probabilities to a preference batch,
    which is required for DPO loss computation.

    Args:
        batch: BatchedDataDict containing preference pairs.
        policy: Policy with get_reference_policy_logprobs method.
        micro_batch_size: Micro batch size for reference logprob computation.
        dp_size: Data parallel size.

    Returns:
        BatchedDataDict with added reference_policy_logprobs key.
    """
    from nemo_rl.algorithms.utils import maybe_pad_last_batch

    # Pad batch if needed for DPO's paired processing
    if batch.size % (dp_size * micro_batch_size) != 0:
        batch = maybe_pad_last_batch(batch, dp_size, micro_batch_size)

    # Compute reference policy logprobs
    logprobs = policy.get_reference_policy_logprobs(
        batch, micro_batch_size=micro_batch_size
    )["reference_logprobs"]

    # Roll logprobs to align with next-token prediction
    batch["reference_policy_logprobs"] = torch.roll(logprobs, -1, dims=-1)

    return batch


def add_ref_logprobs_to_batch(
    batch: "BatchedDataDict",
    policy: "PolicyInterface",
    config: dict[str, Any],
    is_val: bool = False,
) -> "BatchedDataDict":
    """Add reference policy logprobs to a batch.

    Convenience wrapper around prepare_preference_batch.

    Args:
        batch: BatchedDataDict containing preference data.
        policy: Policy with reference model.
        config: Configuration dict with micro batch sizes.
        is_val: Whether this is for validation.

    Returns:
        BatchedDataDict with reference logprobs added.
    """
    micro_batch_size = (
        config.get("dpo", {}).get("val_micro_batch_size", 1) * 2
        if is_val
        else config.get("policy", {}).get("train_micro_batch_size", 1) * 2
    )
    dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
    return prepare_preference_batch(batch, policy, micro_batch_size, dp_size)


def iterate_with_ref_logprobs(
    dataloader: Iterator,
    policy: "PolicyInterface",
    config: dict[str, Any],
    is_val: bool = False,
) -> Iterator["BatchedDataDict"]:
    """Iterate over dataloader, adding reference logprobs to each batch.

    Args:
        dataloader: Data iterator yielding batches.
        policy: Policy with reference model.
        config: Configuration dict.
        is_val: Whether this is for validation.

    Yields:
        BatchedDataDict with reference logprobs added.
    """
    for batch in dataloader:
        yield add_ref_logprobs_to_batch(batch, policy, config, is_val)
