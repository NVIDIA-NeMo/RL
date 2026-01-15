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
"""SFT data transforms and batch processing.

This module provides functions for preparing batches for SFT training,
including tokenization, masking, and padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def prepare_batch_for_sft(
    batch: dict[str, Any],
    tokenizer: Any,
    make_sequence_length_divisible_by: int = 1,
    roles_to_train_on: list[str] | None = None,
) -> "BatchedDataDict":
    """Prepare a batch for SFT training.

    This function processes a batch of message logs into the format
    required for SFT training:
    1. Adds loss masks based on roles
    2. Concatenates and pads sequences
    3. Creates the BatchedDataDict with required keys

    Args:
        batch: Batch containing 'message_log' and 'loss_multiplier' keys.
        tokenizer: Tokenizer with pad_token_id attribute.
        make_sequence_length_divisible_by: Pad sequences to be divisible by this value.
        roles_to_train_on: List of roles to compute loss on. Defaults to ["assistant"].

    Returns:
        BatchedDataDict containing:
            - input_ids: Tokenized and padded sequences
            - input_lengths: Length of each sequence
            - token_mask: Mask for loss computation
            - sample_mask: Mask for valid samples

    Example:
        >>> batch = {"message_log": messages, "loss_multiplier": torch.ones(batch_size)}
        >>> train_data = prepare_batch_for_sft(batch, tokenizer)
    """
    from nemo_rl.data.llm_message_utils import (
        add_loss_mask_to_message_log,
        batched_message_log_to_flat_message,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    if roles_to_train_on is None:
        roles_to_train_on = ["assistant"]

    # Add loss mask based on role to every message
    add_loss_mask_to_message_log(
        batch["message_log"],
        roles_to_train_on=roles_to_train_on,
    )

    # Convert message logs to flat tensors
    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    # Create BatchedDataDict
    train_data: BatchedDataDict = BatchedDataDict(
        {
            "input_ids": cat_and_padded["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": cat_and_padded["token_loss_mask"],
            "sample_mask": batch["loss_multiplier"],
        }
    )

    # Add multimodal data if present
    train_data.update(cat_and_padded.get_multimodal_dict(as_tensors=False))

    return train_data
