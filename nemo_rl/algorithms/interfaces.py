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

import enum
from typing import Any, Protocol

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class LossType(enum.Enum):
    TOKEN_LEVEL = "token_level"
    SEQUENCE_LEVEL = "sequence_level"


class LossFunction(Protocol):
    """Signature for loss functions used in reinforcement learning algorithms.

    Loss functions compute a scalar loss value and associated metrics from
    model logprobs and other data contained in a BatchedDataDict.
    """

    loss_type: LossType

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss and metrics from logprobs and other data.

        Args:
            next_token_logits: Logits from the model, typically with shape [batch_size, seq_len, vocab_size].
                               For each position (b, i), contains the logit distribution over the entire vocabulary
                               for predicting the next token (at position i+1). For example, if processing "The cat sat on",
                               then next_token_logits[b, 3] would contain the logits for predicting the word
                               that follows "on".
            data: Dictionary containing all relevant data for loss computation
                  such as rewards, values, actions, advantages, masks, and other
                  algorithm-specific information needed for the particular loss calculation.
            global_valid_seqs: torch.Tensor
                this tensor should contain the number of valid sequences in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the sequence level
                and needs to be aggregated across all microbatches.
            global_valid_toks: torch.Tensor
                This tensor should contain the number of valid tokens in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the token level
                and needs to be aggregated across all microbatches.

        Returns:
            tuple: (loss, metrics)
                - loss: A scalar tensor representing the loss value to be minimized during training
                - metrics: A dictionary of metrics related to the loss computation, which may include
                  component losses, statistics about gradients/rewards, and other diagnostic information
        """
        pass
