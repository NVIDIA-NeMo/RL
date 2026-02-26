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
from typing import Any

import torch

from nemo_rl.algorithms.interfaces import LossInputType, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class SimpleLoss:
    loss_type = LossType.SEQUENCE_LEVEL
    input_type = LossInputType.LOGIT

    def __call__(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Just return mean of logits as the loss for testing
        loss = logits.mean()
        metrics = {
            "loss": loss.item(),
            "test_metric": loss.item() * 0.5,
            "num_valid_samples": 1,
        }
        return loss, metrics


# Create a simple masked NLL loss function
class SimpleNLLLoss:
    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __call__(
        self,
        next_token_logprobs: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Only compute loss on generated tokens (not input tokens)
        # by applying the token_mask (shifted by 1 since we're predicting next tokens)
        mask = data["token_mask"][:, 1:].cuda()
        loss = -torch.sum(next_token_logprobs * mask)

        return loss, {
            "loss": loss.item(),
            "num_valid_samples": 1,
        }
