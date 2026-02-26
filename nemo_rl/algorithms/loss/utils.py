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

from typing import Any

import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    get_distillation_topk_logprobs_from_logits,
    get_next_token_logprobs_from_logits,
)


def prepare_loss_input(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
) -> dict[str, Any]:
    """Prepare loss input for a loss function.

    Args:
        logits: Logits from the model.
        data: Microbatch data.
        loss_fn: Loss function.

    Returns:
        Loss input.
    """
    if loss_fn.input_type == LossInputType.LOGIT:
        loss_input = {"logits": logits}

    elif loss_fn.input_type == LossInputType.LOGPROB:
        logprobs = get_next_token_logprobs_from_logits(
            input_ids=data["input_ids"],
            next_token_logits=logits,
            seq_index=data.get("seq_index", None),
        )

        loss_input = {"next_token_logprobs": logprobs}

    elif loss_fn.input_type == LossInputType.DISTILLATION:
        calculate_entropy = loss_fn.zero_outside_topk and loss_fn.kl_type != "forward"
        student_topk_logprobs, teacher_topk_logprobs, H_all = (
            get_distillation_topk_logprobs_from_logits(
                student_logits=logits,
                teacher_topk_logits=data["teacher_topk_logits"],
                teacher_topk_indices=data["teacher_topk_indices"],
                zero_outside_topk=loss_fn.zero_outside_topk,
                calculate_entropy=calculate_entropy,
            )
        )

        loss_input = {
            "student_topk_logprobs": student_topk_logprobs,
            "teacher_topk_logprobs": teacher_topk_logprobs,
            "H_all": H_all,
        }

    else:
        raise ValueError(f"Unknown loss function input type: {loss_fn.input_type}")

    return loss_input
