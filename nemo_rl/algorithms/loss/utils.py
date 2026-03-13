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

from typing import Any, Optional

import torch

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    get_distillation_topk_logprobs_from_logits,
    get_next_token_logprobs_from_logits,
)


def prepare_loss_input(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: LossFunction,
    vocab_parallel_rank: Optional[int] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
) -> tuple[dict[str, Any], BatchedDataDict[Any]]:
    """Prepare loss input for a loss function.

    Args:
        logits: Logits from the model.
        data: Microbatch data. Will be updated if sampling_params is not None.
        loss_fn: Loss function.
        vocab_parallel_rank: Vocab parallel rank.
        vocab_parallel_group: Vocab parallel group.
        context_parallel_group: Context parallel group.
        sampling_params: Sampling parameters.

    Notes:
        vocab_parallel_rank, vocab_parallel_group, context_parallel_group are only used for megatron policy worker.
        sampling_params is only used for LossInputType.LOGPROB, and currently only supported for ClippedPGLossFn.

    Returns:
        tuple(loss_input, maybe_updated_data)
    """
    if loss_fn.input_type == LossInputType.LOGIT:
        loss_input = {"logits": logits}

    elif loss_fn.input_type == LossInputType.LOGPROB:
        # Linear CE fusion patch returns precomputed next-token logprobs (2D tensor).
        # Keep normal path unchanged for standard logits (3D tensor).
        if logits.ndim == 2:
            logprobs = logits
            logprobs = logprobs.to(torch.float32)
            logprobs = logprobs[:, : data["input_ids"].shape[1] - 1]
        else:
            logprobs = get_next_token_logprobs_from_logits(
                input_ids=data["input_ids"],
                next_token_logits=logits,
                seq_index=data.get("seq_index", None),
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
                sampling_params=sampling_params,
            )

        # handle top-k/top-p filtering for logprobs, only used for ClippedPGLossFn now
        if need_top_k_or_top_p_filtering(sampling_params):
            # mask out negative infinity logprobs
            # prev_logprobs is already masked out in the previous step
            mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
            logprobs = mask_out_neg_inf_logprobs(logprobs, mask[:, 1:], "curr_logprobs")

            # compute unfiltered logprobs for reference policy KL penalty
            if (
                hasattr(loss_fn, "reference_policy_kl_penalty")
                and loss_fn.reference_policy_kl_penalty != 0
            ):
                data["curr_logprobs_unfiltered"] = get_next_token_logprobs_from_logits(
                    input_ids=data["input_ids"],
                    next_token_logits=logits,
                    seq_index=data.get("seq_index", None),
                    vocab_parallel_rank=vocab_parallel_rank,
                    vocab_parallel_group=vocab_parallel_group,
                    context_parallel_group=context_parallel_group,
                    sampling_params=None,  # no filtering
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
                vocab_parallel_rank=vocab_parallel_rank,
                vocab_parallel_group=vocab_parallel_group,
                context_parallel_group=context_parallel_group,
            )
        )

        loss_input = {
            "student_topk_logprobs": student_topk_logprobs,
            "teacher_topk_logprobs": teacher_topk_logprobs,
            "H_all": H_all,
        }

    else:
        raise ValueError(f"Unknown loss function input type: {loss_fn.input_type}")

    return loss_input, data
