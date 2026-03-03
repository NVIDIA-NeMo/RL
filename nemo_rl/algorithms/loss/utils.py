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

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    get_distillation_topk_logprobs_from_logits,
    get_next_token_logprobs_from_logits,
)
from nemo_rl.models.policy.utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)


def _mask_out_neg_inf_logprobs(
    logprobs: torch.Tensor, mask: torch.Tensor, logprobs_name: str
) -> torch.Tensor:
    """Mask out negative infinity log probabilities.

    Handling sampling mask mismatch:
    vLLM samples token X from top-k/p filtered distribution -> generation_logprobs[X] is always finite (e.g., -5.41)
    during training: policy computes logprobs with same top-k/p settings, but the distribution can be slightly different
    token X may fall outside the training policy's top-k/p set -> curr_logprobs[X] = -inf, prev_logprobs[X] = -inf
    Detect positions with -inf in any logprobs (generation_logprobs is always finite for valid tokens)

    Args:
        logprobs: Log probabilities.
        mask: Mask.

    Returns:
        Masked log probabilities.
    """
    is_neginf = torch.isinf(logprobs)
    neginf_count = (is_neginf & mask.bool()).sum().item()
    if neginf_count > 0:
        print(
            f"[WARNING]: {neginf_count}/{int(mask.sum().item())} valid tokens have -inf in {logprobs_name} "
            "(policy top-k/top-p mismatch). Masking out these positions."
        )

    mask = mask * (~is_neginf).float()
    logprobs = torch.where(mask.bool(), logprobs, 0.0)

    return logprobs


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
        if sampling_params is not None and need_top_k_or_top_p_filtering(
            sampling_params.top_k, sampling_params.top_p
        ):
            # mask out negative infinity logprobs
            mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
            logprobs = _mask_out_neg_inf_logprobs(
                logprobs, mask[:, 1:], "curr_logprobs"
            )
            data["prev_logprobs"] = _mask_out_neg_inf_logprobs(
                data["prev_logprobs"], mask, "prev_logprobs"
            )

            # currently only used for ClippedPGLossFn
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
