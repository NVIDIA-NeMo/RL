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


class MetricNormalizer(enum.Enum):
    """Global denominator a loss-returned metric was normalized by.

    Losses reduce most metrics with ``masked_mean(...,
    global_normalization_factor=...)`` where the factor is the global valid
    *token* count, the global valid *sequence* count, or absent entirely (raw
    counts, per-microbatch means, extrema). Split-API trainers run each
    microbatch with placeholder ``global_valid_*=1`` (collecting raw sums) and
    rescale once per optimizer step — to do that they must know, per metric,
    which denominator applies.

    Losses advertise the mapping via a ``metric_normalizations:
    dict[str, MetricNormalizer]`` instance attribute, built in ``__init__``
    from the same flags that pick the denominators, so it lives next to the
    metric definitions instead of in a consumer-side table. Metrics absent
    from the mapping fall back to the gradient normalization (the
    ``loss_type`` denominator) on the consumer side.
    """

    TOKENS = "tokens"  # divided by global_valid_toks
    SEQUENCES = "sequences"  # divided by global_valid_seqs
    NONE = "none"  # not normalized: raw counts, local means, min/max


class LossInputType(enum.Enum):
    LOGIT = "logit"
    LOGPROB = "logprob"
    DISTILLATION = "distillation"
    DISTILLATION_CROSS_TOKENIZER = "distillation_cross_tokenizer"
    DRAFT = "draft"


class LossFunction(Protocol):
    """Signature for loss functions used in reinforcement learning algorithms.

    Loss functions compute a scalar loss value and associated metrics from
    model logprobs and other data contained in a BatchedDataDict.

    Losses may additionally expose a ``metric_normalizations:
    dict[str, MetricNormalizer]`` attribute advertising the global denominator
    each returned metric was normalized by (see ``MetricNormalizer``). It is
    optional: consumers fall back to the ``loss_type`` denominator for
    metrics (or losses) that do not advertise.
    """

    loss_type: LossType
    input_type: LossInputType

    def __call__(
        self,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss and metrics from logprobs and other data.

        Args:
            data: Dictionary containing all relevant data for loss computation
                  such as rewards, values, actions, advantages, masks, and other
                  algorithm-specific information needed for the particular loss calculation.
            global_valid_seqs: torch.Tensor
                This tensor should contain the number of valid sequences in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the sequence level
                and needs to be aggregated across all microbatches.
            global_valid_toks: torch.Tensor
                This tensor should contain the number of valid tokens in the microbatch.
                It's used for global normalization for losses/metrics that are computed at the token level
                and needs to be aggregated across all microbatches.
            **kwargs: Loss function input, which varies by input_type:
                - For LossInputType.LOGPROB: next_token_logprobs (torch.Tensor)
                - For LossInputType.LOGIT: logits (torch.Tensor)
                - For LossInputType.DISTILLATION: student_topk_logprobs, teacher_topk_logprobs, H_all (torch.Tensor)
                - For LossInputType.DISTILLATION_CROSS_TOKENIZER: logits (torch.Tensor), teacher_full_logits_by_idx (dict[int, torch.Tensor])
                - For LossInputType.DRAFT: teacher_logits, student_logits, mask (torch.Tensor)

        Returns:
            tuple: (loss, metrics)
                - loss: A scalar tensor representing the loss value to be minimized during training
                - metrics: A dictionary of metrics related to the loss computation, which may include
                  component losses, statistics about gradients/rewards, and other diagnostic information
        """
        ...
