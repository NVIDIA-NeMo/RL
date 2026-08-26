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
"""``num_valid_samples`` must come from the sample mask, in every loss.

Workers gate on ``num_valid_samples > 0`` to decide whether a microbatch's
loss and metrics are recorded at all -- ``dtensor_policy_worker.py``,
``dtensor_policy_worker_v2.py`` and ``dtensor_value_worker_v2.py`` all do. A
loss that reports the raw batch dimension makes a fully-masked microbatch look
like it contributed, and its zero loss then dilutes the step's reported mean.

CPU-only: these construct the loss inputs directly rather than going through
``prepare_loss_input``, so no GPU and no distributed init.
"""

from __future__ import annotations

import torch

from nemo_rl.algorithms.loss.loss_functions import (
    DistillationLossConfig,
    DistillationLossFn,
    MseValueLossConfig,
    MseValueLossFn,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

B, S = 3, 4
# One live sample out of three. Everything below asserts on 1.0, not 3.
LIVE_SAMPLE_MASK = torch.tensor([1.0, 0.0, 0.0])
ALL_MASKED = torch.zeros(B)


def _value_call(sample_mask: torch.Tensor):
    values = torch.randn(B, S)
    data = BatchedDataDict(
        {
            "token_mask": torch.ones(B, S),
            "sample_mask": sample_mask,
            "returns": torch.randn(B, S),
            "values": torch.randn(B, S),
        }
    )
    gvs = sample_mask.sum().clamp(min=1.0)
    gvt = (data["token_mask"] * sample_mask.unsqueeze(-1)).sum().clamp(min=1.0)
    loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=None))
    return loss_fn(values, data, gvs, gvt)


def _distillation_call(sample_mask: torch.Tensor, k: int = 5):
    data = {
        "input_ids": torch.randint(0, 8, (B, S)),
        "token_mask": torch.ones(B, S),
        "sample_mask": sample_mask,
    }
    student = torch.randn(B, S - 1, k).log_softmax(-1)
    teacher = torch.randn(B, S - 1, k).log_softmax(-1)
    gvs = sample_mask.sum().clamp(min=1.0)
    gvt = (data["token_mask"] * sample_mask.unsqueeze(-1)).sum().clamp(min=1.0)
    loss_fn = DistillationLossFn(DistillationLossConfig(kl_type="forward"))
    return loss_fn(
        student_topk_logprobs=student,
        teacher_topk_logprobs=teacher,
        H_all=None,
        data=data,
        global_valid_seqs=gvs,
        global_valid_toks=gvt,
    )


class TestMseValueLossFn:
    def test_counts_the_mask_not_the_batch(self):
        _, metrics = _value_call(LIVE_SAMPLE_MASK)
        assert metrics["num_valid_samples"] == 1.0

    def test_a_fully_masked_microbatch_reports_zero(self):
        """The gate is ``num_valid_samples > 0``. Reporting the batch size here
        lets a microbatch that contributed nothing through, and its zero loss
        is then averaged into the step."""
        _, metrics = _value_call(ALL_MASKED)
        assert metrics["num_valid_samples"] == 0.0


class TestDistillationLossFn:
    def test_counts_the_mask_not_the_batch(self):
        _, metrics = _distillation_call(LIVE_SAMPLE_MASK)
        assert metrics["num_valid_samples"] == 1.0

    def test_a_fully_masked_microbatch_reports_zero(self):
        _, metrics = _distillation_call(ALL_MASKED)
        assert metrics["num_valid_samples"] == 0.0

    def test_without_a_mask_every_sample_is_valid(self):
        """The unmasked branch has nothing to count, so the batch dimension is
        the honest answer there.

        Written to hold whichever way this lands relative to #3496, which
        replaces that branch with a raise. Until it lands the fallback is live
        and must report the batch size; after it lands the branch is gone and
        the only correct behaviour is the raise. Asserting one of them
        unconditionally makes the two PRs fail as a pair while each passes
        alone -- which is exactly what happened when I merged the stack.
        """
        data = {"input_ids": torch.randint(0, 8, (B, S))}
        student = torch.randn(B, S - 1, 5).log_softmax(-1)
        teacher = torch.randn(B, S - 1, 5).log_softmax(-1)
        loss_fn = DistillationLossFn(DistillationLossConfig(kl_type="forward"))
        try:
            _, metrics = loss_fn(
                student_topk_logprobs=student,
                teacher_topk_logprobs=teacher,
                H_all=None,
                data=data,
                global_valid_seqs=torch.tensor(1.0),
                global_valid_toks=torch.tensor(1.0),
            )
        except ValueError as exc:
            assert "token_mask" in str(exc) and "sample_mask" in str(exc), exc
            return  # #3496 has landed: the branch this covers no longer exists
        assert metrics["num_valid_samples"] == B
