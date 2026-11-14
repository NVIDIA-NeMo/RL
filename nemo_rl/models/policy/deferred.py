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
from dataclasses import dataclass
from typing import Any

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@dataclass(frozen=True)
class DeferredTopkLogits:
    """Ray references to DP-local teacher top-k results.

    Keeping the large tensors behind object references avoids materializing them in
    the driver before the corresponding student worker consumes them.
    """

    refs: list[ray.ObjectRef]
    global_indices_per_dp: list[list[int]]
    batch_size: int


@dataclass(frozen=True)
class DeferredTopkWorkerResult:
    """Small worker response containing an object-store reference to top-k data."""

    payload_ref: ray.ObjectRef


def attach_deferred_topk_logits(
    data: BatchedDataDict[Any],
    teacher_result: dict[str, torch.Tensor] | BatchedDataDict[Any] | None,
) -> None:
    """Attach one resolved teacher result to its DP-local student batch."""
    if teacher_result is None:
        return
    if "teacher_topk_logits" in data or "teacher_topk_indices" in data:
        raise ValueError(
            "Deferred teacher top-k results cannot be combined with materialized "
            "teacher_topk_logits/teacher_topk_indices fields."
        )

    topk_logits = teacher_result["topk_logits"]
    topk_indices = teacher_result["topk_indices"]
    if topk_logits.shape != topk_indices.shape:
        raise ValueError(
            "Teacher top-k logits/indices shape mismatch: "
            f"{topk_logits.shape} vs {topk_indices.shape}."
        )
    if topk_logits.shape[0] != data.size:
        raise ValueError(
            "Teacher top-k result and student DP shard have different batch sizes: "
            f"{topk_logits.shape[0]} vs {data.size}."
        )

    data["teacher_topk_logits"] = topk_logits
    data["teacher_topk_indices"] = topk_indices
