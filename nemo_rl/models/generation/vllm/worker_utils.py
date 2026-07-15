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


def generated_token_logprob(
    logprob_dict: dict[int, Any], token_id: int
) -> float | None:
    logprob = logprob_dict.get(token_id)
    return None if logprob is None else float(logprob.logprob)


def resolve_distributed_executor_backend(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> str | None:
    if tensor_parallel_size * pipeline_parallel_size > 1:
        return "ray"
    if expert_parallel_size > tensor_parallel_size:
        # External DP actors already own one GPU each.
        return "uni"
    return None


def resolve_data_parallel_local_rank(
    rank: int, model_parallel_size: int, executor_backend: str | None
) -> int:
    # Ray remaps one GPU into each external-DP actor.
    if executor_backend == "uni":
        return 0
    return (rank % 8) // model_parallel_size
