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

from collections.abc import Iterable

TOKENIZER_REQUIRED_ARCHITECTURES = frozenset(
    {
        "Gemma3ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
        "Mistral3ForConditionalGeneration",
        "NemotronH_Omni_Reasoning_V3",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)


def find_tokenizer_required_architectures(
    architectures: Iterable[str] | None,
) -> list[str]:
    """Return architectures for which vLLM must initialize a tokenizer."""
    return [
        architecture
        for architecture in architectures or ()
        if architecture in TOKENIZER_REQUIRED_ARCHITECTURES
    ]


def resolve_distributed_executor_backend(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> str | None:
    if tensor_parallel_size * pipeline_parallel_size > 1:
        return "ray"
    if resolve_vllm_dp_size(tensor_parallel_size, expert_parallel_size) > 1:
        # External DP actors already own one GPU each.
        return "uni"
    return None


def resolve_vllm_dp_size(tensor_parallel_size: int, expert_parallel_size: int) -> int:
    """Return the number of DP ranks in one vLLM instance."""
    if expert_parallel_size <= tensor_parallel_size:
        return 1
    return expert_parallel_size // tensor_parallel_size


def resolve_vllm_rank_prefixes(
    total_workers: int, dp_size: int, vllm_dp_size: int
) -> list[int]:
    """Return the first rollout worker rank of each vLLM instance."""
    if (
        total_workers <= 0
        or dp_size <= 0
        or vllm_dp_size <= 0
        or total_workers % dp_size != 0
        or dp_size % vllm_dp_size != 0
    ):
        raise ValueError("Invalid vLLM DP worker topology")
    workers_per_instance = total_workers // dp_size * vllm_dp_size
    return [
        dp_rank // vllm_dp_size * workers_per_instance for dp_rank in range(dp_size)
    ]


def resolve_data_parallel_local_rank(
    rank: int, model_parallel_size: int, executor_backend: str | None
) -> int:
    # Ray remaps one GPU into each external-DP actor.
    if executor_backend == "uni":
        return 0
    return (rank % 8) // model_parallel_size


def resolve_vllm_dp_group(
    *,
    rank: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
) -> tuple[int, int, int]:
    """Return the vLLM DP size, DP rank, and global leader rank for a worker."""
    vllm_dp_size = resolve_vllm_dp_size(tensor_parallel_size, expert_parallel_size)
    group_size = vllm_dp_size * tensor_parallel_size * pipeline_parallel_size
    rank_in_group = rank % group_size
    return (
        vllm_dp_size,
        rank_in_group // (tensor_parallel_size * pipeline_parallel_size),
        rank - rank_in_group,
    )


def resolve_vllm_dp_settings(
    *,
    rank: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    addresses: list[str],
    ports: list[int],
    async_engine: bool,
    executor_backend: str | None,
) -> tuple[dict[str, str], dict[str, int | str | bool]]:
    """Build the vLLM DP settings for one externally managed worker."""
    vllm_dp_size, vllm_dp_rank, leader_rank = resolve_vllm_dp_group(
        rank=rank,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
    )
    address = addresses[leader_rank]
    port = ports[leader_rank]
    if async_engine:
        return {}, {
            "data_parallel_size": vllm_dp_size,
            "data_parallel_rank": vllm_dp_rank,
            "data_parallel_external_lb": True,
            "data_parallel_address": address,
            "data_parallel_rpc_port": port,
        }
    local_rank = resolve_data_parallel_local_rank(
        rank - leader_rank,
        tensor_parallel_size * pipeline_parallel_size,
        executor_backend,
    )
    return {
        "VLLM_DP_SIZE": str(vllm_dp_size),
        "VLLM_DP_RANK": str(vllm_dp_rank),
        "VLLM_DP_RANK_LOCAL": str(local_rank),
        "VLLM_DP_MASTER_IP": address,
        "VLLM_DP_MASTER_PORT": str(port),
    }, {}
