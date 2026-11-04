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

"""Tests for vLLM worker helper functions."""

import pytest

from nemo_rl.models.generation.vllm.worker_utils import (
    find_tokenizer_required_architectures,
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
    resolve_vllm_dp_group,
    resolve_vllm_dp_settings,
    resolve_vllm_dp_size,
    resolve_vllm_rank_prefixes,
)


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (None, []),
        ([], []),
        (["Gemma4ForCausalLM"], []),
        (
            ["Gemma4ForConditionalGeneration"],
            ["Gemma4ForConditionalGeneration"],
        ),
        (
            [
                "Gemma4ForCausalLM",
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
            [
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
        ),
        (
            ["NemotronH_Omni_Reasoning_V3"],
            ["NemotronH_Omni_Reasoning_V3"],
        ),
    ],
)
def test_find_tokenizer_required_architectures(architectures, expected):
    assert find_tokenizer_required_architectures(architectures) == expected


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(("total_workers", "tp", "ep"), [(8, 1, 4), (32, 2, 16)])
def test_resolve_vllm_rank_prefixes(total_workers, tp, ep):
    vllm_dp_size = resolve_vllm_dp_size(tp, ep)
    dp_size = total_workers // tp
    prefixes = resolve_vllm_rank_prefixes(total_workers, dp_size, vllm_dp_size)
    rollout_ranks = sorted(
        prefixes[dp_rank] + dp_rank % vllm_dp_size * tp + tp_rank
        for dp_rank in range(dp_size)
        for tp_rank in range(tp)
    )
    assert rollout_ranks == list(range(total_workers))


@pytest.mark.parametrize(
    ("total_workers", "dp_size", "vllm_dp_size"),
    [(0, 8, 4), (-8, 8, 4), (8, 0, 4), (7, 8, 4), (8, 3, 2), (8, 8, 0)],
)
def test_resolve_vllm_rank_prefixes_rejects_invalid_topology(
    total_workers, dp_size, vllm_dp_size
):
    with pytest.raises(ValueError, match="Invalid vLLM DP worker topology"):
        resolve_vllm_rank_prefixes(total_workers, dp_size, vllm_dp_size)


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )


@pytest.mark.parametrize(
    ("rank", "tp", "pp", "ep", "expected"),
    [
        (7, 1, 1, 8, (8, 7, 0)),
        (14, 2, 1, 16, (8, 7, 0)),
        (18, 2, 1, 16, (8, 1, 16)),
        (9, 2, 2, 8, (4, 2, 0)),
        (17, 2, 2, 8, (4, 0, 16)),
        (6, 2, 1, 4, (2, 1, 4)),
    ],
)
def test_resolve_vllm_dp_group(rank, tp, pp, ep, expected):
    assert (
        resolve_vllm_dp_group(
            rank=rank,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            expert_parallel_size=ep,
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "rank",
        "tp",
        "pp",
        "ep",
        "async_engine",
        "backend",
        "expected_env",
        "expected_kwargs",
    ),
    [
        (
            7,
            1,
            1,
            8,
            False,
            "uni",
            {
                "VLLM_DP_SIZE": "8",
                "VLLM_DP_RANK": "7",
                "VLLM_DP_RANK_LOCAL": "0",
                "VLLM_DP_MASTER_IP": "host0",
                "VLLM_DP_MASTER_PORT": "3000",
            },
            {},
        ),
        (
            6,
            2,
            1,
            4,
            False,
            "ray",
            {
                "VLLM_DP_SIZE": "2",
                "VLLM_DP_RANK": "1",
                "VLLM_DP_RANK_LOCAL": "1",
                "VLLM_DP_MASTER_IP": "host4",
                "VLLM_DP_MASTER_PORT": "3004",
            },
            {},
        ),
        (
            18,
            2,
            1,
            16,
            True,
            "ray",
            {},
            {
                "data_parallel_size": 8,
                "data_parallel_rank": 1,
                "data_parallel_external_lb": True,
                "data_parallel_address": "host16",
                "data_parallel_rpc_port": 3016,
            },
        ),
    ],
)
def test_resolve_vllm_dp_settings(
    rank, tp, pp, ep, async_engine, backend, expected_env, expected_kwargs
):
    env, kwargs = resolve_vllm_dp_settings(
        rank=rank,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        addresses=[f"host{index}" for index in range(32)],
        ports=[3000 + index for index in range(32)],
        async_engine=async_engine,
        executor_backend=backend,
    )
    assert env == expected_env
    assert kwargs == expected_kwargs
