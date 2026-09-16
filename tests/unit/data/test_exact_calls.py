# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations

import pytest
import torch

from nemo_rl.data.exact_calls import (
    _build_exact_call_tree,
    _tensor_parts_common_prefix_length,
)


def _call(tokens: list[int], routes: list[int]) -> list[dict]:
    metadata = {
        "ng_generation_replica_id": "replica-0",
        "ng_generation_weight_version": 0,
        "ng_generation_weight_version_end": 0,
        "ng_kv_cache_scheduler_block_size": 2,
        "ng_kv_cache_hash_block_size": 2,
        "ng_kv_cache_num_cached_tokens": 2,
    }
    return [
        {
            "role": "user",
            "token_ids": torch.tensor(tokens[:-1]),
            "routed_experts": torch.tensor(routes[:-1]).reshape(-1, 1, 1),
            **metadata,
        },
        {
            "role": "assistant",
            "token_ids": torch.tensor(tokens[-1:]),
            "generation_logprobs": torch.tensor([-0.5]),
            "routed_experts": torch.tensor(routes[-1:]).reshape(-1, 1, 1),
            **metadata,
        },
    ]


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_terminal_alias_cannot_merge_conflicting_executed_routes(order) -> None:
    calls = [
        _call([1, 2, 3], [0, 0, -1]),
        _call([1, 2, 3, 4], [0, 0, 1, -1]),
        _call([1, 2, 3, 5], [0, 0, 2, -1]),
    ]
    result = _build_exact_call_tree([calls[index] for index in order])
    tokens = torch.cat([message["token_ids"] for message in result.unique_message_log])
    routes = torch.cat(
        [message["routed_experts"] for message in result.unique_message_log]
    )
    targets = torch.cat(
        [message["token_ids"] for message in result.edge_message_log[1:]]
    )
    sources = torch.tensor(result.layout.edge_source_indices)

    # The two executions of token 3 need separate physical nodes, while the
    # original sample of token 3 still contributes just one loss edge.
    assert tokens.numel() == 6
    assert targets.tolist() == [[3, 4, 5][index] for index in order]
    assert tokens[sources].tolist() == [[2, 3, 3][index] for index in order]
    assert routes[sources].flatten().tolist() == [[0, 1, 2][index] for index in order]
    assert (
        sum(
            message["generation_logprobs"].numel()
            for message in result.edge_message_log[1:]
        )
        == 3
    )


def test_terminal_promotion_keeps_compatible_descendants_shared() -> None:
    result = _build_exact_call_tree(
        [
            _call([1, 2, 3], [0, 0, -1]),
            _call([1, 2, 3, 4], [0, 0, 1, -1]),
            _call([1, 2, 3, 5], [0, 0, 1, -1]),
        ]
    )

    assert result.layout.unique_token_count == 5
    assert result.layout.edge_source_indices == (1, 2, 2)


def test_common_prefix_ignores_empty_message_parts() -> None:
    assert (
        _tensor_parts_common_prefix_length(
            [torch.tensor([], dtype=torch.long), torch.tensor([1, 2])],
            [torch.tensor([1]), torch.tensor([], dtype=torch.long), torch.tensor([3])],
        )
        == 1
    )
