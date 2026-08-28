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

from __future__ import annotations

import torch

from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.payload import pack_payload, record_to_train_batch


def _routes(start: int, count: int) -> torch.Tensor:
    token_routes = torch.arange(start, start + count, dtype=torch.int16).view(
        count, 1, 1
    )
    topk_offsets = torch.arange(2, dtype=torch.int16).view(1, 1, 2)
    return (token_routes + topk_offsets).expand(count, 2, 2).contiguous()


def _fallback_routes(count: int) -> torch.Tensor:
    return torch.arange(2, dtype=torch.int16).view(1, 1, 2).expand(count, 2, 2)


def _completion(
    route_start: int,
    reward: float,
    *,
    env_token_ids: tuple[int, ...] = (30,),
    with_routes: bool = True,
) -> Completion:
    message_log = [
        {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([10, 11]),
            "routed_experts": _routes(route_start, 2),
        },
        {
            "role": "assistant",
            "content": "answer",
            "token_ids": torch.tensor([20, 21]),
            "generation_logprobs": torch.tensor([-0.1, -0.2]),
            "routed_experts": _routes(route_start + 2, 2),
        },
        {
            "role": "user",
            "content": "environment",
            "token_ids": torch.tensor(env_token_ids),
            "routed_experts": _fallback_routes(len(env_token_ids)),
        },
    ]
    if not with_routes:
        for message in message_log:
            message.pop("routed_experts")
    return Completion(
        message_log=message_log,
        env_extras=None,
        truncated=False,
        reward=reward,
    )


def _record(completions: list[Completion]) -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        extra_env_info=None,
        metadata={"task_name": "test"},
        completions=completions,
        rollout_metrics={},
    )


def test_record_to_train_batch_preserves_routed_experts_in_tq_payload() -> None:
    record = _record(
        [
            _completion(route_start=10, reward=1.0),
            _completion(
                route_start=30,
                reward=2.0,
                env_token_ids=(30, 31),
            ),
        ]
    )

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    expected_routes = [
        torch.cat((_routes(10, 4), _fallback_routes(1))),
        torch.cat((_routes(30, 4), _fallback_routes(2))),
    ]
    assert train_batch["input_lengths"].tolist() == [5, 6]
    assert train_batch["routed_experts"].shape == (2, 6, 2, 2)
    assert torch.equal(
        train_batch["routed_experts"][0, :5],
        expected_routes[0],
    )
    assert torch.equal(
        train_batch["routed_experts"][1],
        expected_routes[1],
    )

    sample_ids, fields, tags = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert sample_ids == ["group_g0", "group_g1"]
    assert "routed_experts" in fields
    packed_routes = fields["routed_experts"]
    assert packed_routes.is_nested
    packed_rows = list(packed_routes.unbind())
    assert torch.equal(packed_rows[0], expected_routes[0])
    assert torch.equal(packed_rows[1], expected_routes[1])
    no_violations = {
        "num_invalid_tool_calls": 0,
        "num_malformed_thinking": 0,
        "num_assistant_messages": 1,
    }
    assert tags == [
        {"weight_version": 3, **no_violations},
        {"weight_version": 3, **no_violations},
    ]


def test_record_to_train_batch_omits_routed_experts_when_absent() -> None:
    record = _record([_completion(route_start=10, reward=1.0, with_routes=False)])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )
    assert "routed_experts" not in train_batch

    _, fields, _ = pack_payload(
        train_batch,
        weight_version=3,
        group_id="group",
    )
    assert "routed_experts" not in fields


def _failed_completion() -> Completion:
    """A trajectory whose first generation raised: prompt only, no routes."""
    return Completion(
        message_log=[
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11]),
            }
        ],
        env_extras=None,
        truncated=False,
        reward=0.0,
    )


def test_record_to_train_batch_backfills_routes_for_failed_completion() -> None:
    """A group is packable when only some completions generated (and so have routes)."""
    record = _record([_completion(route_start=10, reward=1.0), _failed_completion()])

    train_batch = record_to_train_batch(
        record,
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )

    assert train_batch["input_lengths"].tolist() == [5, 2]
    routes = train_batch["routed_experts"]
    assert routes.shape == (2, 5, 2, 2)
    assert torch.equal(routes[0, :5], torch.cat((_routes(10, 4), _fallback_routes(1))))
    # The completion that never generated gets the all--1 missing-route sentinel,
    # so Megatron routes those tokens with its own router.
    assert torch.equal(routes[1, :2], torch.full((2, 2, 2), -1, dtype=routes.dtype))
    # It is fully loss-masked either way.
    assert train_batch["token_mask"][1, :2].tolist() == [0, 0]

    _, fields, _ = pack_payload(train_batch, weight_version=3, group_id="group")
    assert "routed_experts" in fields
    assert list(fields["routed_experts"].unbind())[1].shape == (2, 2, 2)


def test_pack_payload_stamps_violation_counts_on_tags() -> None:
    """Each flag lands in its own counter; a row that never generated counts zero."""
    completions = [
        _completion(route_start=10, reward=1.0),
        _completion(route_start=30, reward=1.0),
        _failed_completion(),
    ]
    completions[0].message_log[1]["is_invalid_tool_call"] = True
    completions[1].message_log[1]["has_malformed_thinking"] = True

    train_batch = record_to_train_batch(
        _record(completions),
        pad_value_dict={"token_ids": 0, "input_ids": 0},
    )
    _, fields, tags = pack_payload(train_batch, weight_version=7, group_id="g")

    assert "violation_counts" not in fields
    assert tags == [
        {
            "weight_version": 7,
            "num_invalid_tool_calls": 1,
            "num_malformed_thinking": 0,
            "num_assistant_messages": 1,
        },
        {
            "weight_version": 7,
            "num_invalid_tool_calls": 0,
            "num_malformed_thinking": 1,
            "num_assistant_messages": 1,
        },
        {
            "weight_version": 7,
            "num_invalid_tool_calls": 0,
            "num_malformed_thinking": 0,
            "num_assistant_messages": 0,
        },
    ]
