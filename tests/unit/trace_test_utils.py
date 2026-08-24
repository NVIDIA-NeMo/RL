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

"""Compact builders for exact physical-trace unit tests."""

from __future__ import annotations

from typing import Any

from nemo_rl.environments.nemo_gym_trace import build_rollout_trace_plan


def _call(
    *,
    rollout_id: str,
    turn_id: int,
    prompt_token_ids: list[int],
    sampled_token_id: int,
    segment_index: int,
    expected_append_compatible: bool,
    compaction_event_id: str | None = None,
) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "prompt_token_ids": prompt_token_ids,
        "sampled_token_ids": [sampled_token_id],
        "sampled_logprobs": [-0.01 * turn_id],
        "media_ids": [],
        "completion_id": f"completion-{turn_id}",
        "rollout_id": rollout_id,
        "action_id": f"action-{turn_id}",
        "prepared_request_id": f"prepared-{turn_id}",
        "request_id": f"request-{turn_id}",
        "context_epoch": segment_index,
        "segment_index": segment_index,
        "segment_id": f"{rollout_id}:segment-{segment_index}",
        "expected_append_compatible": expected_append_compatible,
        "compaction_event_id": compaction_event_id,
        "finish_reason": "stop",
        "eligible": True,
        "evidence_source": "unit_test",
    }


def trace_bundle(*, compacted: bool) -> dict[str, Any]:
    """Build a five-turn identity or K=2 compacted rollout bundle."""
    if compacted:
        rollout_id = "computer-use-k2"
        group_id = "golden-k2"
        policy_name = "recency"
        prompt_tokens = [
            [100, 101],
            [100, 101, 201, 102],
            [300, 301, 302],
            [300, 301, 302, 203, 103],
            [400, 401],
        ]
        segment_indices = [0, 0, 1, 1, 2]
        compaction_events = [None, None, "boundary-3", None, "boundary-5"]
        boundaries = [
            {
                "event_id": f"boundary-{turn_id}",
                "applies_to_step": turn_id,
                "reason": "history_policy_rewrite",
                "policy_name": policy_name,
                "policy_version": "1",
                "config_digest": "test-policy-config",
            }
            for turn_id in (3, 5)
        ]
    else:
        rollout_id = "computer-use-without-cc"
        group_id = "golden-without-cc"
        policy_name = "identity"
        prompt_tokens = [
            [100, 101],
            [100, 101, 201, 102],
            [100, 101, 201, 102, 202, 103],
            [100, 101, 201, 102, 202, 103, 203, 104],
            [100, 101, 201, 102, 202, 103, 203, 104, 204, 105],
        ]
        segment_indices = [0] * 5
        compaction_events = [None] * 5
        boundaries = []

    calls = [
        _call(
            rollout_id=rollout_id,
            turn_id=turn_id,
            prompt_token_ids=prompt_tokens[turn_id - 1],
            sampled_token_id=200 + turn_id,
            segment_index=segment_indices[turn_id - 1],
            expected_append_compatible=(
                turn_id > 1 and compaction_events[turn_id - 1] is None
            ),
            compaction_event_id=compaction_events[turn_id - 1],
        )
        for turn_id in range(1, 6)
    ]
    plan = build_rollout_trace_plan(
        rollout_id=rollout_id,
        calls=calls,
        boundary_events=boundaries,
        media_assets={},
        strict=True,
    )
    physical_traces = []
    previous_context: list[int] = []
    for call, call_plan in zip(calls, plan["model_calls"], strict=True):
        if call_plan["starts_physical_trace"]:
            physical_traces.append(
                {
                    "trace_id": call_plan["trace_id"],
                    "trace_index": len(physical_traces),
                    "segments": [],
                }
            )
            previous_context = []
        prompt_delta = call["prompt_token_ids"][len(previous_context) :]
        eligible = bool(call["eligible"])
        physical_traces[-1]["segments"].extend(
            [
                {
                    "kind": "prompt",
                    "token_ids": prompt_delta,
                    "loss_mask": [0] * len(prompt_delta),
                },
                {
                    "kind": "completion",
                    "token_ids": call["sampled_token_ids"],
                    "loss_mask": [int(eligible)] * len(call["sampled_token_ids"]),
                    "generation_logprobs": call["sampled_logprobs"],
                },
            ]
        )
        previous_context = [
            *call["prompt_token_ids"],
            *call["sampled_token_ids"],
        ]
    return {
        "rollout_id": rollout_id,
        "group_id": group_id,
        "reward": 1.0,
        "policy_name": policy_name,
        "generation_policy_version": "test-policy-version",
        "physical_traces": physical_traces,
    }
