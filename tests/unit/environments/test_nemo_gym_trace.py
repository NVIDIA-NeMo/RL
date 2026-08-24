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

import pytest

from nemo_rl.environments.nemo_gym_trace import build_rollout_trace_plan


def _call(
    turn_id: int,
    *,
    prompt_token_ids: list[int],
    sampled_token_ids: list[int],
    media_ids: list[str],
    segment_index: int,
    append_compatible: bool,
    boundary_id: str | None = None,
) -> dict:
    return {
        "turn_id": turn_id,
        "completion_id": f"completion-{turn_id}",
        "action_id": f"action-{turn_id}",
        "rollout_id": "rollout-1",
        "prompt_token_ids": prompt_token_ids,
        "sampled_token_ids": sampled_token_ids,
        "sampled_logprobs": [-0.1] * len(sampled_token_ids),
        "media_ids": media_ids,
        "context_epoch": segment_index,
        "segment_index": segment_index,
        "segment_id": f"segment-{segment_index}",
        "expected_append_compatible": append_compatible,
        "compaction_event_id": boundary_id,
        "finish_reason": "stop",
        "eligible": True,
    }


def test_identity_plan_is_lightweight_and_append_compatible():
    plan = build_rollout_trace_plan(
        rollout_id="rollout-1",
        calls=[
            _call(
                1,
                prompt_token_ids=[1],
                sampled_token_ids=[2],
                media_ids=[],
                segment_index=0,
                append_compatible=False,
            ),
            _call(
                2,
                prompt_token_ids=[1, 2, 3],
                sampled_token_ids=[4],
                media_ids=[],
                segment_index=0,
                append_compatible=True,
            ),
        ],
        media_assets={},
        strict=True,
    )

    assert plan["physical_trace_ids"] == ["rollout-1:trace-000000"]
    assert [call["starts_physical_trace"] for call in plan["model_calls"]] == [
        True,
        False,
    ]
    assert set(plan["model_calls"][1]) == {
        "trace_id",
        "starts_physical_trace",
        "new_media_ids",
        "finish_reason",
        "eligible",
    }


def test_declared_media_rewrite_starts_a_physical_trace():
    boundary = {"event_id": "boundary-2", "applies_to_step": 2}
    plan = build_rollout_trace_plan(
        rollout_id="rollout-1",
        calls=[
            _call(
                1,
                prompt_token_ids=[1],
                sampled_token_ids=[2],
                media_ids=["image-a"],
                segment_index=0,
                append_compatible=False,
            ),
            _call(
                2,
                prompt_token_ids=[1, 2, 3],
                sampled_token_ids=[4],
                media_ids=["image-b"],
                segment_index=1,
                append_compatible=False,
                boundary_id="boundary-2",
            ),
        ],
        boundary_events=[boundary],
        media_assets={"image-a": {}, "image-b": {}},
        strict=True,
    )

    assert plan["physical_trace_ids"] == [
        "rollout-1:trace-000000",
        "rollout-1:trace-000001",
    ]
    assert plan["model_calls"][1]["new_media_ids"] == ["image-b"]
    assert plan["checks"]["declared_boundary_count"] == 1


def test_strict_plan_rejects_undeclared_rewrite():
    with pytest.raises(ValueError, match="has no boundary record"):
        build_rollout_trace_plan(
            rollout_id="rollout-1",
            calls=[
                _call(
                    1,
                    prompt_token_ids=[1],
                    sampled_token_ids=[2],
                    media_ids=[],
                    segment_index=0,
                    append_compatible=False,
                ),
                _call(
                    2,
                    prompt_token_ids=[9],
                    sampled_token_ids=[10],
                    media_ids=[],
                    segment_index=1,
                    append_compatible=False,
                    boundary_id="boundary-2",
                ),
            ],
            media_assets={},
            strict=True,
        )


def test_legacy_plan_marks_undeclared_rewrite_for_adapter_rejection():
    plan = build_rollout_trace_plan(
        rollout_id="rollout-1",
        calls=[
            {
                "completion_id": "completion-1",
                "prompt_token_ids": [1],
                "sampled_token_ids": [2],
                "sampled_logprobs": [-0.1],
            },
            {
                "completion_id": "completion-2",
                "prompt_token_ids": [9],
                "sampled_token_ids": [10],
                "sampled_logprobs": [-0.2],
            },
        ],
    )

    assert plan["checks"]["inferred_boundary_count"] == 1
