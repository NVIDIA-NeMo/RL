# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from nemo_rl.experience.loop_penalties import (
    LoopPenaltyConfig,
    apply_loop_reward_penalties,
    detect_output_loops,
)

LINE_A = (
    "Let me reconsider this implementation and inspect the exact same function again."
)
LINE_B = "Actually, the issue still requires the same change and I should reconsider it again."


def thought(n):
    return {"type": "reasoning", "summary": [{"text": "\n".join([LINE_A, LINE_B] * n)}]}


def call(name, arguments):
    return {"type": "function_call", "name": name, "arguments": json.dumps(arguments)}


def test_defaults_preserve_rewards_and_do_not_add_evidence():
    result = {"full_result": {"reward": 1.0, "response": {"output": [thought(30)]}}}
    assert apply_loop_reward_penalties([result], LoopPenaltyConfig()) == {}
    assert result["full_result"]["reward"] == 1.0
    assert "loop_penalties" not in result["full_result"]


@pytest.mark.parametrize("n,hit", [(19, False), (20, True)])
def test_reasoning_threshold_within_one_output(n, hit):
    config = LoopPenaltyConfig(penalize_reasoning_loops=True)
    assert detect_output_loops([thought(n)], config)["reasoning_loop"] is hit


@pytest.mark.parametrize("delimiter", ["```", "~~~"])
def test_fenced_code_is_excluded_from_reasoning_repetition(delimiter):
    config = LoopPenaltyConfig(
        penalize_reasoning_loops=True, reasoning_loop_min_repeats=3
    )
    repeated = "\n".join([LINE_A, LINE_B] * 3)
    text = f"{delimiter}python\n{repeated}\n{delimiter}\n"
    assert not detect_output_loops([{"type": "reasoning", "summary": text}], config)[
        "reasoning_loop"
    ]
    assert detect_output_loops(
        [{"type": "reasoning", "summary": text + repeated}], config
    )["reasoning_loop"]


def test_quoted_reasoning_is_excluded():
    config = LoopPenaltyConfig(
        penalize_reasoning_loops=True, reasoning_loop_min_repeats=3
    )
    text = "\n".join([f"> {LINE_A}", f"> {LINE_B}"] * 3)
    assert not detect_output_loops([{"type": "reasoning", "summary": text}], config)[
        "reasoning_loop"
    ]


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_three_cycle_recipe_boundary(period):
    config = LoopPenaltyConfig(
        penalize_tool_call_loops=True,
        tool_call_loop_min_calls=3,
        tool_call_loop_min_cycles=3,
        tool_call_loop_max_period=4,
    )
    output = [
        call("bash", {"command": f"read file{i % period}"}) for i in range(3 * period)
    ]
    assert detect_output_loops(output, config)["tool_call_loop"]
    assert not detect_output_loops(output[:-1], config)["tool_call_loop"]


def test_reasoning_does_not_accumulate_across_turns_or_tool_outputs():
    config = LoopPenaltyConfig(penalize_reasoning_loops=True)
    output = [thought(1) for _ in range(30)]
    output.append(
        {"type": "function_call_output", "output": "\n".join([LINE_A, LINE_B] * 30)}
    )
    assert not detect_output_loops(output, config)["reasoning_loop"]


def test_one_repeated_line_does_not_meet_two_distinct_line_threshold():
    config = LoopPenaltyConfig(penalize_reasoning_loops=True)
    item = {"type": "reasoning", "summary": [{"text": "\n".join([LINE_A] * 30)}]}
    assert not detect_output_loops([item], config)["reasoning_loop"]


def test_unparsed_thinking_tags_but_not_final_answer_are_checked():
    config = LoopPenaltyConfig(penalize_reasoning_loops=True)
    text = "\n".join([LINE_A, LINE_B] * 20)
    assert detect_output_loops(
        [{"type": "message", "content": f"<think>{text}</think>"}], config
    )["reasoning_loop"]
    assert not detect_output_loops([{"type": "message", "content": text}], config)[
        "reasoning_loop"
    ]


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_contiguous_tool_cycles_with_interleaved_results(period):
    config = LoopPenaltyConfig(penalize_tool_call_loops=True)
    output = []
    for i in range(12):
        output.extend(
            [
                call("bash", {"command": f"read file{i % period}"}),
                {"type": "function_call_output", "output": "not inspected"},
            ]
        )
    assert detect_output_loops(output, config)["tool_call_loop"]
    assert not detect_output_loops(output[:-2], config)["tool_call_loop"]


def test_interruption_and_changed_arguments_prevent_a_cycle():
    config = LoopPenaltyConfig(penalize_tool_call_loops=True)
    output = [call("bash", {"command": "test"}) for _ in range(6)]
    output += [call("edit", {"filePath": "source.py", "content": "fix"})]
    output += [call("bash", {"command": "test"}) for _ in range(6)]
    assert not detect_output_loops(output, config)["tool_call_loop"]
    assert not detect_output_loops(
        [call("read", {"offset": i}) for i in range(30)], config
    )["tool_call_loop"]


def test_argument_key_order_is_canonicalized_but_full_arguments_matter():
    config = LoopPenaltyConfig(penalize_tool_call_loops=True)
    output = [call("bash", {"a": 1, "b": 2}), call("bash", {"b": 2, "a": 1})] * 6
    assert detect_output_loops(output, config)["tool_call_loop"]
    changed = [
        call("bash", {"command": "test", "description": str(i)}) for i in range(12)
    ]
    assert not detect_output_loops(changed, config)["tool_call_loop"]


def test_receipt_reward_zeroing_is_per_rollout_and_idempotent():
    config = LoopPenaltyConfig(penalize_reasoning_loops=True)
    hit = {
        "message_log": [],
        "full_result": {"reward": 1.0, "response": {"output": [thought(20)]}},
    }
    miss = {
        "message_log": [],
        "full_result": {"reward": 1.0, "response": {"output": [thought(1)]}},
    }
    counts = apply_loop_reward_penalties([hit, miss], config)
    assert hit["full_result"]["reward"] == 0.0
    assert miss["full_result"]["reward"] == 1.0
    assert counts == {"reasoning_loop": 1, "tool_call_loop": 0, "loop_reward_zeroed": 1}
    assert apply_loop_reward_penalties([hit], config)["loop_reward_zeroed"] == 1
    assert hit["full_result"]["reward_before_loop_penalties"] == 1.0


def test_original_zero_reward_is_not_reported_as_a_zeroing_event():
    config = LoopPenaltyConfig(penalize_reasoning_loops=True)
    result = {"full_result": {"reward": 0.0, "response": {"output": [thought(20)]}}}
    for _ in range(2):
        counts = apply_loop_reward_penalties([result], config)
        assert counts["reasoning_loop"] == 1
        assert counts["loop_reward_zeroed"] == 0
    assert result["full_result"]["reward_before_loop_penalties"] == 0.0
    assert result["full_result"]["loop_reward_zeroed"] is False


def test_invalid_thresholds_fail_at_config_load():
    with pytest.raises(ValidationError):
        LoopPenaltyConfig(reasoning_loop_min_repeats=1)
