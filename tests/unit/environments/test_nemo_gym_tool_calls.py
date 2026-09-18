# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_rl.environments.nemo_gym_tool_calls import was_image_tool_call_executed


@pytest.mark.parametrize("agent", ["gym_v_agent", "visgym_agent", "simple_agent", None])
def test_other_agents_never_bypass_detection(agent):
    item = {"id": "turn1", "role": "assistant", "type": "message"}
    assert not was_image_tool_call_executed(
        item,
        {"agent_ref": {"name": agent}},
        {"image_tools_executed_output_ids": ["turn1"]},
    )


def test_provenance_is_turn_specific_and_cannot_come_from_the_row():
    row = {
        "agent_ref": {"name": "image_tools_simple_agent"},
        "image_tools_executed_output_ids": ["turn2"],
    }
    item = {"id": "turn1", "role": "assistant", "type": "message"}
    result = {"image_tools_executed_output_ids": ["turn1"]}
    assert was_image_tool_call_executed(item, row, result)
    assert not was_image_tool_call_executed({**item, "id": "turn2"}, row, result)
    assert not was_image_tool_call_executed(item, row, {})
    assert not was_image_tool_call_executed({**item, "type": "reasoning"}, row, result)
    assert not was_image_tool_call_executed({**item, "role": "user"}, row, result)
