# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consume image-tool execution provenance without changing other agent protocols."""


def was_image_tool_call_executed(
    output_item: dict, request_row: dict, result: dict
) -> bool:
    """Recognize only this requested agent's server-recorded executed turn IDs.

    Provenance comes from the agent response, never from input-row metadata or
    model-authored text. Missing provenance retains the generic detector.
    """
    agent_ref = request_row.get("agent_ref")
    if (
        not isinstance(agent_ref, dict)
        or agent_ref.get("name") != "image_tools_simple_agent"
    ):
        return False
    if output_item.get("type") != "message" or output_item.get("role") != "assistant":
        return False
    output_id = output_item.get("id")
    executed_ids = result.get("image_tools_executed_output_ids")
    return (
        isinstance(output_id, str)
        and isinstance(executed_ids, list)
        and output_id in executed_ids
    )
