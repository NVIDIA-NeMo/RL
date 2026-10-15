# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = (
    ROOT / "scripts" / "agentic_vbench" / "debug_subagents_sync_8n.sh",
    ROOT / "scripts" / "agentic_vbench" / "debug_subagents_async_16n.sh",
)


def test_subagent_launchers_require_and_validate_solution_schema() -> None:
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        assert "top-level JSON object with a segments list and never a bare JSON list" in text
        assert (
            "use Python to assert solution.json parses as a top-level JSON object "
            "with a segments list containing only objects"
        ) in text


def test_subagent_launchers_bound_observations_without_changing_training_horizon() -> None:
    for launcher in LAUNCHERS:
        text = launcher.read_text()
        assert "AGENTIC_VBENCH_TOOL_OUTPUT_MAX_LINES:-400" in text
        assert "AGENTIC_VBENCH_TOOL_OUTPUT_MAX_BYTES:-8192" in text
        assert "exactly two labeled overview JPEGs total" in text
        assert "at most 1280 by 720 pixels" in text
        assert "ffmpeg -hide_banner -loglevel error -nostats" in text
        assert "MAX_STEPS:-1000000" in text
