# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from tools.super_rl.prepare_regular_data import prepare


def inputs(tmp_path):
    row = {
        "metadata": {"curriculum": {"ordinal": 0, "task_sha256": "task"}},
        "agent_ref": {"name": "ns_tools_simple_agent"},
        "responses_create_params": {
            "input": [{"role": "user", "content": "Original prompt"}],
            "max_output_tokens": 131072,
        },
    }
    profile = row | {
        "reasoning_effort": "low",
        "reasoning_effort_profile": {
            "profile_length_cap": 131072,
            "baseline_output_tokens": 20000,
        },
    }
    source, profiled = tmp_path / "regular.jsonl", tmp_path / "kimi.jsonl"
    for path, value in [(source, row), (profiled, profile)]:
        path.write_text(json.dumps(value) + "\n")
    return (
        row,
        source,
        profiled,
        [hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, profiled)],
    )


def test_cap_preserves_prompt_and_historical_profile_without_effort_shaping(tmp_path):
    row, source, profile, hashes = inputs(tmp_path)
    output = tmp_path / "prepared.jsonl"
    receipt = prepare(source, profile, output, *hashes, 102400, 1)
    actual = json.loads(output.read_text())
    assert (
        actual["responses_create_params"]["input"]
        == row["responses_create_params"]["input"]
    )
    assert actual["responses_create_params"]["max_output_tokens"] == 102400
    assert actual["reasoning_effort_profile"]["profile_length_cap"] == 131072
    assert "reasoning_effort" not in actual
    assert receipt["rows"] == 1
    assert receipt["effort_shaping"] is False


def test_wrong_checksum_does_not_publish(tmp_path):
    _, source, profile, hashes = inputs(tmp_path)
    output = tmp_path / "prepared.jsonl"
    with pytest.raises(ValueError, match="checksum"):
        prepare(source, profile, output, "0" * 64, hashes[1], 102400, 1)
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    _, source, profile, hashes = inputs(tmp_path)
    output = tmp_path / "prepared.jsonl"
    output.write_text("user-owned")
    with pytest.raises(FileExistsError):
        prepare(source, profile, output, *hashes, 102400, 1)
    assert output.read_text() == "user-owned"


def test_changed_profile_order_is_rejected(tmp_path):
    _, source, profile, hashes = inputs(tmp_path)
    record = json.loads(profile.read_text())
    record["metadata"]["curriculum"]["ordinal"] = 1
    profile.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="identity/order"):
        prepare(source, profile, tmp_path / "prepared.jsonl", *hashes, 102400, 1)
