# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public asset preparation must preserve the recorded tasks and input files."""

import json

import pytest
from nano35.build_swe_sifs import read_data
from nano35.prepare_swe_data import make_rows, write_matching


def source_row(instance_id: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "base_commit": "abc123",
        "problem_statement": "Fix the failure",
        "patch": "gold patch",
    }


def test_recorded_order_and_evaluation_metadata_are_preserved():
    rows = make_rows([source_row("first"), source_row("second")], ["second", "first"])
    assert [row["instance_id"] for row in rows] == ["second", "first"]
    for row in rows:
        assert row["agent_ref"]["name"] == "swe_agents_train"
        assert row["responses_create_params"]["input"] == []
        metadata = row["responses_create_params"]["metadata"]
        assert metadata["golden_patch"] == row["patch"]
        assert (
            json.loads(metadata["instance_dict"])["base_commit"] == row["base_commit"]
        )


def test_missing_task_is_not_silently_dropped():
    with pytest.raises(ValueError, match="Missing public instances"):
        make_rows([source_row("first")], ["first", "missing"])


@pytest.mark.parametrize("duplicate_source", [True, False])
def test_duplicate_tasks_are_rejected(duplicate_source):
    sources = [source_row("first")] * (2 if duplicate_source else 1)
    selected = ["first"] * (1 if duplicate_source else 2)
    with pytest.raises(ValueError, match="Duplicate instance IDs"):
        make_rows(sources, selected)


def test_preparation_reuses_identical_files_without_overwriting(tmp_path):
    output = tmp_path / "data.jsonl"
    write_matching(output, b"original\n")
    write_matching(output, b"original\n")
    with pytest.raises(ValueError, match="Existing file differs"):
        write_matching(output, b"changed\n")
    assert output.read_bytes() == b"original\n"


def test_sif_builder_rejects_a_partial_dataset(tmp_path):
    data = tmp_path / "partial.jsonl"
    data.write_text(json.dumps(source_row("first")) + "\n")
    with pytest.raises(ValueError, match="Raw data checksum differs"):
        read_data(data)
