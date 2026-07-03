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

import json

import pytest

from tools.create_r2e_gym_easy_subset_jsonl import (
    R2E_COLUMNS,
    build_instance_id,
    convert_row,
    validate_converted_row,
)


def _source_row() -> dict:
    commit_hash = "a" * 40
    row = {column: "value" for column in R2E_COLUMNS}
    row.update(
        {
            "repo_name": "numpy",
            "docker_image": f"registry.example/numpy_final:{commit_hash}",
            "commit_hash": commit_hash,
            "parsed_commit_content": json.dumps(
                {
                    "new_commit_hash": commit_hash,
                    "old_commit_hash": f"{'b' * 40}^",
                    "commit_date": "2026-01-02T03:04:05Z",
                }
            ),
            "execution_result_content": json.dumps({"new_commit_hash": commit_hash}),
            "expected_output_json": json.dumps({"test_case": "PASSED"}),
            "problem_statement": "Fix the failing test.",
        }
    )
    return row


def test_convert_row_emits_valid_swe_agents_record() -> None:
    source_row = _source_row()
    converted = convert_row(
        source_row,
        row_number=1,
        row_index=0,
        split="train",
        model="default",
        temperature=1.0,
        top_p=1.0,
        agent_ref_name="swe_agents_train",
        base_commit=None,
        container_image_dir="/sif/r2egym",
        repo_formatter="/repos/{repo}",
    )

    validate_converted_row(converted, row_number=1)
    params = converted["responses_create_params"]
    metadata = params["metadata"]
    instance = json.loads(metadata["instance_dict"])

    assert metadata["instance_id"] == f"numpy__numpy-{'a' * 40}"
    assert metadata["base_commit"] == "b" * 40
    assert instance["container_formatter"].endswith(f"numpy_final_{'a' * 40}.sif")
    assert converted["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "swe_agents_train",
    }


def test_build_instance_id_rejects_non_sha_commit() -> None:
    with pytest.raises(ValueError, match="40-character lowercase SHA-1"):
        build_instance_id("numpy", "not-a-sha")
