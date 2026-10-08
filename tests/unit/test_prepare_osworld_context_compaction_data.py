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

import copy
import json

import pytest

from examples.nemo_gym import prepare_osworld_context_compaction_data as prepare
from examples.nemo_gym.prepare_osworld_stable_cc_split import (
    is_stable_eval_task,
    is_stable_train_task,
)


def _row(task_id: str = "task-1", **metadata):
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Complete the desktop task."}],
        },
        "verifier_metadata": {
            "id": task_id,
            "instruction": "Complete the desktop task.",
            **metadata,
        },
    }


def test_prepare_rows_adds_v2_routing_and_keeps_only_stable_provenance():
    source = _row("task-7", proxy=False, possibility_of_env_change="low")
    source.update(
        {
            "_ng_rollout_id": "stale-runtime-owner",
            "_ng_group_id": "stale-runtime-group",
            "context_compaction_contract_version": 2,
            "context_compaction_rollout_index": 0,
            "context_compaction_attempt_index": 0,
        }
    )
    original = copy.deepcopy(source)

    rows = prepare.prepare_osworld_rows(
        [source],
        num_repeats=2,
        agent_name="nemotron_osworld",
        max_output_tokens=2048,
        temperature=0.6,
        top_p=0.95,
    )

    assert source == original
    assert len(rows) == 2
    assert [row["context_compaction_group_id"] for row in rows] == [
        "osworld:task-7:repeat:0",
        "osworld:task-7:repeat:1",
    ]
    for row in rows:
        assert row["context_compaction_task_id"] == "task-7"
        assert row["verifier_metadata"] == original["verifier_metadata"]
        assert row["agent_ref"] == {
            "type": "responses_api_agents",
            "name": "nemotron_osworld",
        }
        assert row["responses_create_params"] == {
            "input": [
                {"role": "user", "content": "Complete the desktop task."},
            ],
            "max_output_tokens": 2048,
            "temperature": 0.6,
            "top_p": 0.95,
        }
        assert not any(key.startswith("_ng_") for key in row)
        assert "context_compaction_contract_version" not in row
        assert "context_compaction_rollout_index" not in row
        assert "context_compaction_attempt_index" not in row


def test_read_jsonl_filters_in_source_order_and_rejects_missing_ids(tmp_path):
    input_path = tmp_path / "osworld.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(_row(task_id)) for task_id in ("a", "b", "c")) + "\n",
        encoding="utf-8",
    )

    assert [
        prepare.task_id(row)
        for row in prepare.read_jsonl(
            input_path,
            selected_task_ids=["c", "a"],
        )
    ] == ["a", "c"]

    with pytest.raises(ValueError, match="Requested task IDs were not found: missing"):
        prepare.read_jsonl(input_path, selected_task_ids=["missing"])


def test_read_jsonl_rejects_duplicate_task_ids(tmp_path):
    input_path = tmp_path / "duplicates.jsonl"
    input_path.write_text(
        json.dumps(_row("same")) + "\n" + json.dumps(_row("same")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate task ID same"):
        prepare.read_jsonl(input_path)


@pytest.mark.parametrize(
    ("metadata", "stable_train", "stable_eval"),
    [
        (
            {"proxy": False, "possibility_of_env_change": "low"},
            True,
            True,
        ),
        (
            {"proxy": True, "possibility_of_env_change": "low"},
            False,
            True,
        ),
        (
            {"proxy": False, "possibility_of_env_change": "high"},
            False,
            False,
        ),
    ],
)
def test_stable_split_filters(metadata, stable_train, stable_eval):
    row = _row(**metadata)

    assert is_stable_train_task(row) is stable_train
    assert is_stable_eval_task(row) is stable_eval
