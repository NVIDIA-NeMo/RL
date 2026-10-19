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
from copy import deepcopy

import pytest

from nemo_rl.environments.nemo_gym_request import (
    _chat_template_kwargs_for_processor,
    _metadata_extra_body,
)
from nemo_rl.environments.nemo_gym_task import (
    get_nemo_gym_task_input,
    is_nemo_gym_task,
)


@pytest.mark.parametrize(
    "row",
    [
        {"responses_create_params": {}},
        {"task_id": "flat-id", "task_input": {"custom": True}},
        {"task_id": {"taskset": "workplace:train"}},
    ],
)
def test_nonmaterialized_task_fields_preserve_flat_row(row):
    assert not is_nemo_gym_task(row)
    assert get_nemo_gym_task_input(row) is row


@pytest.mark.parametrize("taskset", ["workplace:train", "", None])
def test_materialized_task_input_returns_nested_object(taskset):
    row = {
        "task_id": {"taskset": taskset, "task_id": "17"},
        "task_input": {
            "responses_create_params": {"input": "Find the next meeting."},
        },
    }

    assert is_nemo_gym_task(row)
    assert get_nemo_gym_task_input(row) is row["task_input"]


@pytest.mark.parametrize("task_input", [None, [], "serialized"])
def test_malformed_materialized_task_input_is_rejected(task_input):
    row = {
        "task_id": {"taskset": "workplace:train", "task_id": "17"},
        "task_input": task_input,
    }
    with pytest.raises(TypeError, match="materialized task_input must be a dict"):
        get_nemo_gym_task_input(row)


@pytest.mark.parametrize("materialized", [False, True], ids=["flat", "materialized"])
def test_request_metadata_is_read_without_rewriting_envelope(materialized):
    task_input = {
        "responses_create_params": {
            "metadata": {
                "extra_body": '{"seed": 7, "chat_template_kwargs": {"enable_thinking": true}}',
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
        "task_data": {"expected_answer": "opaque"},
    }
    row = (
        {
            "task_id": {"taskset": "math:train", "task_id": "4"},
            "task_input": task_input,
        }
        if materialized
        else task_input
    )
    original = deepcopy(row)

    assert _metadata_extra_body(row) == {
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert _chat_template_kwargs_for_processor(row) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
    }
    assert row == original
