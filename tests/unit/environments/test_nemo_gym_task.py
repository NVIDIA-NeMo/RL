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
        {"task_id": "legacy-id", "task_input": {"custom": True}},
        {"task_id": {"taskset": ""}, "task_input": {}},
        {"task_id": {"taskset": "workplace:train"}},
    ],
)
def test_legacy_task_fields_do_not_change_wire_format(row):
    assert not is_nemo_gym_task(row)
    assert get_nemo_gym_task_input(row) is row


def test_native_task_input_returns_nested_object():
    row = {
        "task_id": {"taskset": "workplace:train", "task_id": "17"},
        "task_input": {
            "responses_create_params": {"input": "Find the next meeting."},
        },
    }

    assert is_nemo_gym_task(row)
    assert get_nemo_gym_task_input(row) is row["task_input"]


@pytest.mark.parametrize("task_input", [None, [], "serialized"])
def test_malformed_native_task_input_is_rejected(task_input):
    row = {
        "task_id": {"taskset": "workplace:train", "task_id": "17"},
        "task_input": task_input,
    }
    with pytest.raises(TypeError, match="materialized task_input must be a dict"):
        get_nemo_gym_task_input(row)


@pytest.mark.parametrize("native", [False, True])
def test_request_metadata_is_read_without_rewriting_envelope(native):
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
        if native
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
