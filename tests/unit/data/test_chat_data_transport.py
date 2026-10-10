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

"""Raw chat metadata must survive dataset loading and deferred tokenization."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nemo_rl.data.chat_utils import normalize_message_loss_mask
from nemo_rl.data.datasets.processed_dataset import AllTaskProcessedDataset
from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
    OpenAIFormatDataset,
    PreservingDataset,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import (
    PROCESSOR_REGISTRY,
    chat_kd_processor,
    kd_data_processor,
)


def _row() -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "First request"},
            {
                "role": "assistant",
                "content": "Earlier answer",
                "reasoning_content": "Earlier reasoning",
            },
            {"role": "user", "content": "Search again"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Latest reasoning",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "你好", "exact": True},
                        },
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        "message_loss_mask": [0, 0, 0, 1],
        "task_name": "chat_kd",
    }


@pytest.mark.parametrize("processor", [kd_data_processor, chat_kd_processor])
def test_processors_preserve_context_and_isolate_nested_metadata(processor) -> None:
    row = _row()
    original = deepcopy(row)
    result = processor(row, TaskDataSpec(), None, 1, 7)
    assert result["message_log"] == row["messages"]
    assert result["tools"] == row["tools"]
    assert result["message_loss_mask"] == [0, 0, 0, 1]
    assert result["idx"] == 7
    assert result["task_name"] == "chat_kd"
    assert all("token_ids" not in message for message in result["message_log"])

    # Neither unselected historical turns nor tool-only turns are discarded.
    assert result["message_log"][1]["reasoning_content"] == "Earlier reasoning"
    assert result["message_log"][3]["content"] is None
    result["message_log"][3]["tool_calls"][0]["function"]["arguments"]["query"] = "x"
    result["tools"][0]["function"]["parameters"]["properties"]["query"]["type"] = "x"
    result["message_loss_mask"][1] = 1
    assert row == original


def test_chat_processor_accepts_conversation_and_is_registered() -> None:
    row = _row()
    row["conversation"] = row.pop("messages")
    assert PROCESSOR_REGISTRY["chat_kd_processor"] is chat_kd_processor
    result = chat_kd_processor(row, TaskDataSpec(), None, 1, 0)
    assert result["message_log"] == row["conversation"]
    assert result["message_loss_mask"] == row["message_loss_mask"]


@pytest.mark.parametrize("messages", [None, "answer", ["answer"]])
def test_chat_processor_rejects_malformed_messages(messages) -> None:
    with pytest.raises(TypeError, match="list of message objects"):
        chat_kd_processor({"messages": messages}, TaskDataSpec(), None, None, 0)


def test_chat_processor_requires_conversation() -> None:
    with pytest.raises(KeyError, match="'messages' or 'conversation'"):
        chat_kd_processor({}, TaskDataSpec(), None, None, 0)


@pytest.mark.parametrize("processor", [kd_data_processor, chat_kd_processor])
@pytest.mark.parametrize("tools", [{"name": "search"}, [None], ["search"]])
def test_processors_reject_malformed_tools(processor, tools) -> None:
    with pytest.raises(TypeError, match="'tools' to be a list of objects"):
        processor({**_row(), "tools": tools}, TaskDataSpec(), None, None, 0)


@pytest.mark.parametrize(
    "mask,error,match",
    [
        ([0, 1], ValueError, "one entry per message"),
        ([0, 0, 0, 2], ValueError, "binary integers"),
        ([0, 0, 0, 1.0], ValueError, "binary integers"),
        ([0, 0, 0, "1"], ValueError, "binary integers"),
        ([0, 0, 1, 1], ValueError, "only assistant"),
        ("0001", TypeError, "list or tuple"),
    ],
)
@pytest.mark.parametrize("processor", [kd_data_processor, chat_kd_processor])
def test_processors_reject_invalid_loss_masks(processor, mask, error, match) -> None:
    with pytest.raises(error, match=match):
        processor({**_row(), "message_loss_mask": mask}, TaskDataSpec(), None, None, 0)


def test_mask_defaults_include_all_assistants_without_dropping_context() -> None:
    messages = _row()["messages"]
    assert normalize_message_loss_mask(messages, None) == [0, 1, 0, 1]
    assert normalize_message_loss_mask(messages, [0, 0, 0, 0]) == [0, 0, 0, 0]
    assert normalize_message_loss_mask(messages, (0, np.int64(1), 0, True)) == [
        0,
        1,
        0,
        1,
    ]


@pytest.mark.parametrize("role", ["user", "system", "tool"])
def test_mask_cannot_select_non_assistant_roles(role: str) -> None:
    with pytest.raises(ValueError, match="only assistant"):
        normalize_message_loss_mask([{"role": role, "content": "context"}], [1])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("preserving", [False, True])
@pytest.mark.parametrize("system_source", ["none", "prompt", "key"])
def test_dataset_shifts_loss_mask_for_inserted_system_message(
    tmp_path: Path, preserving: bool, system_source: str
) -> None:
    row = _row()
    row.pop("tools")
    row["messages"][3].pop("tool_calls")
    row["system"] = "System instructions"
    path = tmp_path / "chat.jsonl"
    _write_rows(path, [row])
    kwargs = {}
    if system_source == "prompt":
        kwargs["system_prompt"] = "System instructions"
    elif system_source == "key":
        kwargs["system_key"] = "system"
    dataset = OpenAIFormatDataset(
        str(path), use_preserving_dataset=preserving, **kwargs
    )
    actual = dataset.dataset[0]
    expected_mask = [0, 0, 0, 1]
    if system_source != "none":
        expected_mask.insert(0, 0)
        assert actual["messages"][0]["role"] == "system"
    assert actual["message_loss_mask"] == expected_mask
    assert len(actual["messages"]) == len(expected_mask)


def test_preserving_dataset_to_processor_retains_heterogeneous_tool_payloads(
    tmp_path: Path,
) -> None:
    rows = [_row(), _row()]
    rows[1]["messages"][3]["tool_calls"][0]["function"]["arguments"] = {
        "document": {"id": 3},
        "options": None,
    }
    path = tmp_path / "tools.jsonl"
    _write_rows(path, rows)
    raw = OpenAIFormatDataset(str(path), use_preserving_dataset=True)
    assert isinstance(raw.dataset, PreservingDataset)
    processed = AllTaskProcessedDataset(
        dataset=raw.dataset,
        tokenizer=None,
        default_task_data_spec=TaskDataSpec(),
        task_data_processors=chat_kd_processor,
        max_seq_length=1,
    )
    for index, row in enumerate(rows):
        datum = processed[index]
        assert datum["message_log"] == row["messages"]
        assert datum["tools"] == row["tools"]
        assert datum["message_loss_mask"] == row["message_loss_mask"]
    assert set(
        processed[0]["message_log"][3]["tool_calls"][0]["function"]["arguments"]
    ) == {
        "query",
        "exact",
    }


def test_dataset_rejects_invalid_mask_before_system_insertion(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    _write_rows(path, [{**_row(), "message_loss_mask": [1, 0, 0, 1]}])
    with pytest.raises(ValueError, match="only assistant"):
        OpenAIFormatDataset(
            str(path), system_prompt="System", use_preserving_dataset=True
        )
