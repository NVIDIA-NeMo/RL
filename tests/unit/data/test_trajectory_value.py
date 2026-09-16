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

import torch

from nemo_rl.data.collate_fn import trajectory_value_collate_fn
from nemo_rl.data.datasets.response_datasets.trajectory_value_dataset import (
    IndexedJsonlDataset,
)
from nemo_rl.data.interfaces import TaskDataSpec, TrajectoryValueDatumSpec
from nemo_rl.data.responses import (
    ResponsesMaterialization,
    materialize_responses_create_params,
    normalize_assistant_thinking_prefixes,
    responses_to_chat_messages,
)
from nemo_rl.data.trajectory_value import trajectory_value_processor


class _FakeTokenizer:
    pad_token_id: int | None = 0
    eos_token_id = 99

    def __init__(self) -> None:
        self.messages = None
        self.tools = None

    def apply_chat_template(self, messages, tools, **kwargs):
        self.messages = messages
        self.tools = tools
        assert kwargs["add_generation_prompt"] is True
        return "formatted trajectory"

    def __call__(self, text, **kwargs):
        assert text == "formatted trajectory"
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": torch.tensor([[10, 11, 12]])}


class _BoundaryTokenizer:
    """Character tokenizer with a prefix-stable Responses chat template."""

    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def _content(message):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content)
        return ""

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
        truncate_history_thinking=False,
    ):
        assert tokenize is False
        assert add_special_tokens is False
        assert truncate_history_thinking is False
        rendered = "<tools>" + json.dumps(tools or [], sort_keys=True) + "</tools>"
        for message in messages:
            role = message["role"]
            rendered += f"<{role}>" + self._content(message)
            if role == "assistant":
                rendered += json.dumps(message.get("tool_calls", []), sort_keys=True)
            elif role == "tool":
                rendered += str(message.get("tool_call_id", ""))
            rendered += f"</{role}>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        output = {
            "input_ids": torch.tensor(
                [[ord(character) + 2 for character in text]], dtype=torch.long
            )
        }
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = torch.tensor(
                [[(index, index + 1) for index in range(len(text))]],
                dtype=torch.long,
            )
        return output


class _MergedBoundaryTokenizer(_BoundaryTokenizer):
    """Tokenizer whose BPE-like merge crosses an assistant boundary."""

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        merge = "\n</think>"
        while index < len(text):
            if text.startswith(merge, index):
                token_ids.append(500)
                offsets.append((index, index + len(merge)))
                index += len(merge)
            else:
                token_ids.append(ord(text[index]) + 2)
                offsets.append((index, index + 1))
                index += 1
        output = {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        if kwargs.get("return_offsets_mapping"):
            output["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return output


def _row() -> dict:
    return {
        "schema_version": "trajectory_value_v1",
        "pivot_id": "pivot-1",
        "instance_id": "task-1",
        "label_source": "observed_continuation",
        "value_target": 0.75,
        "pass_count": 3,
        "rollout_count": 4,
        "split": "test",
        "metadata": {"dataset_name": "swebench", "repo": "owner/repo"},
        "responses_create_params": {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix it"}],
                },
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Inspect first"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "shell",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "/repo",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "run a command",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
        },
    }


def _multi_target_params() -> dict:
    return {
        "input": [
            {"type": "message", "role": "user", "content": "Fix it"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "First plan"}],
            },
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "shell",
                "arguments": '{"cmd":"pwd"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "/repo",
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Second plan"}],
            },
            {
                "type": "function_call",
                "call_id": "call-2",
                "name": "shell",
                "arguments": '{"cmd":"pytest"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-2",
                "output": "passed",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "description": "run a command",
                "parameters": {"type": "object"},
            }
        ],
    }


def _multi_target_row(targets: list[dict]) -> dict:
    return {
        "schema_version": "trajectory_value_v2",
        "experiment_id": "unit-test",
        "trajectory_id": "source-line-1",
        "instance_id": "task-1",
        "label_source": "observed",
        "append_query_token": True,
        "targets": targets,
        "responses_create_params": _multi_target_params(),
    }


def test_responses_conversion_keeps_reasoning_tool_call_and_output():
    messages, tools = responses_to_chat_messages(_row()["responses_create_params"])

    assert messages[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "Fix it"}],
    }
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "<think>Inspect first</think>"
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "shell",
        "arguments": {"cmd": "pwd"},
    }
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "/repo",
    }
    assert tools[0]["function"]["name"] == "shell"
    assert "strict" not in tools[0]["function"]


def test_responses_materialization_maps_parallel_items_to_one_action():
    params = _multi_target_params()
    params["input"][3:3] = [
        {
            "type": "function_call",
            "call_id": "call-parallel",
            "name": "shell",
            "arguments": '{"cmd":"ls"}',
        }
    ]
    params["input"][5:5] = [
        {
            "type": "function_call_output",
            "call_id": "call-parallel",
            "output": "files",
        }
    ]

    materialized = materialize_responses_create_params(params)

    assert materialized.message_item_indices[1] == [1, 2, 3]
    assert materialized.item_message_indices[1:4] == [1, 1, 1]
    assert materialized.messages[1]["role"] == "assistant"
    assert len(materialized.messages[1]["tool_calls"]) == 2
    assert materialized.messages[2]["role"] == "tool"
    assert materialized.messages[3]["role"] == "tool"


def test_normalize_assistant_thinking_prefixes_repairs_closing_only_content():
    materialized = ResponsesMaterialization(
        messages=[
            {
                "role": "assistant",
                "content": "Continue checking the patch.</think>\n\n",
                "tool_calls": [],
            }
        ],
        tools=[],
        item_message_indices=[0],
        message_item_indices=[[0]],
    )

    normalized = normalize_assistant_thinking_prefixes(materialized)

    assert normalized.messages[0]["content"].startswith(
        "<think>\nContinue checking the patch.</think>"
    )


def test_processor_appends_query_token_and_preserves_scalar_provenance():
    tokenizer = _FakeTokenizer()
    datum = trajectory_value_processor(
        {
            "trajectory_value_json": json.dumps(_row()),
            "task_name": "trajectory-value-test",
        },
        TaskDataSpec(),
        tokenizer,
        max_seq_length=4,
        idx=7,
    )

    token_ids = datum["message_log"][0]["token_ids"]
    assert isinstance(token_ids, torch.Tensor)
    assert torch.equal(token_ids, torch.tensor([10, 11, 12, 99]))
    assert datum["length"] == 4
    assert datum["untruncated_length"] == 4
    assert datum["loss_multiplier"] == 1.0
    assert datum["value_target"] == 0.75
    assert datum["pass_count"] == 3
    assert datum["rollout_count"] == 4
    assert datum["label_source"] == "observed_continuation"


def test_processor_masks_overlength_without_training_on_truncation():
    datum = trajectory_value_processor(
        {"trajectory_value_json": json.dumps(_row())},
        TaskDataSpec(),
        _FakeTokenizer(),
        max_seq_length=3,
        idx=0,
    )

    assert datum["untruncated_length"] == 4
    assert datum["length"] == 2
    assert datum["loss_multiplier"] == 0.0
    token_ids = datum["message_log"][0]["token_ids"]
    assert isinstance(token_ids, torch.Tensor)
    assert torch.equal(token_ids, torch.tensor([99, 99]))


def test_v2_processor_preserves_flat_group_metadata():
    row = _multi_target_row(
        [
            {
                "target_id": "root",
                "target_type": "direct_root_v",
                "value_target": 0.5,
                "location": {"type": "post_item_state", "item_index": 0},
            }
        ]
    )
    row["split"] = "test"
    row["metadata"] = {
        "dataset_name": "swebench",
        "repo_language": "python",
        "original_task_num_passed": 2,
        "original_task_num_rollouts": 4,
    }

    datum = trajectory_value_processor(
        {"trajectory_value_row": row},
        TaskDataSpec(),
        _BoundaryTokenizer(),
        max_seq_length=None,
        idx=0,
    )

    assert datum["group_metadata"] == {
        "dataset_split": "test",
        "label_source": "observed",
        "dataset_name": "swebench",
        "original_task_num_passed": 2,
        "original_task_num_rollouts": 4,
        "repo_language": "python",
    }


def test_v3_evaluation_targets_share_positions_without_entering_loss():
    location = {"type": "post_item_state", "item_index": 0}
    row = _multi_target_row(
        [
            {
                "target_id": "dense-root",
                "target_type": "dense_exp_root",
                "value_target": 0.5,
                "location": location,
            }
        ]
    )
    row["schema_version"] = "trajectory_value_v3"
    row["evaluation_targets"] = [
        {
            "target_id": "raw-root",
            "target_type": "anchor_raw_root_v",
            "evaluation_suite": "anchor_raw",
            "anchor_kind": "root",
            "value_target": 0.25,
            "location": location,
        },
        {
            "target_id": "exp-root",
            "target_type": "anchor_exp_root_v",
            "evaluation_suite": "anchor_exp",
            "anchor_kind": "root",
            "value_target": 0.75,
            "location": location,
        },
    ]

    tokenizer = _BoundaryTokenizer()
    datum = trajectory_value_processor(
        {"trajectory_value_row": row},
        TaskDataSpec(),
        tokenizer,
        max_seq_length=None,
        idx=0,
    )
    collated = trajectory_value_collate_fn(
        [datum], tokenizer, make_sequence_length_divisible_by=1
    )

    assert datum["evaluation_positions"].tolist() == [
        datum["target_positions"].item(),
        datum["target_positions"].item(),
    ]
    assert datum["evaluation_values"].tolist() == [0.25, 0.75]
    assert float(collated["token_mask"].sum()) == 1.0
    assert collated["returns"][0, datum["target_positions"].item()] == 0.5
    assert collated["evaluation_definition_indices"] == [[0, 1]]


def test_v2_processor_resolves_root_intermediate_and_terminal_v_states():
    targets = [
        {
            "target_id": "root",
            "target_type": "direct_root_v",
            "value_target": 0.5,
            "location": {"type": "post_item_state", "item_index": 0},
        },
        {
            "target_id": "pivot-1",
            "target_type": "direct_pivot_v",
            "value_target": 0.25,
            "location": {"type": "post_item_state", "item_index": 3},
        },
        {
            "target_id": "pivot-2",
            "target_type": "direct_pivot_v",
            "value_target": 1.0,
            "location": {"type": "post_item_state", "item_index": 6},
        },
    ]
    tokenizer = _BoundaryTokenizer()
    datum = trajectory_value_processor(
        {"trajectory_value_row": _multi_target_row(targets)},
        TaskDataSpec(),
        tokenizer,
        max_seq_length=None,
        idx=0,
    )
    materialized = normalize_assistant_thinking_prefixes(
        materialize_responses_create_params(_multi_target_params())
    )
    first_action_start = len(
        tokenizer(
            tokenizer.apply_chat_template(
                materialized.messages[:1],
                tools=materialized.tools,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            ),
            add_special_tokens=False,
        )["input_ids"][0]
    )
    second_action_start = len(
        tokenizer(
            tokenizer.apply_chat_template(
                materialized.messages[:3],
                tools=materialized.tools,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            ),
            add_special_tokens=False,
        )["input_ids"][0]
    )

    assert datum["target_positions"].tolist() == [
        first_action_start,
        second_action_start,
        datum["length"] - 1,
    ]
    assert torch.equal(datum["target_values"], torch.tensor([0.5, 0.25, 1.0]))
    assert datum["target_is_point"] == [True, True, True]
    assert datum["message_log"][0]["token_ids"][-1] == tokenizer.eos_token_id


def test_v2_span_is_assistant_only_and_point_overrides_its_first_state():
    targets = [
        {
            "target_id": "segment-to-pivot-1",
            "target_type": "next_pivot_bootstrap",
            "value_target": 0.25,
            "location": {
                "type": "assistant_items",
                "start_item_exclusive": 0,
                "end_item_inclusive": 3,
                "exclude_first_position": False,
            },
        },
        {
            "target_id": "root",
            "target_type": "direct_root_v",
            "value_target": 0.5,
            "location": {"type": "post_item_state", "item_index": 0},
        },
    ]
    datum = trajectory_value_processor(
        {"trajectory_value_row": _multi_target_row(targets)},
        TaskDataSpec(),
        _BoundaryTokenizer(),
        max_seq_length=None,
        idx=0,
    )

    assert datum["target_is_point"].count(True) == 1
    point_index = datum["target_is_point"].index(True)
    assert datum["target_values"][point_index] == 0.5
    assert all(
        value == 0.25
        for index, value in enumerate(datum["target_values"].tolist())
        if index != point_index
    )


def _assistant_token_positions(tokenizer: _BoundaryTokenizer, params: dict) -> set[int]:
    materialized = normalize_assistant_thinking_prefixes(
        materialize_responses_create_params(params)
    )
    positions: set[int] = set()
    for message_index, message in enumerate(materialized.messages):
        if message["role"] != "assistant":
            continue
        start = len(
            tokenizer(
                tokenizer.apply_chat_template(
                    materialized.messages[:message_index],
                    tools=materialized.tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                ),
                add_special_tokens=False,
            )["input_ids"][0]
        )
        end = len(
            tokenizer(
                tokenizer.apply_chat_template(
                    materialized.messages[: message_index + 1],
                    tools=materialized.tools,
                    tokenize=False,
                    add_generation_prompt=False,
                    add_special_tokens=False,
                ),
                add_special_tokens=False,
            )["input_ids"][0]
        )
        positions.update(range(start, end))
    return positions


def test_v2_dense_mask_contains_only_assistant_states_and_synthetic_query():
    targets = [
        {
            "target_id": "first-action",
            "target_type": "dense",
            "value_target": 0.25,
            "location": {
                "type": "assistant_items",
                "start_item_exclusive": 0,
                "end_item_inclusive": 3,
                "exclude_first_position": False,
            },
        },
        {
            "target_id": "second-action",
            "target_type": "dense",
            "value_target": 0.75,
            "location": {
                "type": "assistant_items",
                "start_item_exclusive": 3,
                "end_item_inclusive": 6,
                "exclude_first_position": False,
            },
        },
        {
            "target_id": "terminal-state",
            "target_type": "point",
            "value_target": 1.0,
            "location": {"type": "post_item_state", "item_index": 6},
        },
    ]
    tokenizer = _BoundaryTokenizer()
    datum = trajectory_value_processor(
        {"trajectory_value_row": _multi_target_row(targets)},
        TaskDataSpec(),
        tokenizer,
        max_seq_length=None,
        idx=0,
    )
    assistant_positions = _assistant_token_positions(tokenizer, _multi_target_params())
    query_position = datum["length"] - 1
    supervised = set(datum["target_positions"].tolist())

    assert supervised <= assistant_positions | {query_position}
    assert query_position in supervised
    assert all(
        position in assistant_positions
        for position, is_point in zip(
            datum["target_positions"].tolist(), datum["target_is_point"]
        )
        if not is_point
    )


def test_v2_tool_call_span_allows_bpe_merge_across_generated_prompt_boundary():
    params = {
        "input": [
            {"type": "message", "role": "user", "content": "Fix it"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "shell",
                "arguments": '{"cmd":"pwd"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "/repo",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "description": "run a command",
                "parameters": {"type": "object"},
            }
        ],
    }
    row = _multi_target_row(
        [
            {
                "target_id": "tool-call-action",
                "target_type": "dense",
                "value_target": 0.25,
                "location": {
                    "type": "assistant_items",
                    "start_item_exclusive": 0,
                    "end_item_inclusive": 1,
                    "exclude_first_position": False,
                },
            }
        ]
    )
    row["responses_create_params"] = params

    datum = trajectory_value_processor(
        {"trajectory_value_row": row},
        TaskDataSpec(),
        _MergedBoundaryTokenizer(),
        max_seq_length=None,
        idx=0,
    )

    assert len(datum["target_positions"]) > 0
    assert torch.all(datum["target_values"] == 0.25)
    assert not any(datum["target_is_point"])


def test_v2_q_target_is_after_assistant_action_and_before_tool_result():
    target = {
        "target_id": "pivot-1-q",
        "target_type": "direct_pivot_q",
        "value_target": 0.75,
        "location": {"type": "post_assistant_action", "item_index": 1},
    }
    tokenizer = _BoundaryTokenizer()
    datum = trajectory_value_processor(
        {"trajectory_value_row": _multi_target_row([target])},
        TaskDataSpec(),
        tokenizer,
        max_seq_length=None,
        idx=0,
    )
    materialized = normalize_assistant_thinking_prefixes(
        materialize_responses_create_params(_multi_target_params())
    )
    expected = len(
        tokenizer(
            tokenizer.apply_chat_template(
                materialized.messages[:2],
                tools=materialized.tools,
                tokenize=False,
                add_generation_prompt=False,
                add_special_tokens=False,
            ),
            add_special_tokens=False,
        )["input_ids"][0]
    )

    assert datum["target_positions"].tolist() == [expected]
    assert materialized.messages[2]["role"] == "tool"


def _single_target_datum(
    token_ids: list[int], value: float, index: int
) -> TrajectoryValueDatumSpec:
    target_id = f"p{index}"
    return {
        "message_log": [
            {
                "role": "trajectory",
                "content": target_id,
                "token_ids": torch.tensor(token_ids),
            }
        ],
        "length": len(token_ids),
        "untruncated_length": len(token_ids),
        "loss_multiplier": 1.0,
        "target_positions": torch.tensor([len(token_ids) - 1]),
        "target_values": torch.tensor([value]),
        "target_is_point": [True],
        "target_definition_indices": [0],
        "target_definitions": [{"target_id": target_id}],
        "trajectory_id": target_id,
        "experiment_id": "unit-test",
        "value_target": value,
        "pivot_id": target_id,
        "instance_id": f"i{index}",
        "label_source": "observed",
        "group_metadata": {},
        "idx": index,
    }


def test_collator_supervises_only_the_query_position():
    batch = [
        _single_target_datum([1, 2, 99], 0.25, 1),
        _single_target_datum([3, 4, 5, 99], 1.0, 2),
    ]
    collated = trajectory_value_collate_fn(
        batch, _FakeTokenizer(), make_sequence_length_divisible_by=4
    )

    assert collated["input_ids"].shape == (2, 4)
    assert torch.equal(collated["query_positions"], torch.tensor([2, 3]))
    assert torch.equal(
        collated["token_mask"],
        torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    )
    assert collated["returns"][0, 2] == 0.25
    assert collated["returns"][1, 3] == 1.0
    assert int((collated["token_mask"][:, 1:] > 0).sum()) == 2
    assert float(collated["token_mask"][:, 1:].sum()) == 2.0


def test_collator_does_not_supervise_zero_weight_samples():
    batch = [
        _single_target_datum([1, 2, 99], 0.25, 1),
        _single_target_datum([3, 4, 5, 99], 1.0, 2),
    ]
    batch[1]["loss_multiplier"] = 0.0

    collated = trajectory_value_collate_fn(
        batch, _FakeTokenizer(), make_sequence_length_divisible_by=4
    )

    effective_mask = collated["token_mask"] * collated["sample_mask"].unsqueeze(-1)
    assert effective_mask[0, 2] == 1.0
    assert effective_mask[1, 3] == 0.0
    assert float(effective_mask.sum()) == 1.0


def test_collator_gives_every_supervised_position_unit_weight():
    datum = trajectory_value_processor(
        {
            "trajectory_value_row": _multi_target_row(
                [
                    {
                        "target_id": "span",
                        "target_type": "bootstrap",
                        "value_target": 0.25,
                        "location": {
                            "type": "assistant_items",
                            "start_item_exclusive": 0,
                            "end_item_inclusive": 3,
                            "exclude_first_position": True,
                        },
                    },
                    {
                        "target_id": "point",
                        "target_type": "direct_root_v",
                        "value_target": 0.5,
                        "location": {
                            "type": "post_item_state",
                            "item_index": 0,
                        },
                    },
                ],
            )
        },
        TaskDataSpec(),
        _BoundaryTokenizer(),
        max_seq_length=None,
        idx=0,
    )
    collated = trajectory_value_collate_fn(
        [datum],
        _BoundaryTokenizer(),
        make_sequence_length_divisible_by=1,
    )
    positions = datum["target_positions"]

    assert torch.all(collated["token_mask"][0, positions] == 1.0)
    assert float(collated["token_mask"].sum()) == len(positions)


def test_indexed_jsonl_uses_existing_byte_offsets(tmp_path):
    data_path = tmp_path / "values.jsonl"
    rows = [{"id": 1}, {"id": 2}]
    encoded = [json.dumps(row).encode() + b"\n" for row in rows]
    data_path.write_bytes(b"".join(encoded))
    index_path = tmp_path / "values.jsonl.idx"
    index_path.write_text(f"0\n{len(encoded[0])}\n", encoding="ascii")

    dataset = IndexedJsonlDataset(data_path, "test")

    assert len(dataset) == 2
    assert json.loads(dataset[1]["trajectory_value_json"]) == {"id": 2}
    assert dataset[-1]["task_name"] == "test"


def test_indexed_jsonl_resolves_canonical_trajectory_reference(tmp_path):
    canonical_path = tmp_path / "canonical.jsonl"
    canonical = {
        "schema_version": "canonical_trajectory_v1",
        "trajectory_id": "source-line-1",
        "instance_id": "task-1",
        "responses_create_params": _multi_target_params(),
        "metadata": {"canonical": True},
    }
    canonical_path.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    experiment = _multi_target_row(
        [
            {
                "target_id": "root",
                "target_type": "direct_root_v",
                "value_target": 0.5,
                "location": {"type": "post_item_state", "item_index": 0},
            }
        ]
    )
    experiment.pop("responses_create_params")
    experiment["trajectory_ref"] = {
        "path": "canonical.jsonl",
        "byte_offset": 0,
        "trajectory_id": "source-line-1",
        "instance_id": "task-1",
    }
    experiment["metadata"] = {"experiment": True}
    data_path = tmp_path / "experiment.jsonl"
    data_path.write_text(json.dumps(experiment) + "\n", encoding="utf-8")
    (tmp_path / "experiment.jsonl.idx").write_text("0\n", encoding="ascii")

    resolved = IndexedJsonlDataset(data_path, "test")[0]["trajectory_value_row"]

    assert resolved["schema_version"] == "trajectory_value_v2"
    assert resolved["responses_create_params"] == _multi_target_params()
    assert resolved["metadata"] == {"canonical": True, "experiment": True}
