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

"""Processing for scalar-labeled trajectory-prefix critic examples."""

import json
import math
from typing import Any

import torch

from nemo_rl.data.interfaces import TaskDataSpec, TrajectoryValueDatumSpec
from nemo_rl.data.responses import responses_to_chat_messages
from nemo_rl.data.trajectory_value_targets import tokenize_multi_target_row


def _validated_target(row: dict[str, Any]) -> float:
    target = row.get("value_target")
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        raise ValueError("value_target must be numeric")
    target = float(target)
    if not math.isfinite(target):
        raise ValueError("value_target must be finite")
    if not 0 <= target <= 1:
        raise ValueError("value_target must be in [0, 1]")
    return target


def _count_provenance(row: dict[str, Any]) -> tuple[int, int] | None:
    pass_count = row.get("pass_count")
    rollout_count = row.get("rollout_count")
    if (
        not isinstance(pass_count, int)
        or isinstance(pass_count, bool)
        or not isinstance(rollout_count, int)
        or isinstance(rollout_count, bool)
        or rollout_count <= 0
        or not 0 <= pass_count <= rollout_count
    ):
        return None
    return pass_count, rollout_count


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _group_metadata(row: dict[str, Any], label_source: str) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    grouped: dict[str, Any] = {
        "dataset_split": row.get("split"),
        "label_source": label_source,
        "dataset_name": metadata.get("dataset_name"),
        "repo": metadata.get("repo"),
        "source_rollout_reward": metadata.get("source_rollout_reward"),
        "original_task_reward_mean": metadata.get("original_task_reward_mean"),
        "original_task_num_passed": metadata.get("original_task_num_passed"),
        "original_task_num_rollouts": metadata.get("original_task_num_rollouts"),
        "repo_language": metadata.get("repo_language"),
        "reasoning_rank": metadata.get("reasoning_rank"),
        "position_fraction": metadata.get("position_fraction"),
        "location_bucket": metadata.get("location_bucket"),
    }
    instance_dict = _parse_json_object(metadata.get("instance_dict"))
    if grouped["repo_language"] is None:
        grouped["repo_language"] = instance_dict.get("repo_language")

    pivot_source = _parse_json_object(metadata.get("pivot_source"))
    pivot = pivot_source.get("pivot", {})
    if isinstance(pivot, dict):
        for key in ("reasoning_rank", "position_fraction", "location_bucket"):
            if grouped[key] is None:
                grouped[key] = pivot.get(key)
    return {key: value for key, value in grouped.items() if value is not None}


def _query_token_id(tokenizer: Any) -> int:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(token_id, int):
        raise ValueError("critic tokenization requires an EOS token id")
    return token_id


def _trajectory_value_row(datum_dict: dict[str, Any]) -> dict[str, Any]:
    materialized = datum_dict.get("trajectory_value_row")
    raw_json = datum_dict.get("trajectory_value_json")
    if materialized is not None and raw_json is not None:
        raise ValueError(
            "datum cannot contain both trajectory_value_row and trajectory_value_json"
        )
    if materialized is not None:
        if not isinstance(materialized, dict):
            raise ValueError("trajectory_value_row must be an object")
        return materialized
    if not isinstance(raw_json, str):
        raise ValueError("datum must contain a trajectory-value row")
    row = json.loads(raw_json)
    if not isinstance(row, dict):
        raise ValueError("trajectory-value row must be a JSON object")
    return row


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _mask_overlength(
    token_ids: torch.Tensor,
    max_seq_length: int | None,
    tokenizer: Any,
) -> tuple[torch.Tensor, int, float, bool]:
    untruncated_length = len(token_ids)
    overlength = max_seq_length is not None and untruncated_length > max_seq_length
    if overlength:
        query_token_id = _query_token_id(tokenizer)
        token_ids = torch.tensor([query_token_id, query_token_id], dtype=torch.long)
    return token_ids, untruncated_length, 0.0 if overlength else 1.0, overlength


def _task_name(output: TrajectoryValueDatumSpec, datum_dict: dict[str, Any]) -> None:
    task_name = datum_dict.get("task_name")
    if isinstance(task_name, str):
        output["task_name"] = task_name


def _process_v1(
    row: dict[str, Any],
    datum_dict: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int | None,
    idx: int,
) -> TrajectoryValueDatumSpec:
    """Preserve the original one-prefix, one-terminal-query row contract."""
    target = _validated_target(row)
    pivot_id = _required_string(row, "pivot_id")
    instance_id = _required_string(row, "instance_id")
    label_source = _required_string(row, "label_source")

    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        raise ValueError("responses_create_params must be an object")
    messages, tools = responses_to_chat_messages(responses_create_params)
    formatted = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    if not isinstance(formatted, str):
        raise TypeError("tokenizer.apply_chat_template must return a string")
    encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0].to(dtype=torch.long)
    token_ids = torch.cat(
        [token_ids, torch.tensor([_query_token_id(tokenizer)], dtype=torch.long)]
    )
    token_ids, untruncated_length, loss_multiplier, overlength = _mask_overlength(
        token_ids, max_seq_length, tokenizer
    )
    target_positions = (
        torch.empty(0, dtype=torch.long)
        if overlength
        else torch.tensor([len(token_ids) - 1], dtype=torch.long)
    )
    target_values = (
        torch.empty(0, dtype=torch.float32)
        if overlength
        else torch.tensor([target], dtype=torch.float32)
    )
    target_definition = {
        "target_id": pivot_id,
        "target_type": "terminal_query",
        "value_target": target,
        "label_source": label_source,
        "definition_index": 0,
    }

    output: TrajectoryValueDatumSpec = {
        "message_log": [
            {"role": "trajectory", "content": formatted, "token_ids": token_ids}
        ],
        "length": len(token_ids),
        "untruncated_length": untruncated_length,
        "loss_multiplier": loss_multiplier,
        "target_positions": target_positions,
        "target_values": target_values,
        "target_is_point": [] if overlength else [True],
        "target_definition_indices": [] if overlength else [0],
        "target_definitions": [target_definition],
        "evaluation_positions": torch.empty(0, dtype=torch.long),
        "evaluation_values": torch.empty(0, dtype=torch.float32),
        "evaluation_definition_indices": [],
        "evaluation_definitions": [],
        "trajectory_id": pivot_id,
        "experiment_id": "trajectory_value_v1",
        "value_target": target,
        "pivot_id": pivot_id,
        "instance_id": instance_id,
        "label_source": label_source,
        "group_metadata": _group_metadata(row, label_source),
        "idx": idx,
    }
    count_provenance = _count_provenance(row)
    if count_provenance is not None:
        output["pass_count"], output["rollout_count"] = count_provenance
        target_definition["pass_count"], target_definition["rollout_count"] = (
            count_provenance
        )
    _task_name(output, datum_dict)
    return output


def _process_v2(
    row: dict[str, Any],
    datum_dict: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int | None,
    idx: int,
) -> TrajectoryValueDatumSpec:
    tokenized = tokenize_multi_target_row(row, tokenizer)
    token_ids, untruncated_length, loss_multiplier, overlength = _mask_overlength(
        tokenized.token_ids, max_seq_length, tokenizer
    )
    trajectory_id = _required_string(row, "trajectory_id")
    instance_id = _required_string(row, "instance_id")
    experiment_id = _required_string(row, "experiment_id")
    label_source = row.get("label_source", "multi_target")
    if not isinstance(label_source, str):
        raise ValueError("label_source must be a string when present")

    output: TrajectoryValueDatumSpec = {
        "message_log": [
            {
                "role": "trajectory",
                "content": tokenized.formatted,
                "token_ids": token_ids,
            }
        ],
        "length": len(token_ids),
        "untruncated_length": untruncated_length,
        "loss_multiplier": loss_multiplier,
        "target_positions": (
            torch.empty(0, dtype=torch.long)
            if overlength
            else tokenized.target_positions
        ),
        "target_values": (
            torch.empty(0, dtype=torch.float32)
            if overlength
            else tokenized.target_values
        ),
        "target_is_point": [] if overlength else tokenized.target_is_point,
        "target_definition_indices": (
            [] if overlength else tokenized.target_definition_indices
        ),
        "target_definitions": tokenized.target_definitions,
        "evaluation_positions": (
            torch.empty(0, dtype=torch.long)
            if overlength
            else tokenized.evaluation_positions
        ),
        "evaluation_values": (
            torch.empty(0, dtype=torch.float32)
            if overlength
            else tokenized.evaluation_values
        ),
        "evaluation_definition_indices": (
            [] if overlength else tokenized.evaluation_definition_indices
        ),
        "evaluation_definitions": tokenized.evaluation_definitions,
        "trajectory_id": trajectory_id,
        "experiment_id": experiment_id,
        "instance_id": instance_id,
        "label_source": label_source,
        "group_metadata": _group_metadata(row, label_source),
        "idx": idx,
    }
    pivot_id = row.get("pivot_id")
    if isinstance(pivot_id, str):
        output["pivot_id"] = pivot_id
    _task_name(output, datum_dict)
    return output


def trajectory_value_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int | None,
    idx: int,
) -> TrajectoryValueDatumSpec:
    """Tokenize a trajectory-value row and resolve all supervised states."""
    del task_data_spec
    row = _trajectory_value_row(datum_dict)
    schema_version = row.get("schema_version")
    if schema_version == "trajectory_value_v1":
        return _process_v1(row, datum_dict, tokenizer, max_seq_length, idx)
    if schema_version in {"trajectory_value_v2", "trajectory_value_v3"}:
        return _process_v2(row, datum_dict, tokenizer, max_seq_length, idx)
    raise ValueError(f"unsupported schema_version: {schema_version}")
