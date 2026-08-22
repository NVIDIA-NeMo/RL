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
from typing import Any


def _json_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a JSON object string or a dict")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must contain valid JSON") from exc
    if not isinstance(decoded, dict):
        raise TypeError(f"{field_name} JSON must decode to an object")
    return decoded


def _metadata_extra_body(nemo_gym_example: dict[str, Any]) -> dict[str, Any]:
    params = nemo_gym_example.get("responses_create_params", {})
    if not isinstance(params, dict):
        raise TypeError("responses_create_params must be a dict")
    metadata = params.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("responses_create_params.metadata must be a dict")
    if "extra_body" not in metadata:
        return {}
    return _json_mapping(
        metadata["extra_body"],
        field_name="responses_create_params.metadata.extra_body",
    )


def _chat_template_kwargs_for_processor(
    nemo_gym_example: dict[str, Any],
) -> dict[str, Any]:
    params = nemo_gym_example.get("responses_create_params", {})
    if not isinstance(params, dict):
        raise TypeError("responses_create_params must be a dict")
    metadata = params.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("responses_create_params.metadata must be a dict")

    extra_body = _metadata_extra_body(nemo_gym_example)
    processor_kwargs: dict[str, Any] = {}
    raw_chat_template_kwargs = metadata.get(
        "chat_template_kwargs", extra_body.get("chat_template_kwargs")
    )
    chat_template_kwargs = (
        _json_mapping(
            raw_chat_template_kwargs,
            field_name="responses_create_params.metadata.chat_template_kwargs",
        )
        if raw_chat_template_kwargs is not None
        else {}
    )
    if chat_template_kwargs:
        processor_kwargs["chat_template_kwargs"] = chat_template_kwargs
    enable_thinking = chat_template_kwargs.get(
        "enable_thinking", extra_body.get("enable_thinking")
    )
    if enable_thinking is not None:
        processor_kwargs["enable_thinking"] = enable_thinking
    return processor_kwargs


def _deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
