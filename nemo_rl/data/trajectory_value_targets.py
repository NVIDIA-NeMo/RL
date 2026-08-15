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

"""Token-boundary resolution for multi-target trajectory-value examples."""

from dataclasses import dataclass
import math
from typing import Any

import torch

from nemo_rl.data.responses import (
    ResponsesMaterialization,
    materialize_responses_create_params,
    normalize_assistant_thinking_prefixes,
)


@dataclass(frozen=True)
class MultiTargetTokenization:
    formatted: str
    token_ids: torch.Tensor
    target_positions: torch.Tensor
    target_values: torch.Tensor
    target_is_point: list[bool]
    target_definition_indices: list[int]
    target_definitions: list[dict[str, Any]]
    evaluation_positions: torch.Tensor
    evaluation_values: torch.Tensor
    evaluation_definition_indices: list[int]
    evaluation_definitions: list[dict[str, Any]]


def _query_token_id(tokenizer: Any) -> int:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(token_id, int):
        raise ValueError("critic tokenization requires an EOS token id")
    return token_id


def _encoded_ids(tokenizer: Any, formatted: str) -> torch.Tensor:
    encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
        raise TypeError("tokenizer must return one flat input_ids tensor")
    return token_ids.to(dtype=torch.long)


class _BoundaryResolver:
    def __init__(
        self,
        tokenizer: Any,
        materialized: ResponsesMaterialization,
        formatted: str,
        full_token_ids: torch.Tensor,
        append_query_token: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.materialized = materialized
        self.formatted = formatted
        self.full_token_ids = full_token_ids
        self.append_query_token = append_query_token
        self._prefix_cache: dict[tuple[int, bool], int] = {}
        self._token_offsets_cache: list[tuple[int, int]] | None = None
        self._assistant_bounds_cache: dict[int, tuple[int, int]] = {}

    def _prefix_char_count(
        self, message_count: int, add_generation_prompt: bool
    ) -> int:
        cache_key = (message_count, add_generation_prompt)
        cached = self._prefix_cache.get(cache_key)
        if cached is not None:
            return cached
        formatted = self.tokenizer.apply_chat_template(
            self.materialized.messages[:message_count],
            tools=self.materialized.tools or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=False,
            truncate_history_thinking=False,
        )
        if not isinstance(formatted, str):
            raise TypeError("tokenizer.apply_chat_template must return a string")
        if not self.formatted.startswith(formatted):
            raise ValueError(
                "chat template text is not prefix-stable at message boundary "
                f"{message_count} (add_generation_prompt={add_generation_prompt})"
            )
        char_count = len(formatted)
        self._prefix_cache[cache_key] = char_count
        return char_count

    def _token_offsets(self) -> list[tuple[int, int]]:
        cached = self._token_offsets_cache
        if cached is not None:
            return cached
        encoded = self.tokenizer(
            self.formatted,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError(
                "tokenizer must support return_offsets_mapping for multi-target "
                "trajectory-value rows"
            )
        if isinstance(offsets, torch.Tensor):
            if offsets.ndim != 3 or offsets.shape[0] != 1 or offsets.shape[2] != 2:
                raise TypeError("offset_mapping must have shape [1, tokens, 2]")
            pairs = [
                (int(start), int(end)) for start, end in offsets[0].to("cpu").tolist()
            ]
        else:
            raw_pairs = (
                offsets[0] if offsets and isinstance(offsets[0], list) else offsets
            )
            pairs = [(int(start), int(end)) for start, end in raw_pairs]
        rendered_token_count = len(self.full_token_ids) - int(self.append_query_token)
        if len(pairs) != rendered_token_count:
            raise ValueError(
                "token offset count does not match rendered token count: "
                f"{len(pairs)} != {rendered_token_count}"
            )
        self._token_offsets_cache = pairs
        return pairs

    def _first_token_overlapping_or_after(self, char_count: int) -> int:
        if char_count >= len(self.formatted):
            return len(self.full_token_ids) - int(self.append_query_token)
        for token_index, (_start, end) in enumerate(self._token_offsets()):
            if end > char_count:
                return token_index
        return len(self.full_token_ids) - int(self.append_query_token)

    def _first_token_starting_at_or_after(self, char_count: int) -> int:
        if char_count >= len(self.formatted):
            return len(self.full_token_ids) - int(self.append_query_token)
        for token_index, (start, _end) in enumerate(self._token_offsets()):
            if start >= char_count:
                return token_index
        return len(self.full_token_ids) - int(self.append_query_token)

    def assistant_bounds(self, message_index: int) -> tuple[int, int]:
        cached = self._assistant_bounds_cache.get(message_index)
        if cached is not None:
            return cached
        messages = self.materialized.messages
        if not 0 <= message_index < len(messages):
            raise ValueError(
                f"assistant message index is out of range: {message_index}"
            )
        if messages[message_index].get("role") != "assistant":
            raise ValueError(f"message {message_index} is not an assistant message")
        start_char = self._prefix_char_count(message_index, add_generation_prompt=True)
        end_char = self._prefix_char_count(
            message_index + 1, add_generation_prompt=False
        )
        start = self._first_token_overlapping_or_after(start_char)
        end = self._first_token_starting_at_or_after(end_char)
        if not 0 <= start < end <= len(self.full_token_ids):
            raise ValueError(
                f"invalid assistant token bounds for message {message_index}: "
                f"start={start}, end={end}, sequence={len(self.full_token_ids)}"
            )
        self._assistant_bounds_cache[message_index] = (start, end)
        return start, end

    def _validated_item_index(self, item_index: Any) -> int:
        if not isinstance(item_index, int) or isinstance(item_index, bool):
            raise ValueError("target item_index must be an integer")
        if not 0 <= item_index < len(self.materialized.item_message_indices):
            raise ValueError(f"target item_index is out of range: {item_index}")
        return item_index

    def post_item_state(self, item_index: Any) -> int:
        item_index = self._validated_item_index(item_index)
        source_message_index = self.materialized.item_message_indices[item_index]
        for message_index in range(
            source_message_index + 1, len(self.materialized.messages)
        ):
            if self.materialized.messages[message_index].get("role") == "assistant":
                return self.assistant_bounds(message_index)[0]
        if not self.append_query_token:
            raise ValueError(
                "post_item_state has no later assistant action and the row has no "
                "terminal query token"
            )
        return len(self.full_token_ids) - 1

    def post_assistant_action(self, item_index: Any) -> int:
        item_index = self._validated_item_index(item_index)
        message_index = self.materialized.item_message_indices[item_index]
        if self.materialized.messages[message_index].get("role") != "assistant":
            raise ValueError(
                f"item {item_index} does not belong to an assistant action"
            )
        position = self.assistant_bounds(message_index)[1]
        if position >= len(self.full_token_ids):
            raise ValueError(
                "post_assistant_action needs a following token or an appended query"
            )
        return position

    def assistant_items(
        self,
        start_item_exclusive: Any,
        end_item_inclusive: Any,
        exclude_first_position: Any,
    ) -> list[tuple[int, int]]:
        if not isinstance(start_item_exclusive, int) or isinstance(
            start_item_exclusive, bool
        ):
            raise ValueError("start_item_exclusive must be an integer")
        end_item_inclusive = self._validated_item_index(end_item_inclusive)
        if not -1 <= start_item_exclusive < end_item_inclusive:
            raise ValueError(
                "assistant_items requires -1 <= start_item_exclusive < "
                "end_item_inclusive"
            )
        if not isinstance(exclude_first_position, bool):
            raise ValueError("exclude_first_position must be boolean")

        selected_messages: list[int] = []
        for message_index, item_indices in enumerate(
            self.materialized.message_item_indices
        ):
            if self.materialized.messages[message_index].get("role") != "assistant":
                continue
            inside = [
                start_item_exclusive < item_index <= end_item_inclusive
                for item_index in item_indices
            ]
            if any(inside) and not all(inside):
                raise ValueError(
                    "assistant_items boundary splits assistant message "
                    f"{message_index}: items={item_indices}"
                )
            if inside and all(inside):
                selected_messages.append(message_index)
        if not selected_messages:
            raise ValueError("assistant_items target selects no assistant messages")

        spans = [self.assistant_bounds(index) for index in selected_messages]
        if exclude_first_position:
            first_start, first_end = spans[0]
            first_start += 1
            if first_start < first_end:
                spans[0] = (first_start, first_end)
            else:
                spans.pop(0)
        if not spans:
            raise ValueError("assistant_items target has no positions after exclusion")
        return spans


def _responses_params_with_limit(row: dict[str, Any]) -> dict[str, Any]:
    params = row.get("responses_create_params")
    if not isinstance(params, dict):
        raise ValueError("responses_create_params must be an object")
    input_items = params.get("input")
    if not isinstance(input_items, list) or not input_items:
        raise ValueError("responses_create_params.input must be a nonempty list")
    limit = row.get("input_item_limit")
    if limit is None:
        return params
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 0 < limit <= len(input_items)
    ):
        raise ValueError(f"invalid input_item_limit: {limit}")
    limited = dict(params)
    limited["input"] = input_items[:limit]
    return limited


def _validated_target_value(target: dict[str, Any]) -> float:
    raw_value = target.get("value_target")
    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        raise ValueError("target value_target must be numeric")
    value = float(raw_value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("target value_target must be finite and in [0, 1]")
    return value


def _resolved_point_position(
    resolver: _BoundaryResolver, target: dict[str, Any]
) -> int:
    location = target["location"]
    location_type = location.get("type")
    if location_type == "post_item_state":
        return resolver.post_item_state(location.get("item_index"))
    if location_type == "post_assistant_action":
        return resolver.post_assistant_action(location.get("item_index"))
    raise ValueError(
        f"target {target['target_id']} has unsupported point location type: "
        f"{location_type}"
    )


def tokenize_multi_target_row(
    row: dict[str, Any], tokenizer: Any
) -> MultiTargetTokenization:
    """Tokenize one v2 row and resolve its semantic targets to token positions."""
    params = _responses_params_with_limit(row)
    materialized = normalize_assistant_thinking_prefixes(
        materialize_responses_create_params(params)
    )
    append_query_token = row.get("append_query_token")
    if not isinstance(append_query_token, bool):
        raise ValueError("append_query_token must be boolean")
    formatted = tokenizer.apply_chat_template(
        materialized.messages,
        tools=materialized.tools or None,
        tokenize=False,
        add_generation_prompt=append_query_token,
        add_special_tokens=False,
        truncate_history_thinking=False,
    )
    if not isinstance(formatted, str):
        raise TypeError("tokenizer.apply_chat_template must return a string")
    token_ids = _encoded_ids(tokenizer, formatted)
    if append_query_token:
        token_ids = torch.cat(
            [token_ids, torch.tensor([_query_token_id(tokenizer)], dtype=torch.long)]
        )

    raw_targets = row.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("targets must be a nonempty list")
    resolver = _BoundaryResolver(
        tokenizer, materialized, formatted, token_ids, append_query_token
    )

    values = torch.zeros(len(token_ids), dtype=torch.float32)
    assigned = torch.zeros(len(token_ids), dtype=torch.bool)
    definition_indices = torch.full((len(token_ids),), -1, dtype=torch.long)
    is_point = torch.zeros(len(token_ids), dtype=torch.bool)
    target_definitions: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()

    def validated_definition(raw_target: Any, definition_index: int) -> dict[str, Any]:
        if not isinstance(raw_target, dict):
            raise ValueError("target definitions must be objects")
        target = dict(raw_target)
        target_id = target.get("target_id")
        location = target.get("location")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be a nonempty string")
        if target_id in seen_target_ids:
            raise ValueError(f"duplicate target_id: {target_id}")
        if not isinstance(location, dict):
            raise ValueError(f"target {target_id} location must be an object")
        target["value_target"] = _validated_target_value(target)
        target["definition_index"] = definition_index
        seen_target_ids.add(target_id)
        return target

    validated_targets = [
        validated_definition(target, index) for index, target in enumerate(raw_targets)
    ]
    target_definitions.extend(validated_targets)

    # Spans are assigned first. Point targets represent exact states and may
    # intentionally replace the first token of an adjacent span.
    for definition_index, target in enumerate(validated_targets):
        location = target["location"]
        if location.get("type") != "assistant_items":
            continue
        spans = resolver.assistant_items(
            location.get("start_item_exclusive"),
            location.get("end_item_inclusive"),
            location.get("exclude_first_position"),
        )
        for start, end in spans:
            if torch.any(assigned[start:end]):
                raise ValueError(
                    f"span target {target['target_id']} overlaps another span target"
                )
            values[start:end] = target["value_target"]
            assigned[start:end] = True
            definition_indices[start:end] = definition_index

    for definition_index, target in enumerate(validated_targets):
        location = target["location"]
        location_type = location.get("type")
        if location_type == "assistant_items":
            continue
        position = _resolved_point_position(resolver, target)
        if is_point[position]:
            raise ValueError(f"multiple point targets resolve to token {position}")
        values[position] = target["value_target"]
        assigned[position] = True
        definition_indices[position] = definition_index
        is_point[position] = True

    target_positions = torch.nonzero(assigned, as_tuple=False).flatten()
    if len(target_positions) == 0:
        raise ValueError("multi-target row resolved to no supervised positions")

    raw_evaluation_targets = row.get("evaluation_targets", [])
    if not isinstance(raw_evaluation_targets, list):
        raise ValueError("evaluation_targets must be a list when present")
    evaluation_definitions: list[dict[str, Any]] = []
    evaluation_positions: list[int] = []
    evaluation_values: list[float] = []
    seen_evaluation_ids: set[str] = set()
    for definition_index, raw_target in enumerate(raw_evaluation_targets):
        if not isinstance(raw_target, dict):
            raise ValueError("evaluation target definitions must be objects")
        target = dict(raw_target)
        target_id = target.get("target_id")
        location = target.get("location")
        evaluation_suite = target.get("evaluation_suite")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("evaluation target_id must be a nonempty string")
        if target_id in seen_evaluation_ids:
            raise ValueError(f"duplicate evaluation target_id: {target_id}")
        if not isinstance(location, dict):
            raise ValueError(
                f"evaluation target {target_id} location must be an object"
            )
        if not isinstance(evaluation_suite, str) or not evaluation_suite:
            raise ValueError(
                f"evaluation target {target_id} evaluation_suite must be a "
                "nonempty string"
            )
        target["value_target"] = _validated_target_value(target)
        target["definition_index"] = definition_index
        position = _resolved_point_position(resolver, target)
        evaluation_definitions.append(target)
        evaluation_positions.append(position)
        evaluation_values.append(target["value_target"])
        seen_evaluation_ids.add(target_id)

    return MultiTargetTokenization(
        formatted=formatted,
        token_ids=token_ids,
        target_positions=target_positions,
        target_values=values[target_positions],
        target_is_point=is_point[target_positions].tolist(),
        target_definition_indices=definition_indices[target_positions].tolist(),
        target_definitions=target_definitions,
        evaluation_positions=torch.tensor(evaluation_positions, dtype=torch.long),
        evaluation_values=torch.tensor(evaluation_values, dtype=torch.float32),
        evaluation_definition_indices=list(range(len(evaluation_definitions))),
        evaluation_definitions=evaluation_definitions,
    )
