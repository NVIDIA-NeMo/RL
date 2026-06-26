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

from collections.abc import Iterable, Sequence
from typing import Any


def _get_message_field(message: Any, field: str) -> Any:
    if isinstance(message, dict):
        return message.get(field)
    return getattr(message, field, None)


def _coerce_token_id_list(value: Any, field_name: str) -> list[int]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{field_name} must be an iterable of token IDs.")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must contain only integer token IDs.") from e


def derive_required_prefix_token_ids(messages: Iterable[Any]) -> list[int] | None:
    """Return the latest assistant prefix carried by NeMo-Gym message metadata."""
    for message in reversed(list(messages)):
        prompt_token_ids = _get_message_field(message, "prompt_token_ids")
        generation_token_ids = _get_message_field(message, "generation_token_ids")
        if prompt_token_ids is None or generation_token_ids is None:
            continue
        return _coerce_token_id_list(
            prompt_token_ids,
            "prompt_token_ids",
        ) + _coerce_token_id_list(generation_token_ids, "generation_token_ids")
    return None


def messages_to_last_assistant(messages: Sequence[Any]) -> list[Any]:
    """Return messages through the latest assistant turn, inclusive."""
    for index in reversed(range(len(messages))):
        if _get_message_field(messages[index], "role") == "assistant":
            return list(messages[: index + 1])
    return list(messages)


def replace_prefix_tokens(
    tokenizer: Any,
    *,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> list[int]:
    """Preserve prior model token IDs when a chat template retokenizes history."""
    if not model_prefix_token_ids:
        return template_token_ids

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids[-1] == eos_token_id:
        model_cut_end -= 1

    assert len(template_token_ids) > len(
        template_prefix_token_ids
    ), f"""Found possibly non-monotonically increasing trajectory!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}
"""

    template_cut_start = -1
    for pos in reversed(range(len(template_prefix_token_ids))):
        if template_token_ids[pos] == eos_token_id:
            template_cut_start = pos
            break

    assert (
        template_cut_start >= 0
    ), f"""No EOS token ID found in the chat-templated messages!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"""

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )
