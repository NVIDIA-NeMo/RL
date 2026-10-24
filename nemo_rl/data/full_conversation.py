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

"""Full-conversation, text-only SFT formatting with explicit generation spans."""

import re
from copy import deepcopy
from functools import partial
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils.chat_template_utils import render_jinja_template


def get_full_conversation_message_log(
    messages: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt: str | None,
    tools: list[dict[str, Any]] | None,
    add_bos_token: bool,
    add_eos_token: bool,
) -> list[dict[str, str | torch.Tensor]]:
    """Render and tokenize once, separating assistant targets from context.

    A template must mark each complete assistant target, including its turn
    terminator and trailing whitespace, with one generation block. The template
    author is responsible for model-specific terminator coverage; generation
    ranges alone cannot establish semantic completeness. Headers may remain
    outside the block. Returned segments are not source conversation turns;
    their roles let the existing SFT mask helper select assistant targets without
    changing, retokenizing, or shifting the token sequence.
    Context segments are masked out of the training loss; assistant segments
    are supervised unless SFT is configured to train only on the final target.

    Args:
        messages: Text-only messages ending with an assistant target.
        tokenizer: Fast tokenizer with an explicitly generation-marked template.
        prompt: Optional format string applied to the first user message.
        tools: Tool schemas overriding configured defaults. None preserves the
            defaults; an empty list explicitly disables tools.
        add_bos_token: Prepend the configured BOS if not already present.
        add_eos_token: Append the configured EOS if not already present.

    Returns:
        Context/assistant segments containing contiguous slices of the full tokens.

    Raises:
        ValueError: Unsupported inputs, missing/ambiguous generation spans, or
            tokens crossing a supervision boundary.
    """
    if not tokenizer.is_fast:
        raise ValueError("Full-conversation SFT requires a fast tokenizer.")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Full-conversation SFT must end with an assistant target.")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("Full-conversation SFT supports text-only messages.")
    prepared = deepcopy(messages)
    user_indices = [
        index for index, message in enumerate(prepared) if message["role"] == "user"
    ]
    if not user_indices:
        raise ValueError("Full-conversation SFT requires a user message.")
    if prompt:
        prepared[user_indices[0]]["content"] = prompt.format(
            prepared[user_indices[0]]["content"]
        )
    # get_tokenizer binds defaults with partial; resolve them before selecting
    # the template because tools can select a different template.
    template_options: dict[str, Any] = dict(tokenizer.special_tokens_map)
    if isinstance(tokenizer.apply_chat_template, partial):
        template_options.update(tokenizer.apply_chat_template.keywords)
    if tools is not None:
        template_options["tools"] = tools
    template = tokenizer.get_chat_template(
        chat_template=template_options.get("chat_template"),
        tools=template_options.get("tools"),
    )
    if not re.search(r"\{%-?\s*generation\s*-?%\}", template):
        raise ValueError("Full-conversation SFT requires {% generation %} blocks.")
    template_options.update(
        chat_template=template,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    template_options.pop("tokenize", None)
    rendered_conversations, generation_spans = render_jinja_template(
        conversations=[prepared], **template_options
    )
    text = rendered_conversations[0]
    spans = generation_spans[0]
    if len(spans) != sum(message["role"] == "assistant" for message in prepared):
        raise ValueError("Expected exactly one generation block per assistant target.")
    previous_span_end = 0
    for start, end in spans:
        if not previous_span_end <= start < end <= len(text):
            raise ValueError("Assistant generation spans must be nonempty and ordered.")
        previous_span_end = end
    eos_token = tokenizer.eos_token
    if eos_token:
        for (start, end), (next_start, _) in zip(spans, spans[1:]):
            assistant_ends_with_eos = text[start:end].rstrip().endswith(eos_token)
            following_context_starts_with_eos = (
                text[end:next_start].lstrip().startswith(eos_token)
            )
            # After a supervised EOS, the next EOS-looking token may be a BOS.
            if not assistant_ends_with_eos and following_context_starts_with_eos:
                raise ValueError(
                    "An assistant EOS terminator is outside its generation block."
                )
    if spans[-1][1] != len(text):
        raise ValueError(
            "The final generation block must include the assistant terminator "
            "and all trailing whitespace."
        )
    if add_eos_token:
        if tokenizer.eos_token is None:
            raise ValueError("EOS requested but the tokenizer has no EOS token.")
        # Match the incremental formatter: only newlines are ignored here.
        if not text.rstrip("\n").endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
            spans[-1] = (spans[-1][0], len(text))
    if add_bos_token:
        if tokenizer.bos_token is None:
            raise ValueError("BOS requested but the tokenizer has no BOS token.")
        if not text.startswith(tokenizer.bos_token):
            text = tokenizer.bos_token + text
            spans = [
                (start + len(tokenizer.bos_token), end + len(tokenizer.bos_token))
                for start, end in spans
            ]
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    tokens: torch.Tensor = encoded["input_ids"][0]
    offsets: torch.Tensor = encoded["offset_mapping"][0]
    has_character_offset = offsets[:, 0] < offsets[:, 1]
    output: list[dict[str, str | torch.Tensor]] = []

    def append_segment(start_token: int, end_token: int, role: str) -> None:
        if start_token < end_token:
            output.append(
                {
                    "role": role,
                    "content": tokenizer.decode(
                        tokens[start_token:end_token],
                        clean_up_tokenization_spaces=False,
                    ),
                    "token_ids": tokens[start_token:end_token],
                }
            )

    token_cursor = 0
    next_token_targets = 0
    for start, end in spans:
        overlaps_span = (
            has_character_offset & (offsets[:, 1] > start) & (offsets[:, 0] < end)
        )
        token_indices = torch.nonzero(overlaps_span).flatten()
        if not len(token_indices):
            raise ValueError("An assistant generation span has no supervised tokens.")
        if (
            (offsets[overlaps_span, 0] < start) | (offsets[overlaps_span, 1] > end)
        ).any():
            raise ValueError("A token crosses an assistant supervision boundary.")
        first_token = int(token_indices[0])
        end_token = int(token_indices[-1]) + 1
        if first_token < token_cursor:
            raise ValueError("Assistant generation spans overlap in token space.")
        append_segment(token_cursor, first_token, "context")
        append_segment(first_token, end_token, "assistant")
        # Position zero has no preceding token from which to predict it.
        next_token_targets += max(0, end_token - max(first_token, 1))
        token_cursor = end_token
    if token_cursor != len(tokens):
        raise ValueError("The final assistant span must cover the final token.")
    if not next_token_targets:
        raise ValueError("Full-conversation SFT has no trainable next-token targets.")
    if tokenizer.decode(tokens, clean_up_tokenization_spaces=False) != text:
        raise ValueError("Full-conversation tokenization failed its text round trip.")
    return output
