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

"""Full-conversation SFT formatting with automatic assistant loss ownership."""

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache, partial
from typing import Any

import torch
from jinja2 import Template, nodes
from jinja2.visitor import NodeTransformer
from transformers import PreTrainedTokenizerBase
from transformers.utils.chat_template_utils import (
    _compile_jinja_template,
    render_jinja_template,
)


class _MessageLoopTracker(NodeTransformer):
    """Insert empty-output boundary callbacks without changing Jinja whitespace."""

    def visit_For(self, node: nodes.For, *args: Any, **kwargs: Any) -> nodes.For:
        self.generic_visit(node, *args, **kwargs)
        if isinstance(node.target, nodes.Name):

            def boundary(entering: bool) -> nodes.Output:
                output = nodes.Output(
                    [
                        nodes.Call(
                            nodes.Name("_nemo_sft_boundary", "load"),
                            [
                                nodes.Name(node.target.name, "load"),
                                nodes.Const(entering),
                            ],
                            [],
                            None,
                            None,
                        )
                    ]
                )
                output.set_lineno(node.lineno)
                return output

            node.body = [boundary(True), *node.body, boundary(False)]
        return node


@lru_cache(maxsize=128)
def _has_generation_blocks(template: str) -> bool:
    environment = _compile_jinja_template(template).environment
    generation_extensions = {
        identifier
        for identifier, extension in environment.extensions.items()
        if "generation" in extension.tags
    }
    return any(
        node.identifier in generation_extensions
        for node in environment.parse(template).find_all(nodes.ExtensionAttribute)
    )


@lru_cache(maxsize=128)
def _compile_message_tracking_template(template: str) -> Template:
    environment = _compile_jinja_template(template).environment
    tree = _MessageLoopTracker().visit(environment.parse(template))
    return environment.from_string(tree)


def _render_with_message_spans(
    template: str,
    messages: list[dict[str, Any]],
    template_options: dict[str, Any],
) -> tuple[str, list[tuple[int, int]]]:
    """Track streamed assistant loop iterations, including headers and terminators.

    Rendering is unchanged: callbacks emit empty strings, not markers in message
    content. Loops buffered inside macros, assignments or filters cannot be
    tracked in stream coordinates and are rejected.
    """
    compiled = _compile_message_tracking_template(template)
    spans: list[tuple[int, int]] = []
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    position = 0
    active: list[tuple[Any, int]] = []

    def boundary(value: Any, entering: bool) -> str:
        if not isinstance(value, Mapping) or value.get("role") != "assistant":
            return ""
        if entering:
            active.append((value, position))
        else:
            if not active:
                raise ValueError(
                    "Unbalanced assistant message loop; use explicit "
                    "{% generation %} blocks."
                )
            entered_value, start = active.pop()
            if entered_value is not value:
                raise ValueError(
                    "The template replaced an assistant message inside its loop; "
                    "use explicit {% generation %} blocks."
                )
            # Templates may scan messages for validation or history metadata
            # before rendering them. Such loops do not own any output.
            if start == position:
                return ""
            if (
                len(spans) >= len(assistant_messages)
                or value is not assistant_messages[len(spans)]
            ):
                raise ValueError(
                    "Cannot infer assistant ownership from nested, copied or reordered "
                    "message loops; use explicit {% generation %} blocks."
                )
            spans.append((start, position))
        return ""

    blocks: list[str] = []
    render_options = {
        "tools": None,
        "documents": None,
        "add_generation_prompt": False,
        **template_options,
    }
    for block in compiled.generate(
        **render_options, messages=messages, _nemo_sft_boundary=boundary
    ):
        blocks.append(block)
        position += len(block)
    if active or len(spans) != len(assistant_messages):
        raise ValueError(
            "Expected one rendered loop iteration per assistant message; "
            "use explicit {% generation %} blocks for this template."
        )
    return "".join(blocks), spans


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

    Explicit generation blocks select assistant targets when present. Otherwise,
    streamed assistant message-loop iterations select whole assistant turns,
    including headers, matching incremental SFT without generation prompts.
    Returned segments are not source conversation turns;
    their roles let the existing SFT mask helper select assistant targets without
    changing, retokenizing, or shifting the token sequence.
    Context segments are masked out of the training loss; assistant segments
    are supervised unless SFT is configured to train only on the final target.

    Args:
        messages: Text-only messages ending with an assistant target.
        tokenizer: Fast tokenizer with a Jinja chat template.
        prompt: Optional format string applied to the first user message.
        tools: Tool schemas overriding configured defaults. None preserves the
            defaults; an empty list explicitly disables tools.
        add_bos_token: Prepend the configured BOS if not already present.
        add_eos_token: Append the configured EOS if not already present.

    Returns:
        Context/assistant segments containing contiguous slices of the full tokens.

    Raises:
        ValueError: Unsupported inputs or missing/ambiguous assistant spans.
    """
    if not tokenizer.is_fast:
        raise ValueError("Full-conversation SFT requires a fast tokenizer.")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Full-conversation SFT must end with an assistant target.")
    if any(
        not isinstance(message.get("content"), str)
        and not (
            message.get("content") is None
            and message.get("role") == "assistant"
            and message.get("tool_calls")
        )
        for message in messages
    ):
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
    effective_tools = template_options.get("tools")
    if effective_tools is not None and any(
        not isinstance(tool, dict) for tool in effective_tools
    ):
        raise ValueError(
            "Full-conversation SFT requires tool schemas to be dictionaries."
        )
    documents = template_options.get("documents")
    if documents is not None and any(
        not isinstance(document, dict) for document in documents
    ):
        raise TypeError(
            "Documents should be a list of dicts with 'title' and 'text' keys!"
        )
    template = tokenizer.get_chat_template(
        chat_template=template_options.get("chat_template"),
        tools=template_options.get("tools"),
    )
    if template_options.get("continue_final_message"):
        raise ValueError("Full-conversation SFT cannot continue the final message.")
    for key in (
        "chat_template",
        "tokenize",
        "return_assistant_tokens_mask",
        "continue_final_message",
        "return_dict",
        "return_tensors",
        "tokenizer_kwargs",
        "padding",
        "truncation",
        "max_length",
    ):
        template_options.pop(key, None)
    template_options["add_generation_prompt"] = False
    if _has_generation_blocks(template):
        rendered_conversations, generation_spans = render_jinja_template(
            conversations=[prepared],
            chat_template=template,
            return_assistant_tokens_mask=True,
            **template_options,
        )
        text = rendered_conversations[0]
        spans = generation_spans[0]
    else:
        text, spans = _render_with_message_spans(template, prepared, template_options)
    if len(spans) != sum(message["role"] == "assistant" for message in prepared):
        raise ValueError("Expected exactly one generation block per assistant target.")
    previous_span_end = 0
    for start, end in spans:
        if not previous_span_end <= start < end <= len(text):
            raise ValueError("Assistant generation spans must be nonempty and ordered.")
        previous_span_end = end
    if text[spans[-1][1] :].strip():
        raise ValueError(
            "Non-whitespace content follows the final assistant span; "
            "include the assistant terminator in its generation block or message loop."
        )
    spans[-1] = (spans[-1][0], len(text))
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
        starts_in_span = (
            has_character_offset & (offsets[:, 0] >= start) & (offsets[:, 0] < end)
        )
        token_indices = torch.nonzero(starts_in_span).flatten()
        if not len(token_indices):
            raise ValueError("An assistant generation span has no supervised tokens.")
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
    return output
