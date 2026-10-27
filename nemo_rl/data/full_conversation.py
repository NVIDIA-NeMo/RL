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

import inspect
from collections.abc import Callable, Mapping
from functools import lru_cache, partial
from os.path import commonprefix
from typing import Any

import torch
from jinja2 import Template, nodes
from jinja2.visitor import NodeTransformer
from transformers import PreTrainedTokenizerBase

_UNSUPPORTED_TEMPLATE = (
    "Full-conversation SFT cannot locate the assistant turns in this chat "
    "template; use processor: sft_processor instead."
)

# Named apply_chat_template parameters control tokenization, not rendering.
# Transformers passes only its **kwargs to the template, so drop the rest.
_TOKENIZATION_OPTIONS = frozenset(
    inspect.signature(PreTrainedTokenizerBase.apply_chat_template).parameters
) - {"self", "conversation", "tools", "documents", "kwargs"}

# Like incremental SFT, only these turns end with a generation prompt.
_GENERATION_PROMPT_ROLES = ("user", "tool")


def _boundary_call(name: str, args: list[nodes.Expr], lineno: int) -> nodes.Output:
    output = nodes.Output([nodes.Call(nodes.Name(name, "load"), args, [], None, None)])
    output.set_lineno(lineno)
    return output


class _LoopExitTracker(NodeTransformer):
    """Record a loop iteration's exit before each of its own break or continue."""

    def __init__(self, exit_boundary: Callable[[], nodes.Output]) -> None:
        self.exit_boundary = exit_boundary

    def visit_For(self, node: nodes.For, *args: Any, **kwargs: Any) -> nodes.For:
        # Nested loops own their break and continue statements.
        return node

    def visit_Break(
        self, node: nodes.Break, *args: Any, **kwargs: Any
    ) -> list[nodes.Node]:
        return [self.exit_boundary(), node]

    visit_Continue = visit_Break


class _MessageLoopTracker(NodeTransformer):
    """Insert empty-output callbacks at the start and end of each loop iteration."""

    def visit_For(self, node: nodes.For, *args: Any, **kwargs: Any) -> nodes.For:
        self.generic_visit(node, *args, **kwargs)
        if not isinstance(node.target, nodes.Name):
            return node
        enter = _boundary_call(
            "_nemo_sft_enter", [nodes.Name(node.target.name, "load")], node.lineno
        )
        exits = _LoopExitTracker(
            partial(_boundary_call, "_nemo_sft_exit", [], node.lineno)
        )
        body: list[nodes.Node] = []
        for child in node.body:
            visited = exits.visit(child)
            body.extend(visited if isinstance(visited, list) else [visited])
        node.body = [enter, *body, exits.exit_boundary()]
        return node


def _compile_jinja_template(template: str) -> Template:
    """Compile a chat template exactly as Transformers does (cached by Transformers)."""
    # Imported here, not at module top, because the helper is private. If a
    # Transformers release removes it, only this opt-in processor fails.
    try:
        from transformers.utils.chat_template_utils import _compile_jinja_template
    except ImportError as error:
        raise ImportError(
            "Full-conversation SFT requires transformers.utils.chat_template_utils."
            "_compile_jinja_template, which this Transformers version does not "
            "provide."
        ) from error
    return _compile_jinja_template(template)


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
    """Render once and return the text with each assistant turn's character span.

    The inserted callbacks emit empty strings, so the output matches a normal
    render. An iteration that emits nothing, such as a metadata scan, owns no
    output. A break or continue ends the current iteration. The callbacks cannot
    locate a loop that a macro, assignment or filter buffers, so templates with
    such loops raise an error.
    """
    compiled = _compile_message_tracking_template(template)
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    spans: list[tuple[int, int]] = []
    open_iterations: list[tuple[Any, int]] = []
    position = 0

    def enter_iteration(value: Any) -> str:
        open_iterations.append((value, position))
        return ""

    def leave_iteration() -> str:
        value, start = open_iterations.pop()
        is_assistant = isinstance(value, Mapping) and value.get("role") == "assistant"
        if not is_assistant or start == position:
            return ""
        expected = (
            assistant_messages[len(spans)]
            if len(spans) < len(assistant_messages)
            else None
        )
        if value is not expected or (spans and start < spans[-1][1]):
            raise ValueError(
                "The chat template must render each assistant message once, in "
                "order, in its own loop iteration over the messages list. "
                + _UNSUPPORTED_TEMPLATE
            )
        spans.append((start, position))
        return ""

    blocks: list[str] = []
    for block in compiled.generate(
        **template_options,
        messages=messages,
        _nemo_sft_enter=enter_iteration,
        _nemo_sft_exit=leave_iteration,
    ):
        blocks.append(block)
        position += len(block)
    if open_iterations or len(spans) != len(assistant_messages):
        raise ValueError(
            "The chat template must render each assistant message in one streamed "
            "loop iteration; it skips one or renders it inside a macro, "
            "{% set %} block or filter. " + _UNSUPPORTED_TEMPLATE
        )
    return "".join(blocks), spans


def _generation_prompt(
    template: str,
    messages: list[dict[str, Any]],
    template_options: dict[str, Any],
) -> str:
    """Return the text the template appends when asked for a generation prompt."""
    compiled = _compile_jinja_template(template)

    def render(add_generation_prompt: bool) -> str:
        return compiled.render(
            {**template_options, "add_generation_prompt": add_generation_prompt},
            messages=messages,
        )

    without_prompt = render(False)
    with_prompt = render(True)
    if not with_prompt.startswith(without_prompt):
        raise ValueError(
            "The chat template's generation prompt changes earlier output. "
            + _UNSUPPORTED_TEMPLATE
        )
    return with_prompt[len(without_prompt) :]


def _is_text_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return True
    # Assistant tool-call messages may leave content null or missing.
    return (
        content is None
        and message.get("role") == "assistant"
        and bool(message.get("tool_calls"))
    )


def _prepare_messages(
    messages: list[dict[str, Any]], prompt: str | None
) -> list[dict[str, Any]]:
    """Validate the conversation and apply the prompt to the first user message."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Full-conversation SFT must end with an assistant target.")
    for index, message in enumerate(messages):
        if not _is_text_message(message):
            raise ValueError(
                "Full-conversation SFT supports text-only messages; message "
                f"{index} ({message.get('role')}) has non-text content."
            )
    first_user = next(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        None,
    )
    if first_user is None:
        raise ValueError("Full-conversation SFT requires a user message.")
    # Templates cannot mutate their inputs, so a shallow copy is enough.
    prepared = list(messages)
    if prompt:
        prepared[first_user] = {
            **prepared[first_user],
            "content": prompt.format(prepared[first_user]["content"]),
        }
    return prepared


def _resolve_template(
    tokenizer: PreTrainedTokenizerBase, tools: list[dict[str, Any]] | None
) -> tuple[str, dict[str, Any]]:
    """Return the chat template and the variables Transformers would render it with."""
    options: dict[str, Any] = dict(tokenizer.special_tokens_map)
    # get_tokenizer binds defaults with partial; resolve them before selecting
    # the template because tools can select a different template.
    if isinstance(tokenizer.apply_chat_template, partial):
        options.update(tokenizer.apply_chat_template.keywords)
    if tools is not None:
        options["tools"] = tools
    if options.get("continue_final_message"):
        raise ValueError("Full-conversation SFT cannot continue the final message.")
    effective_tools = options.get("tools")
    if effective_tools is not None and any(
        not isinstance(tool, dict) for tool in effective_tools
    ):
        raise ValueError(
            "Full-conversation SFT requires tool schemas to be dictionaries."
        )
    documents = options.get("documents")
    if documents is not None and any(
        not isinstance(document, dict) for document in documents
    ):
        raise TypeError(
            "Documents should be a list of dicts with 'title' and 'text' keys!"
        )
    template = tokenizer.get_chat_template(
        chat_template=options.get("chat_template"), tools=effective_tools
    )
    template_options = {
        "tools": None,
        "documents": None,
        **{
            key: value
            for key, value in options.items()
            if key not in _TOKENIZATION_OPTIONS
        },
        "add_generation_prompt": False,
    }
    return template, template_options


def _mask_generation_prompts(
    template: str,
    messages: list[dict[str, Any]],
    template_options: dict[str, Any],
    text: str,
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Move the generation-prompt prefix of eligible assistant turns into context."""
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    masked = []
    for message_index, (start, end) in zip(assistant_indices, spans):
        if (
            message_index > 0
            and messages[message_index - 1]["role"] in _GENERATION_PROMPT_ROLES
        ):
            # Like incremental SFT, mask only the prompt prefix the turn shares.
            prompt = _generation_prompt(
                template, messages[:message_index], template_options
            )
            start += len(commonprefix([prompt, text[start : start + len(prompt)]]))
        masked.append((start, end))
    return masked


def _add_bos_eos(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    spans: list[tuple[int, int]],
    *,
    add_bos_token: bool,
    add_eos_token: bool,
) -> tuple[str, list[tuple[int, int]]]:
    """Add missing BOS/EOS tokens; the final span always runs to the end."""
    if text[spans[-1][1] :].strip():
        raise ValueError(
            "The chat template renders non-whitespace content after the final "
            "assistant message's loop iteration. " + _UNSUPPORTED_TEMPLATE
        )
    if add_eos_token:
        if tokenizer.eos_token is None:
            raise ValueError("EOS requested but the tokenizer has no EOS token.")
        # Like the incremental formatter, ignore only trailing newlines here.
        if not text.rstrip("\n").endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
    spans = [*spans[:-1], (spans[-1][0], len(text))]
    if add_bos_token:
        if tokenizer.bos_token is None:
            raise ValueError("BOS requested but the tokenizer has no BOS token.")
        if not text.startswith(tokenizer.bos_token):
            text = tokenizer.bos_token + text
            shift = len(tokenizer.bos_token)
            spans = [(start + shift, end + shift) for start, end in spans]
    return text, spans


def _split_segments(
    tokenizer: PreTrainedTokenizerBase, text: str, spans: list[tuple[int, int]]
) -> list[dict[str, str | torch.Tensor]]:
    """Tokenize once and cut the tokens into context and assistant segments."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    tokens: torch.Tensor = encoded["input_ids"][0]
    offsets: torch.Tensor = encoded["offset_mapping"][0]
    # A token belongs to the segment where it starts.
    token_starts = torch.where(offsets[:, 0] < offsets[:, 1], offsets[:, 0], -1)
    segments: list[dict[str, str | torch.Tensor]] = []

    def append_segment(start_token: int, end_token: int, role: str) -> None:
        if start_token < end_token:
            segments.append(
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
    for turn, (start, end) in enumerate(spans, start=1):
        token_indices = torch.nonzero(
            (token_starts >= start) & (token_starts < end)
        ).flatten()
        if not len(token_indices):
            raise ValueError(
                f"Assistant turn {turn} has no tokens of its own: the generation "
                "prompt covers it, or a token from the preceding text absorbs it. "
                + _UNSUPPORTED_TEMPLATE
            )
        first_token = int(token_indices[0])
        end_token = int(token_indices[-1]) + 1
        append_segment(token_cursor, first_token, "context")
        append_segment(first_token, end_token, "assistant")
        token_cursor = end_token
    if token_cursor != len(tokens):
        raise ValueError(
            "The tokenizer produced trailing tokens without character offsets."
        )
    if len(tokens) < 2:
        raise ValueError(
            "Full-conversation SFT has no trainable next-token targets; the "
            "conversation is a single token."
        )
    return segments


def get_full_conversation_message_log(
    messages: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt: str | None,
    tools: list[dict[str, Any]] | None,
    add_bos_token: bool,
    add_eos_token: bool,
    mask_generation_prompt: bool,
) -> list[dict[str, str | torch.Tensor]]:
    """Render and tokenize once, separating assistant targets from context.

    The target for each assistant message is its streamed loop iteration, which
    covers the whole turn with its header and terminator. This matches
    incremental SFT without generation prompts. The chat template needs no
    annotations. With ``mask_generation_prompt``, the part of an assistant turn
    that the template's own generation prompt supplies after a user or tool
    message, typically the assistant header, becomes context instead. This
    matches incremental SFT with generation prompts.

    The returned segments are not source conversation turns. Their roles let the
    existing SFT mask helper select assistant targets without changing,
    retokenizing or shifting the token sequence. The mask helper excludes context
    segments from the loss. It trains on every assistant segment, or only on the
    last one when SFT trains on the final target alone.

    Args:
        messages: Text-only messages ending with an assistant target.
        tokenizer: Fast tokenizer with a Jinja chat template.
        prompt: Optional format string applied to the first user message.
        tools: Tool schemas overriding configured defaults. None preserves the
            defaults; an empty list explicitly disables tools.
        add_bos_token: Prepend the configured BOS if not already present.
        add_eos_token: Append the configured EOS if not already present.
        mask_generation_prompt: Mask the generation-prompt prefix of assistant
            turns that follow a user or tool message.

    Returns:
        Context/assistant segments containing contiguous slices of the full tokens.

    Raises:
        ValueError: Unsupported inputs, or a chat template whose assistant turns
            the formatter cannot track.
    """
    if not tokenizer.is_fast:
        raise ValueError("Full-conversation SFT requires a fast tokenizer.")
    prepared = _prepare_messages(messages, prompt)
    template, template_options = _resolve_template(tokenizer, tools)
    text, spans = _render_with_message_spans(template, prepared, template_options)
    if mask_generation_prompt:
        spans = _mask_generation_prompts(
            template, prepared, template_options, text, spans
        )
    text, spans = _add_bos_eos(
        tokenizer,
        text,
        spans,
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
    )
    return _split_segments(tokenizer, text, spans)
