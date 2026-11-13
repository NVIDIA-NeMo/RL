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
"""Request-local incremental rendering for verified captured Qwen3 histories."""

import asyncio
import copy
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from nemo_rl.models.generation.capture_context import decide_capture_input

LOGGER = logging.getLogger(__name__)

# Official Qwen/Qwen3-0.6B tokenizer_config.json chat_template. An exact
# template contract prevents custom templates from silently changing the suffix.
QWEN3_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"  # pragma: allowlist secret


class TitoNotApplicable(ValueError):
    """The request needs the existing full-history preprocessing path."""


def qwen3_suffix_text(messages: list[dict[str, Any]], *, enable_thinking: bool) -> str:
    """Render appended observations using the verified Qwen3 template contract."""
    if not messages or any(
        message.get("role") not in ("user", "tool")
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise TitoNotApplicable("Expected appended text user/tool observations")
    # Generation stops at im_end; its following template newline is context.
    parts = ["\n"]
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        else:
            if index == 0 or messages[index - 1]["role"] != "tool":
                parts.append("<|im_start|>user")
            parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if index == len(messages) - 1 or messages[index + 1]["role"] != "tool":
                parts.append("<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if not enable_thinking:
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


@dataclass
class CapturedQwen3Renderer:
    """Delegate engine preparation while rendering only verified observations.

    One instance belongs to one request. The shared vLLM renderer, tokenizer,
    and source-history matcher are never mutated.
    """

    renderer: Any
    worker: Any
    admission: Any
    decision: Any = None
    prefix_token_ids: list[int] | None = None

    @property
    def tokenizer(self) -> Any:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> Any:
        return self.renderer.get_tokenizer()

    async def render_chat_async(
        self,
        conversations: list[list[dict[str, Any]]],
        params: Any,
        tok_params: Any,
        *,
        prompt_extras: dict[str, Any],
        skip_mm_cache: bool,
    ) -> Any:
        # vLLM is optional outside the serving worker.
        from vllm.entrypoints.chat_utils import parse_chat_messages_async
        from vllm.renderers.hf import resolve_chat_template_content_format

        # Checkpoint restore owns interrupted output tokens and remaining budgets.
        if self.admission.generation_cut is not None:
            raise TitoNotApplicable("Generation restore requires native preprocessing")
        if len(conversations) != 1:
            raise TitoNotApplicable("Expected one conversation")
        messages = conversations[0]
        kwargs = params.get_apply_chat_template_kwargs()
        template = self.tokenizer.get_chat_template(
            kwargs.get("chat_template"), tools=kwargs.get("tools")
        )
        if hashlib.sha256(template.encode()).hexdigest() != QWEN3_TEMPLATE_SHA256:
            raise TitoNotApplicable("Unqualified Qwen3 template")
        if kwargs.get("tokenize") or kwargs.get("continue_final_message"):
            raise TitoNotApplicable("Non-text or continuing-final-message template")
        if not kwargs.get("add_generation_prompt"):
            raise TitoNotApplicable("Missing assistant generation prompt")
        if any(
            message.get("role") not in ("system", "user", "assistant", "tool")
            or (
                message.get("content") is not None
                and not isinstance(message["content"], str)
            )
            for message in messages
        ):
            raise TitoNotApplicable("Non-text or unsupported role")
        content_format = resolve_chat_template_content_format(
            chat_template=params.chat_template,
            tools=params.chat_template_kwargs.get("tools"),
            given_format=params.chat_template_content_format,
            tokenizer=self.tokenizer,
            model_config=self.renderer.model_config,
        )
        if content_format != "string":
            raise TitoNotApplicable("Non-string renderer")
        conversation, media, media_ids = await parse_chat_messages_async(
            messages,
            self.renderer.model_config,
            content_format=content_format,
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        if media is not None or media_ids is not None:
            raise TitoNotApplicable("Media requires native preprocessing")
        self.decision = decide_capture_input(
            self.admission,
            messages=messages,
            allow_empty_tool_content=bool(conversation)
            and all(isinstance(item.get("content"), str) for item in conversation),
        )
        serving = self.decision.serving
        if serving is None or serving.mode != "token_in":
            raise TitoNotApplicable("No verified continuation")
        previous = self.admission.candidate_replay
        current = self.admission.request_replay
        suffix_count = len(current.items) - previous.item_count
        cut = len(messages) - suffix_count
        if (
            len(conversation) != len(messages)
            or conversation[cut - 1]["role"] != "assistant"
        ):
            raise TitoNotApplicable("Parsing changed the continuation boundary")
        prefix = await asyncio.to_thread(self.worker._resolve_admission_prefix, serving)
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not prefix or prefix[-1] != end_id or end_id != self.tokenizer.eos_token_id:
            raise TitoNotApplicable("Generation did not end at the Qwen3 boundary")
        suffix = qwen3_suffix_text(
            conversation[cut:],
            enable_thinking=kwargs.get("enable_thinking") is not False,
        )
        suffix_ids = await asyncio.to_thread(
            self.tokenizer.encode, suffix, add_special_tokens=False
        )
        self.prefix_token_ids = list(prefix)
        inputs = await self.renderer.render_cmpl_async(
            [{"prompt_token_ids": list(prefix) + suffix_ids}],
            tok_params,
            prompt_extras=prompt_extras,
            skip_mm_cache=skip_mm_cache,
        )
        return [conversation], inputs


async def preprocess_captured_qwen3(
    *,
    worker: Any,
    online_renderer: Any,
    request: Any,
    messages: list[Any],
    default_template: Any,
    default_template_content_format: Any,
    default_template_kwargs: Any,
    tool_dicts: Any,
    parser: Any,
    skip_mm_cache: bool,
) -> tuple[Any, CapturedQwen3Renderer] | None:
    """Use native vLLM request adjustment with a request-local token renderer."""
    # These imports are only needed when the optional TITO serving path is used.
    import vllm
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.renderers.hf import HfRenderer
    from vllm.renderers.online_renderer import OnlineRenderer

    if vllm.__version__ != "0.29.0":
        raise ValueError("Captured Qwen3 TITO requires vLLM 0.29.0")
    if not isinstance(request, ChatCompletionRequest):
        return None
    if type(online_renderer.renderer) is not HfRenderer:
        return None
    if online_renderer.model_config.enable_prompt_embeds:
        return None
    if any(
        getattr(request, name, None)
        for name in (
            "stream",
            "echo",
            "continue_final_message",
            "return_assistant_tokens_mask",
            "add_special_tokens",
            "return_prompt_text",
            "truncate_prompt_tokens",
            "kv_transfer_params",
        )
    ) or getattr(request, "n", 1) not in (None, 1):
        return None
    if getattr(request, "truncate_prompt_tokens", None) is not None:
        return None
    admission = worker._capture_admission(request)
    if admission is None or admission.mode != "candidate":
        return None
    adapter = CapturedQwen3Renderer(online_renderer.renderer, worker, admission)
    local = copy.copy(online_renderer)
    local.renderer = adapter
    try:
        result = await OnlineRenderer.preprocess_chat(
            local,
            request,
            messages,
            default_template=default_template,
            default_template_content_format=default_template_content_format,
            default_template_kwargs=default_template_kwargs,
            tool_dicts=tool_dicts,
            parser=parser,
            skip_mm_cache=skip_mm_cache,
        )
    except TitoNotApplicable as error:
        LOGGER.debug("Captured Qwen3 TITO uses native preprocessing: %s", error)
        return None
    assert adapter.prefix_token_ids is not None
    LOGGER.debug(
        "Captured Qwen3 TITO reused %s prefix tokens", len(adapter.prefix_token_ids)
    )
    return result, adapter
