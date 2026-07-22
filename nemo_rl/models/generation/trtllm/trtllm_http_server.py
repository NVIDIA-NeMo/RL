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
"""OpenAI-compatible HTTP server wrapping ``tensorrt_llm.LLM``, serving /v1/chat/completions.

Returns NeMoGym fields (prompt_token_ids, generation_token_ids, generation_log_probs).
Supports Qwen3 tool calling, DeepSeekR1Parser reasoning, and prefix token splicing.
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

from nemo_rl.models.generation.openai_server_utils import (
    replace_prefix_tokens,
)

logger = logging.getLogger(__name__)


def create_app(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    max_seq_len: int,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    tool_parser: str | None = None,
) -> "FastAPI":
    """Build a FastAPI application backed by *llm* (``tensorrt_llm.LLM``)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    # Per-request template kwargs override these defaults.
    _server_template_kwargs: dict[str, Any] = {
        "enable_thinking": True,
        **(default_chat_template_kwargs or {}),
    }

    # Use the configured parser or infer one from the model config.
    _tool_parser_name = _resolve_tool_parser_name(tool_parser, model_name)
    _tool_parser_instance = _build_tool_parser(_tool_parser_name)
    _parse_tool_calls = _make_parse_tool_calls(_tool_parser_instance)

    from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

    model_config = getattr(llm, "_hf_model_config", None)
    if model_config is None:
        raise RuntimeError(
            "TRT-LLM HTTP server requires the LLM's loaded Hugging Face model config"
        )

    # Cache parsers for both thinking modes.
    from tensorrt_llm.llmapi.reasoning_parser import DeepSeekR1Parser

    _reasoning_parsers = {
        enabled: DeepSeekR1Parser(reasoning_at_start=enabled)
        for enabled in (False, True)
    }

    # Strip generated stop tokens to preserve prefix continuity across turns.
    _eos_token_ids: set[int] = set()
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        _tid = tokenizer.convert_tokens_to_ids(_tok)
        if isinstance(_tid, int) and _tid != tokenizer.unk_token_id:
            _eos_token_ids.add(_tid)
    if tokenizer.eos_token_id is not None:
        _eos_token_ids.add(tokenizer.eos_token_id)

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body: dict = await request.json()
        messages: list[dict] = body.get("messages", [])
        tools: list[dict] | None = body.get("tools")
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        logprobs_requested = body.get("logprobs", False)

        # Request kwargs override server defaults.
        per_request_kwargs: dict[str, Any] = body.get("chat_template_kwargs") or {}
        effective_template_kwargs = {**_server_template_kwargs, **per_request_kwargs}
        enable_thinking: bool = bool(
            effective_template_kwargs.get("enable_thinking", True)
        )

        reasoning_parser = _reasoning_parsers[enable_thinking]

        try:
            conversation, mm_coroutine, _ = parse_chat_messages_coroutines(
                messages, model_config
            )
            mm_data, mm_embeddings = await mm_coroutine
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        # This token-only adapter does not support multimodal inputs.
        if mm_data is not None or mm_embeddings is not None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "NeMo-RL's TRT-LLM HTTP adapter does not support "
                    "multimodal chat inputs"
                },
            )

        # Full retokenization avoids accumulating generation token IDs twice.
        prompt_token_ids = _build_prompt_token_ids(
            conversation,
            tokenizer,
            tools=tools,
            default_template_kwargs=effective_template_kwargs,
        )

        # Empty required_prefix_ids on turn one returns the template unchanged.
        required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
            messages,
            conversation,
            tokenizer,
            tools,
            effective_template_kwargs,
        )

        adj_prompt = replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=required_prefix_ids,
            template_prefix_token_ids=template_prefix_ids,
            template_token_ids=prompt_token_ids,
        )

        max_tokens_requested = (
            body.get("max_tokens") or body.get("max_completion_tokens") or max_seq_len
        )
        remaining_ctx = max(0, max_seq_len - len(adj_prompt))

        # Return HTTP 400 on context exhaustion.
        if remaining_ctx == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"context length exceeded: prompt ({len(adj_prompt)} tokens) exhausted context window ({max_seq_len})"
                },
            )

        max_tokens = min(int(max_tokens_requested), remaining_ctx)

        from tensorrt_llm import SamplingParams as TrtSamplingParams
        from tensorrt_llm.executor.utils import RequestError

        sampling = TrtSamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            logprobs=True,
        )

        try:
            outputs = await asyncio.to_thread(
                llm.generate,
                [{"prompt_token_ids": adj_prompt}],
                sampling_params=sampling,
            )
        except RequestError as e:
            err = str(e)
            if "max_seq_len" in err or "max_num_tokens" in err:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"context length exceeded: {err}"},
                )
            raise

        output = outputs[0]
        gen = output.outputs[0]
        gen_token_ids = list(gen.token_ids)

        gen_logprobs: list[float] = []
        if gen.logprobs:
            # TRT-LLM returns floats in simple format and token-indexed dicts otherwise.
            for token_id, lp in zip(gen_token_ids, gen.logprobs, strict=True):
                if isinstance(lp, (int, float)):
                    gen_logprobs.append(float(lp))
                elif isinstance(lp, dict):
                    gen_logprobs.append(float(lp[token_id].logprob))
                else:
                    raise TypeError(f"Unsupported TRT-LLM logprob type: {type(lp)}")

        # Strip trailing stop tokens TRT-LLM appends — apply_chat_template doesn't reproduce
        # <|endoftext|>, so they'd break seen_token_ids contiguity. Trim logprobs in lockstep.
        while gen_token_ids and gen_token_ids[-1] in _eos_token_ids:
            gen_token_ids.pop()
            if gen_logprobs:
                gen_logprobs.pop()

        gen_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
        # Strip EOS strings that TRT-LLM may append to decoded output.
        gen_text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")

        finish_reason = "stop"
        if gen.finish_reason is not None:
            fr = str(gen.finish_reason).lower()
            if "length" in fr:
                finish_reason = "length"

        # Split reasoning from answer using the (possibly per-request) parser.
        parsed = reasoning_parser.parse(gen_text)
        reasoning_content: str = parsed.reasoning_content
        answer_text: str = parsed.content

        if tools:
            content_text, parsed_tool_calls = _parse_tool_calls(answer_text, tools)
        else:
            content_text, parsed_tool_calls = answer_text, []

        if parsed_tool_calls:
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": content_text or None,
                "reasoning_content": reasoning_content,
                "tool_calls": parsed_tool_calls,
                "prompt_token_ids": adj_prompt,
                "generation_token_ids": gen_token_ids,
                "generation_log_probs": gen_logprobs,
            }
            finish_reason = "tool_calls"
        else:
            msg_dict = {
                "role": "assistant",
                "content": answer_text,
                "reasoning_content": reasoning_content,
                "prompt_token_ids": adj_prompt,
                "generation_token_ids": gen_token_ids,
                "generation_log_probs": gen_logprobs,
            }

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {"index": 0, "message": msg_dict, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": len(adj_prompt),
                "completion_tokens": len(gen_token_ids),
                "total_tokens": len(adj_prompt) + len(gen_token_ids),
            },
        }

        if logprobs_requested and gen_logprobs:
            response["choices"][0]["logprobs"] = {
                "content": [
                    {
                        "token": tokenizer.decode([tid]),
                        "logprob": lp,
                        "bytes": None,
                        "top_logprobs": [],
                    }
                    for tid, lp in zip(gen_token_ids, gen_logprobs)
                ]
            }

        return JSONResponse(content=response)

    return app


# ---------------------------------------------------------------------------
#  Tool-call parser factory — delegates to TRT-LLM's registered parsers

# ---------------------------------------------------------------------------


def _resolve_tool_parser_name(configured_name: str | None, model_name: str) -> str:
    """Resolve the configured parser or infer it from the model."""
    if configured_name:
        return configured_name

    from tensorrt_llm.serve.tool_parser.tool_parser_factory import (
        resolve_auto_tool_parser,
    )

    resolved_name = resolve_auto_tool_parser(model_name)
    if resolved_name:
        return resolved_name

    raise ValueError(
        f"Could not infer a tool parser from {model_name!r}; "
        "set trtllm_cfg.tool_parser explicitly."
    )


def _build_tool_parser(name: str) -> Any:
    """Instantiate a TRT-LLM tool parser by registered name."""
    # Import lazily and preserve import errors.
    from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

    return ToolParserFactory.create_tool_parser(name)


def _make_parse_tool_calls(tool_parser_instance: Any) -> Any:
    """Return a tool-call parser bound to a specific parser instance."""

    def _parse(text: str, tools: list[dict] | None) -> tuple[str, list[dict[str, Any]]]:
        if not text or not tool_parser_instance.has_tool_call(text):
            return text, []

        # Preserve argument types with TRT-LLM typed tool schemas.
        typed_tools: list[Any] = []
        if tools:
            from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam

            typed_tools = [ChatCompletionToolsParam(**tool) for tool in tools]

        result = tool_parser_instance.detect_and_parse(text, typed_tools)
        calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": item.parameters,
                },
            }
            for item in (result.calls or [])
        ]
        if not calls:
            return text, []
        return result.normal_text.strip(), calls

    return _parse


# ---------------------------------------------------------------------------
#  Prompt construction

# ---------------------------------------------------------------------------


def _to_int_ids(enc: Any) -> list[int]:
    """Coerce chat-template output to a flat list[int]."""
    if hasattr(enc, "input_ids"):  # transformers v5 BatchEncoding
        enc = enc.input_ids
    if len(enc) and isinstance(enc[0], (list, tuple)):  # batch-of-one nesting
        enc = enc[0]
    return [int(t) for t in enc]


def _build_prompt_token_ids(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
    default_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Convert chat messages to token IDs via apply_chat_template (full retokenisation each turn).

    Full retokenisation avoids the gen_token_ids double-counting bug (~4000 tok/turn explosion
    that exhausted context at turn ~32 when prefix accumulation was used).
    """
    template_kwargs: dict[str, Any] = {
        **(default_template_kwargs or {}),
        "add_generation_prompt": True,
        "tokenize": True,
    }
    if tools:
        template_kwargs["tools"] = tools
    return _to_int_ids(tokenizer.apply_chat_template(messages, **template_kwargs))


def _compute_splice_inputs(
    raw_messages: list[dict[str, Any]],
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    default_template_kwargs: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Return preserved and rendered token IDs for the on-policy prefix splice."""
    required_prefix_ids: list[int] = []
    for _m in reversed(raw_messages):
        if _m.get("role") == "assistant" and "prompt_token_ids" in _m:
            required_prefix_ids = list(_m["prompt_token_ids"]) + list(
                _m.get("generation_token_ids") or []
            )
            break

    _last_asst_idx = next(
        (
            i
            for i in reversed(range(len(conversation)))
            if conversation[i].get("role") == "assistant"
        ),
        None,
    )
    _msgs_to_last_asst = (
        conversation[: _last_asst_idx + 1]
        if _last_asst_idx is not None
        else conversation
    )
    _prefix_tkw: dict[str, Any] = {
        **default_template_kwargs,
        "tokenize": True,
        "add_generation_prompt": False,
    }
    if tools:
        _prefix_tkw["tools"] = tools
    template_prefix_ids = _to_int_ids(
        tokenizer.apply_chat_template(_msgs_to_last_asst, **_prefix_tkw)
    )
    return required_prefix_ids, template_prefix_ids


# ---------------------------------------------------------------------------
#  Server lifecycle

# ---------------------------------------------------------------------------


def start_server(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    max_seq_len: int,
    host: str = "0.0.0.0",
    port: int = 0,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    tool_parser: str | None = None,
) -> "tuple[threading.Thread, str, Any]":
    """Start the HTTP server in a daemon thread and return (thread, base_url, server)."""
    import uvicorn
    from nemo_rl.distributed.virtual_cluster import (
        _get_free_port_local,
        _get_node_ip_local,
    )

    if port == 0:
        port = _get_free_port_local()

    node_ip = _get_node_ip_local()
    base_url = f"http://{node_ip}:{port}/v1"

    app = create_app(
        llm,
        tokenizer,
        model_name,
        max_seq_len=max_seq_len,
        default_chat_template_kwargs=default_chat_template_kwargs,
        tool_parser=tool_parser,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info("TRT-LLM HTTP server starting on %s", base_url)

    return thread, base_url, server
