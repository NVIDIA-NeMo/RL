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
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Optional

from nemo_rl.models.generation.openai_server_utils import (
    replace_prefix_tokens,
)

logger = logging.getLogger(__name__)

_TOOL_BOT = "<tool_call>\n"
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL
)


def create_app(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    max_seq_len: int,
    default_chat_template_kwargs: dict[str, Any] | None = None,
) -> "FastAPI":
    """Build a FastAPI application backed by *llm* (``tensorrt_llm.LLM``)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    # default_chat_template_kwargs from config override these built-in defaults.
    _default_template_kwargs: dict[str, Any] = {
        "enable_thinking": True,
        **(default_chat_template_kwargs or {}),
    }

    # reasoning_at_start=True: Qwen3 hybrid template ends gen prompt with "<think>\n"
    # so model output starts inside the reasoning block without a leading <think> tag.
    # parse() is stateless — safe for concurrent requests.
    from tensorrt_llm.llmapi.reasoning_parser import DeepSeekR1Parser
    _reasoning_parser = DeepSeekR1Parser(reasoning_at_start=True)

    # Stop tokens TRT-LLM appends to gen_token_ids but NOT reproduced by apply_chat_template.
    # Must be stripped so seen_token_ids stays a valid prefix of the next turn's prompt_token_ids.
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

        prompt_token_ids = _build_prompt_token_ids(
            messages, tokenizer, tools=tools, default_template_kwargs=_default_template_kwargs
        )

        # Prefix splice: empty required_prefix_ids (no prior assistant turn) → returns template unchanged.
        required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
            messages, tokenizer, tools, _default_template_kwargs
        )

        adj_prompt = replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=required_prefix_ids,
            template_prefix_token_ids=template_prefix_ids,
            template_token_ids=prompt_token_ids,
        )

        max_tokens_requested = body.get("max_tokens") or body.get("max_completion_tokens") or max_seq_len
        remaining_ctx = max(0, max_seq_len - len(adj_prompt))

        # Return HTTP 400 on context exhaustion.
        if remaining_ctx == 0:
            return JSONResponse(
                status_code=400,
                content={"error": f"context length exceeded: prompt ({len(adj_prompt)} tokens) exhausted context window ({max_seq_len})"},
            )

        max_tokens = min(int(max_tokens_requested), remaining_ctx)

        from tensorrt_llm import SamplingParams as TrtSamplingParams

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
        except Exception as e:
            err = str(e)
            if "prompt length" in err or "max_num_tokens" in err or "max_seq_len" in err:
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
            for lp in gen.logprobs:
                if isinstance(lp, (int, float)):
                    gen_logprobs.append(float(lp))
                elif isinstance(lp, dict):
                    gen_logprobs.append(float(next(iter(lp.values())).logprob))
                else:
                    gen_logprobs.append(float(lp))

        # Strip trailing stop tokens TRT-LLM appends — apply_chat_template doesn't reproduce
        # <|endoftext|>, so they'd break seen_token_ids contiguity. Trim logprobs in lockstep.
        while gen_token_ids and gen_token_ids[-1] in _eos_token_ids:
            gen_token_ids.pop()
            if gen_logprobs:
                gen_logprobs.pop()

        gen_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
        # Strip EOS tokens that TRT-LLM may append to output token ids.
        gen_text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")

        finish_reason = "stop"
        if gen.finish_reason is not None:
            fr = str(gen.finish_reason).lower()
            if "length" in fr:
                finish_reason = "length"

        # Qwen3 hybrid: gen prompt ends with "<think>\n"; reasoning_at_start=True splits on "</think>".
        parsed = _reasoning_parser.parse(gen_text)
        reasoning_content: str = parsed.reasoning_content
        answer_text: str = parsed.content

        parsed_tool_calls = _parse_tool_calls(answer_text) if tools else []

        if parsed_tool_calls:
            content_text = _strip_tool_call_tags(answer_text)
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
                {
                    "index": 0,
                    "message": msg_dict,
                    "finish_reason": finish_reason,
                }
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
#  Tool-call parsing — mirrors Qwen3ToolParser.detect_and_parse() exactly
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract <tool_call> blocks; arg serialisation is byte-identical to Qwen3ToolParser."""
    results: list[dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        func_name = obj.get("name", "unknown")
        # json.dumps with ensure_ascii=False matches Qwen3ToolParser.parse_base_json
        arguments = json.dumps(
            obj.get("parameters") or obj.get("arguments", {}),
            ensure_ascii=False,
        )
        results.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments,
            },
        })
    return results


def _strip_tool_call_tags(text: str) -> str:
    """Return text with all <tool_call>...</tool_call> blocks removed (everything before first tag)."""
    idx = text.find(_TOOL_BOT)
    if idx == -1:
        return text.strip()
    return text[:idx].strip()


# ---------------------------------------------------------------------------
#  Prompt construction
# ---------------------------------------------------------------------------

def _to_int_ids(enc: Any) -> list[int]:
    """Coerce apply_chat_template output to flat list[int] (handles transformers v5 BatchEncoding)."""
    if hasattr(enc, "input_ids"):  # v5 BatchEncoding
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

    clean = [_strip_token_fields(m) for m in messages]
    return _to_int_ids(tokenizer.apply_chat_template(clean, **template_kwargs))


def _strip_token_fields(msg: dict[str, Any]) -> dict[str, Any]:
    """Strip NeMo RL on-policy fields before passing to apply_chat_template."""
    skip = {"prompt_token_ids", "generation_token_ids", "generation_log_probs"}
    return {k: v for k, v in msg.items() if k not in skip}


def _compute_splice_inputs(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    default_template_kwargs: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Return (required_prefix_ids, template_prefix_ids) for replace_prefix_tokens.

    required_prefix_ids: last assistant's prompt+gen token IDs (empty on turn 1).
    template_prefix_ids: apply_chat_template up to last assistant turn, used only to count EOS.
    """
    required_prefix_ids: list[int] = []
    for _m in reversed(messages):
        if _m.get("role") == "assistant" and "prompt_token_ids" in _m:
            required_prefix_ids = (
                list(_m["prompt_token_ids"]) + list(_m.get("generation_token_ids") or [])
            )
            break

    _clean = [_strip_token_fields(m) for m in messages]
    _last_asst_idx = next(
        (i for i in reversed(range(len(_clean))) if _clean[i].get("role") == "assistant"),
        None,
    )
    _msgs_to_last_asst = _clean[:_last_asst_idx + 1] if _last_asst_idx is not None else _clean
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
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info("TRT-LLM HTTP server starting on %s", base_url)

    return thread, base_url, server
