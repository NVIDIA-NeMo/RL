"""OpenAI-compatible HTTP server wrapping a ``tensorrt_llm.LLM`` instance.

Designed to be started inside a ``TrtllmGenerationWorker`` Ray actor so that
NeMo Gym's harbor agent can call ``/v1/chat/completions`` for multi-turn
rollout generation.  The response includes the custom fields that
``NemoGymLLM`` expects (``prompt_token_ids``, ``generation_token_ids``,
``generation_log_probs``).

Supports OpenAI-format tool calling: accepts ``tools`` in the request,
passes them to ``apply_chat_template``, parses ``<tool_call>`` tags from
the generated text, and returns structured ``tool_calls`` in the response.

The server runs **uvicorn** in a daemon thread — identical to how vLLM's
async worker exposes its HTTP server in NeMo RL.
"""

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def create_app(
    llm: Any,
    tokenizer: Any,
    model_name: str,
) -> "FastAPI":
    """Build a FastAPI application backed by *llm* (``tensorrt_llm.LLM``)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

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
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 4096
        logprobs_requested = body.get("logprobs", False)

        prompt_token_ids = _build_prompt_token_ids(messages, tokenizer, tools=tools)

        from tensorrt_llm import SamplingParams as TrtSamplingParams

        sampling = TrtSamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            logprobs=True,
        )

        outputs = await asyncio.to_thread(
            llm.generate,
            [{"prompt_token_ids": prompt_token_ids}],
            sampling_params=sampling,
        )

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

        gen_text = tokenizer.decode(gen_token_ids, skip_special_tokens=False)

        finish_reason = "stop"
        if gen.finish_reason is not None:
            fr = str(gen.finish_reason).lower()
            if "length" in fr:
                finish_reason = "length"

        parsed_tool_calls = _parse_tool_calls(gen_text) if tools else []

        if parsed_tool_calls:
            content_text = _strip_tool_call_tags(gen_text)
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": content_text or None,
                "tool_calls": parsed_tool_calls,
                "prompt_token_ids": prompt_token_ids,
                "generation_token_ids": gen_token_ids,
                "generation_log_probs": gen_logprobs,
            }
            finish_reason = "tool_calls"
        else:
            msg_dict = {
                "role": "assistant",
                "content": gen_text,
                "prompt_token_ids": prompt_token_ids,
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
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(gen_token_ids),
                "total_tokens": len(prompt_token_ids) + len(gen_token_ids),
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
#  Tool-call parsing
# ---------------------------------------------------------------------------

def _parse_tool_calls(gen_text: str) -> list[dict[str, Any]]:
    """Extract ``<tool_call>`` blocks from generated text into OpenAI format."""
    results: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(gen_text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        func_name = obj.get("name", "unknown")
        arguments = obj.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
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
    """Remove ``<tool_call>...</tool_call>`` blocks, return remaining text."""
    stripped = _TOOL_CALL_RE.sub("", text).strip()
    return stripped


# ---------------------------------------------------------------------------
#  Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt_token_ids(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Convert chat messages to token IDs, honoring on-policy correction.

    If the last assistant message carries ``prompt_token_ids`` and
    ``generation_token_ids`` (attached by ``NemoGymLLM`` for on-policy
    correction), we reuse those exact token IDs as a prefix and only
    tokenize the messages that follow.

    Handles ``tool_calls`` on assistant messages and ``role: "tool"``
    messages — these are passed through to ``apply_chat_template`` which
    Qwen3's Jinja template handles natively.
    """
    prefix_ids: Optional[list[int]] = None
    new_messages_start: Optional[int] = None

    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "assistant" and "prompt_token_ids" in msg:
            prefix_ids = list(msg["prompt_token_ids"])
            gen_ids = msg.get("generation_token_ids") or []
            prefix_ids = prefix_ids + list(gen_ids)
            new_messages_start = idx + 1
            break

    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
    }
    if tools:
        template_kwargs["tools"] = tools

    if prefix_ids is not None and new_messages_start is not None:
        remaining = messages[new_messages_start:]
        if remaining:
            clean = [_strip_token_fields(m) for m in remaining]
            suffix_ids = tokenizer.apply_chat_template(clean, **template_kwargs)
            return prefix_ids + suffix_ids
        return prefix_ids

    clean = [_strip_token_fields(m) for m in messages]
    return tokenizer.apply_chat_template(clean, **template_kwargs)


def _strip_token_fields(msg: dict[str, Any]) -> dict[str, Any]:
    """Remove NeMo RL on-policy fields before passing to chat template.

    Preserves ``tool_calls`` on assistant messages and ``tool_call_id`` on
    tool messages — these are needed by the chat template.
    """
    skip = {"prompt_token_ids", "generation_token_ids", "generation_log_probs"}
    return {k: v for k, v in msg.items() if k not in skip}


# ---------------------------------------------------------------------------
#  Server lifecycle
# ---------------------------------------------------------------------------

def start_server(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    host: str = "0.0.0.0",
    port: int = 0,
) -> "tuple[threading.Thread, str, Any]":
    """Start the HTTP server in a daemon thread and return (thread, base_url, server).

    Mirrors the pattern from ``VllmAsyncGenerationWorker._setup_vllm_server``.
    """
    import uvicorn
    from nemo_rl.distributed.virtual_cluster import (
        _get_free_port_local,
        _get_node_ip_local,
    )

    if port == 0:
        port = _get_free_port_local()

    node_ip = _get_node_ip_local()
    base_url = f"http://{node_ip}:{port}/v1"

    app = create_app(llm, tokenizer, model_name)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info("TRT-LLM HTTP server starting on %s", base_url)

    return thread, base_url, server
