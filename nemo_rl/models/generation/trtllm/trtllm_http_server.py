"""OpenAI-compatible HTTP server wrapping a ``tensorrt_llm.LLM`` instance.

Started inside a ``TrtllmGenerationWorker`` Ray actor so NeMo Gym's harbor
agent can call ``/v1/chat/completions`` for multi-turn rollout generation.
The response includes the custom fields ``NemoGymLLM`` expects
(``prompt_token_ids``, ``generation_token_ids``, ``generation_log_probs``).

Supports OpenAI-format tool calling: accepts ``tools`` in the request,
passes them through ``apply_chat_template``, parses ``<tool_call>`` tags
from the generated text, and returns structured ``tool_calls``.

Reasoning and tool parsing use TRT-LLM's native parsers directly:
  - tensorrt_llm.llmapi.reasoning_parser.DeepSeekR1Parser (reasoning_at_start=True)
  - Qwen3ToolParser logic (exact token/regex/json.dumps behaviour)

Runs uvicorn in a daemon thread.
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

# Qwen3 tool-call delimiters — must match Qwen3ToolParser exactly so that
# json.dumps normalisation and token boundaries are identical to the native server.
_TOOL_BOT = "<tool_call>\n"
_TOOL_EOT = "\n</tool_call>"
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL
)


def _replace_prefix_tokens(
    tokenizer: Any,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> list[int]:
    """Splice ground-truth model tokens into the re-templated prompt.

    Port of nemo_rl.models.generation.vllm.vllm_worker_async._replace_prefix_tokens.

    When the SWE-bench agent truncates history, apply_chat_template renders
    the truncated messages differently from the actual token sequence the model
    accumulated.  This function replaces the template-tokenized prefix with the
    actual model token IDs from prior turns, keeping only the new-turn suffix
    from the template.

    Args:
        model_prefix_token_ids: Accumulated ground-truth token IDs from all
            prior turns (adj_prompt_prev + gen_token_ids_prev).  Empty on the
            first turn → returns template_token_ids unchanged.
        template_prefix_token_ids: apply_chat_template output for messages up
            to and including the last historical assistant turn, WITHOUT the
            generation prompt.  Used to locate the splice boundary (the last
            EOS token before the new turn).
        template_token_ids: Full apply_chat_template output for all messages
            INCLUDING the generation prompt.  The suffix after the splice point
            is taken from here.

    Returns:
        Spliced token ID list: model_prefix_token_ids (up to last EOS) +
        template_token_ids (from last EOS in template_prefix onward).
    """
    if not model_prefix_token_ids:
        return template_token_ids

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Tokenizer must have an EOS token ID"

    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids[-1] == eos_token_id:
        model_cut_end -= 1

    # Find the splice boundary: the N-th <|im_end|> in template_token_ids, where N =
    # count of <|im_end|> in template_prefix_token_ids.  This is robust to the Qwen3
    # last_query_index rendering inconsistency: when the last message in the full history
    # is role='user' without <tool_response> wrapping, the template sets last_query_index
    # to that last index, stripping <think> blocks from earlier assistants in the full
    # rendering.  The result is len(template_prefix) > len(template_token_ids).
    # The count of <|im_end|> tokens = count of turn boundaries, which is invariant to
    # whether thinking content is included, so the N-th EOS correctly locates the end of
    # the last historical assistant turn in template_token_ids regardless of alignment.
    count_needed = template_prefix_token_ids.count(eos_token_id)
    count_seen = 0
    template_cut_start = -1
    for pos, tid in enumerate(template_token_ids):
        if tid == eos_token_id:
            count_seen += 1
            if count_seen == count_needed:
                template_cut_start = pos
                break

    assert template_cut_start >= 0, (
        f"EOS token #{count_needed} not found in template_token_ids "
        f"(only found {count_seen} EOS tokens total)!\n"
        f"Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}\n\n"
        f"Template token IDs (everything that was sent to the model endpoint): {template_token_ids}\n\n"
        f"Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}\n\n"
        f"Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"
    )

    return model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]


def create_app(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    max_seq_len: int = 131072,
) -> "FastAPI":
    """Build a FastAPI application backed by *llm* (``tensorrt_llm.LLM``)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    # Instantiate native TRT-LLM reasoning parser once per server.
    # DeepSeekR1Parser with reasoning_at_start=True matches the Qwen3 hybrid
    # template that uses enable_thinking=True: the generation prompt ends with
    # "<|im_start|>assistant\n<think>\n", so model output begins inside the
    # reasoning block without a leading <think> tag.
    # parse() is stateless (state is only used by parse_delta for streaming),
    # so a single shared instance is safe for concurrent requests.
    from tensorrt_llm.llmapi.reasoning_parser import DeepSeekR1Parser
    _reasoning_parser = DeepSeekR1Parser(reasoning_at_start=True)

    # Token IDs that TRT-LLM appends to gen_token_ids as stop tokens but that
    # apply_chat_template does NOT reproduce when re-rendering historical turns.
    # These must be stripped from gen_token_ids so that seen_token_ids stays
    # a valid prefix of the next turn's prompt_token_ids (contiguity check).
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

        # ── DEBUG: print incoming message history ──────────────────────────
        n_asst = sum(1 for m in messages if m.get("role") == "assistant")
        turn = n_asst + 1  # turn we are about to generate
        # print(f"[TRTLLM_DEBUG] turn={turn} total_msgs={len(messages)} has_tools={tools is not None}", flush=True)
        # for i, m in enumerate(messages):
        #     role = m.get("role", "?")
        #     content_snippet = str(m.get("content", ""))[:80].replace("\n", "\\n")
        #     has_rc = "reasoning_content" in m
        #     rc_snippet = str(m.get("reasoning_content", ""))[:60].replace("\n", "\\n")
        #     has_tc = "tool_calls" in m
        #     has_ptids = "prompt_token_ids" in m
        #     ptids_len = len(m["prompt_token_ids"]) if has_ptids else 0
        #     tool_call_id = m.get("tool_call_id", "")
        #     print(f"[TRTLLM_DEBUG]   msg[{i}] role={role} has_reasoning_content={has_rc} rc={rc_snippet!r} has_tool_calls={has_tc} has_prompt_token_ids={has_ptids}({ptids_len}) tool_call_id={tool_call_id!r} content={content_snippet!r}", flush=True)
        # ───────────────────────────────────────────────────────────────────

        prompt_token_ids = _build_prompt_token_ids(messages, tokenizer, tools=tools)

        # ── Prefix splice: mirrors vLLM's NeMoRLOpenAIChatRequestMixin.model_post_init ──
        # Read prompt_token_ids + generation_token_ids from the last assistant message
        # in the request.  The server sets these in msg_dict every turn; nemo_gym
        # passes them back in subsequent requests.  If no assistant message carries
        # them (e.g. agent did a hard history reset), required_prefix_ids stays empty
        # and _replace_prefix_tokens returns the template unchanged — matching vLLM's
        # behaviour exactly (required_prefix_token_ids stays None → no splice).
        required_prefix_ids: list[int] = []
        for _m in reversed(messages):
            if _m.get("role") == "assistant" and "prompt_token_ids" in _m:
                required_prefix_ids = (
                    list(_m["prompt_token_ids"]) + list(_m.get("generation_token_ids") or [])
                )
                break
        # print(
        #     f"[TRTLLM_DEBUG] turn={turn} required_prefix_len={len(required_prefix_ids)}",
        #     flush=True,
        # )

        # template_prefix_ids: tokenize messages up to and including the last
        # historical assistant turn, WITHOUT gen prompt — locates the EOS splice point.
        _clean = [_strip_token_fields(m) for m in messages]
        _last_asst_idx = next(
            (i for i in reversed(range(len(_clean))) if _clean[i].get("role") == "assistant"),
            None,
        )
        _msgs_to_last_asst = _clean[:_last_asst_idx + 1] if _last_asst_idx is not None else _clean
        _prefix_tkw: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": False,
            "enable_thinking": True,
        }
        if tools:
            _prefix_tkw["tools"] = tools
        template_prefix_ids = _to_int_ids(tokenizer.apply_chat_template(_msgs_to_last_asst, **_prefix_tkw))

        adj_prompt = _replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=required_prefix_ids,
            template_prefix_token_ids=template_prefix_ids,
            template_token_ids=prompt_token_ids,
        )
        if adj_prompt != prompt_token_ids:
            pass
            # print(
            #     f"[TRTLLM_DEBUG] turn={turn} prefix_splice: template={len(prompt_token_ids)} "
            #     f"→ spliced={len(adj_prompt)} (delta={len(adj_prompt)-len(prompt_token_ids):+d})",
            #     flush=True,
            # )
        # ─────────────────────────────────────────────────────────────────────

        # ── DEBUG: print decoded prompt tail to verify template output ─────
        # try:
        #     decoded_prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        #     prompt_tail = decoded_prompt[-400:].replace("\n", "\\n")
        #     print(f"[TRTLLM_DEBUG] turn={turn} prompt_tokens={len(prompt_token_ids)} prompt_tail={prompt_tail!r}", flush=True)
        #     if turn > 1:
        #         first_asst = decoded_prompt.find("<|im_start|>assistant")
        #         second_asst = decoded_prompt.find("<|im_start|>assistant", first_asst + 1)
        #         if second_asst != -1:
        #             snippet = decoded_prompt[second_asst:second_asst + 300].replace("\n", "\\n")
        #             print(f"[TRTLLM_DEBUG]   2nd_asst_block[:300]={snippet!r}", flush=True)
        # except Exception as e:
        #     print(f"[TRTLLM_DEBUG] prompt decode error: {e}", flush=True)
        # ───────────────────────────────────────────────────────────────────

        max_tokens_requested = body.get("max_tokens") or body.get("max_completion_tokens") or max_seq_len
        remaining_ctx = max(0, max_seq_len - len(adj_prompt))

        # When context is exhausted, return HTTP 400 matching vLLM's behavior.
        # vLLM raises "maximum context length" ValueError → HTTP 400 → app.py catches it
        # → _create_empty_chat_completion() with no generation_token_ids → nemo_gym.py skips it.
        if remaining_ctx == 0:
            # print(
            #     f"[TRTLLM_DEBUG] turn={turn} max_tokens_requested={max_tokens_requested} "
            #     f"remaining_ctx=0 prompt_tokens={len(adj_prompt)} → HTTP 400 (context exhausted)",
            #     flush=True,
            # )
            return JSONResponse(
                status_code=400,
                content={"error": f"context length exceeded: prompt ({len(adj_prompt)} tokens) exhausted context window ({max_seq_len})"},
            )

        max_tokens = min(int(max_tokens_requested), remaining_ctx)
        # print(f"[TRTLLM_DEBUG] turn={turn} max_tokens_requested={max_tokens_requested} remaining_ctx={remaining_ctx} max_tokens_used={max_tokens}", flush=True)

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

        # Strip trailing EOS/stop tokens that TRT-LLM appends to raw output.
        # apply_chat_template re-renders historical turns as "<|im_end|>\n" (no
        # <|endoftext|>), so any extra stop tokens here cause seen_token_ids to
        # diverge from the next turn's prompt_token_ids → contiguity assert.
        # gen_logprobs is trimmed in lockstep — one logprob entry per token.
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

        # Use native TRT-LLM DeepSeekR1Parser (reasoning_at_start=True) for splitting.
        # This matches Qwen3 hybrid template with enable_thinking=True: the generation
        # prompt already ends with "<think>\n" so the model output starts inside the
        # reasoning block. parse() uses str.partition("</think>") with NO whitespace
        # stripping — byte-exact, matching what vLLM's reasoning parser does.
        parsed = _reasoning_parser.parse(gen_text)
        reasoning_content: str = parsed.reasoning_content
        answer_text: str = parsed.content

        parsed_tool_calls = _parse_tool_calls(answer_text) if tools else []

        # ── DEBUG: print generation result ─────────────────────────────────
        has_think_close = "</think>" in gen_text
        # print(
        #     f"[TRTLLM_DEBUG] turn={turn} prompt_tokens={len(prompt_token_ids)} gen_tokens={len(gen_token_ids)} "
        #     f"finish_reason={finish_reason} has_</think>={has_think_close} "
        #     f"n_tool_calls={len(parsed_tool_calls)}",
        #     flush=True,
        # )
        # print(f"[TRTLLM_DEBUG]   gen_text[:200]={gen_text[:200].replace(chr(10), '\\n')!r}", flush=True)
        # print(f"[TRTLLM_DEBUG]   reasoning[:100]={reasoning_content[:100].replace(chr(10), '\\n')!r}", flush=True)
        # print(f"[TRTLLM_DEBUG]   answer[:100]={answer_text[:100].replace(chr(10), '\\n')!r}", flush=True)
        # ───────────────────────────────────────────────────────────────────

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
            # print(f"[TRTLLM_DEBUG]   → tool_calls path: content={str(content_text or '')[:80].replace(chr(10), '\\n')!r} n_tool_calls={len(parsed_tool_calls)} tool_names={[tc.get('function', {}).get('name') for tc in parsed_tool_calls]}", flush=True)
            # print(f"[TRTLLM_DEBUG]   → reasoning_content[:80]={reasoning_content[:80].replace(chr(10), '\\n')!r}", flush=True)
        else:
            msg_dict = {
                "role": "assistant",
                "content": answer_text,
                "reasoning_content": reasoning_content,
                "prompt_token_ids": adj_prompt,
                "generation_token_ids": gen_token_ids,
                "generation_log_probs": gen_logprobs,
            }
            # print(f"[TRTLLM_DEBUG]   → no-tool path: content[:80]={answer_text[:80].replace(chr(10), '\\n')!r}", flush=True)
            # print(f"[TRTLLM_DEBUG]   → reasoning_content[:80]={reasoning_content[:80].replace(chr(10), '\\n')!r}", flush=True)

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
    """Extract ``<tool_call>`` blocks, replicating Qwen3ToolParser behaviour.

    Uses the same delimiters, regex, and json.dumps(ensure_ascii=False)
    normalisation as tensorrt_llm.serve.tool_parser.Qwen3ToolParser so that
    argument serialisation is byte-identical across turns.
    """
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
    """Return text with all ``<tool_call>...</tool_call>`` blocks removed.

    Mirrors Qwen3ToolParser: normal_text is everything before the first bot_token.
    """
    idx = text.find(_TOOL_BOT)
    if idx == -1:
        return text.strip()
    return text[:idx].strip()


# ---------------------------------------------------------------------------
#  Prompt construction
# ---------------------------------------------------------------------------

def _to_int_ids(enc: Any) -> list[int]:
    """Coerce apply_chat_template output to a flat list[int].

    transformers 5.x (TokenizersBackend) returns a BatchEncoding from
    apply_chat_template(tokenize=True), NOT a flat list[int]; TRT-LLM's
    executor asserts isinstance(prompt_token_ids[0], int). Extract the ids
    and flatten/coerce.
    """
    if hasattr(enc, "input_ids"):  # BatchEncoding
        enc = enc.input_ids
    if hasattr(enc, "ids"):  # tokenizers.Encoding
        enc = enc.ids
    if len(enc) and isinstance(enc[0], (list, tuple)):  # batched / nested
        enc = enc[0]
    if len(enc) and hasattr(enc[0], "ids"):  # list[Encoding]
        enc = enc[0].ids
    return [int(t) for t in enc]


def _build_prompt_token_ids(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Convert chat messages to token IDs via apply_chat_template.

    Full retokenisation every turn (no incremental prefix accumulation).
    prefix_opt was disabled because gen_token_ids (~3000–8000 tok, including
    the full <think>...</think> block) was appended on top of apply_chat_template
    output that already renders <think>reasoning</think> for all historical
    assistant turns → double-counting → ~4000 tok/turn explosion → context
    exhausted at turn ~32.

    Handles ``tool_calls`` on assistant messages and ``role: "tool"``
    messages — passed through to apply_chat_template, handled natively by
    the Qwen3 Jinja template.
    """
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        # Qwen3 thinking models require this to produce "<|im_start|>assistant\n<think>\n"
        # as the generation prompt.  Without it the model generates EOS immediately.
        "enable_thinking": True,
    }
    if tools:
        template_kwargs["tools"] = tools

    n_asst_hist = sum(1 for m in messages if m.get("role") == "assistant")
    # print(f"[TRTLLM_DEBUG] _build_prompt_token_ids n_msgs={len(messages)} n_asst_hist={n_asst_hist}", flush=True)
    clean = [_strip_token_fields(m) for m in messages]
    return _to_int_ids(tokenizer.apply_chat_template(clean, **template_kwargs))


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
    max_seq_len: int = 131072,
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

    app = create_app(llm, tokenizer, model_name, max_seq_len=max_seq_len)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info("TRT-LLM HTTP server starting on %s", base_url)

    return thread, base_url, server
