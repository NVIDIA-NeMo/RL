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

Returns prompt and generated token ids alongside per-token logprobs, and supports
Qwen3 tool calling, DeepSeekR1Parser reasoning, and prefix token splicing.

Under PD disaggregation this endpoint is *leg-aware*. A replica's
``OpenAIDisaggServer`` drives it twice per request:

* ``context_only`` -- prefill only. Returns the handshake and the prompt token
  ids, skipping all post-processing; see :func:`_context_leg_response`.
* ``generation_only`` -- decodes and post-processes as usual, but takes the
  prompt token ids the orchestrator relays from the context leg rather than
  rebuilding them, so the sequence matches the KV that was transferred.
"""

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


def _tokenizer_backend_name(tokenizer: Any) -> str:
    """Return the concrete backend that performs encode/decode operations."""
    backend = getattr(tokenizer, "_tokenizer", None)
    implementation = backend if backend is not None else tokenizer
    implementation_type = type(implementation)
    return f"{implementation_type.__module__}.{implementation_type.__name__}"


def _context_leg_response(
    model_name: str,
    prompt_token_ids: list[int],
    gen: Any,
    disagg_params: Any,
) -> Any:
    """Reply to a ``context_only`` request.

    Prefill materialised KV and at most one token; the disagg server only reads
    the handshake back off this response (plus the prompt token ids, so the
    generation server need not re-tokenize). Nothing here is user-visible, so
    the reasoning/tool/stop-token post-processing is skipped entirely.
    """
    from fastapi.responses import JSONResponse
    from tensorrt_llm.serve.openai_protocol import to_disaggregated_params

    ctx_out = getattr(gen, "disaggregated_params", None)
    if ctx_out is None:
        raise RuntimeError(
            "context leg returned no disaggregated_params; the engine is most "
            "likely missing cache_transceiver_config"
        )

    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": gen.finish_reason,
                "disaggregated_params": to_disaggregated_params(ctx_out).model_dump(),
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_token_ids),
            "completion_tokens": 0,
            "total_tokens": len(prompt_token_ids),
        },
    }

    # The orchestrator asks for the base64 int32 buffer when it wants to relay a
    # string instead of materialising the int list on its event loop.
    if getattr(disagg_params, "return_prompt_token_ids_b64", False):
        import base64

        import numpy as np

        response["prompt_token_ids_b64"] = base64.b64encode(
            np.asarray(prompt_token_ids, dtype=np.int32).tobytes()
        ).decode("ascii")
    else:
        response["prompt_token_ids"] = prompt_token_ids

    return JSONResponse(content=response)


def _request_matches_profile(body: dict[str, Any], profile: dict[str, Any]) -> bool:
    """Whether every sampling param the request pins equals *profile*'s value.

    Params the request leaves unset are not evidence against a profile. This
    server samples from the profile it selects and never from the request, so an
    omitted field simply takes that profile's value -- unlike vLLM, where an
    unset ``top_p`` is resolved from the model's ``generation_config.json`` and
    therefore has to be rejected outright (see ``vllm_worker_async.py``).

    Args:
        body: Decoded chat-completions request body.
        profile: Sampling profile to test, keyed ``temperature``/``top_p``/``top_k``.

    Returns:
        True when the request is compatible with *profile*.
    """
    for key in ("temperature", "top_p", "top_k"):
        requested = body.get(key)
        if requested is not None and requested != profile.get(key):
            return False
    return True


def _build_reasoning_parser(name: str, chat_template_kwargs: dict[str, Any]) -> Any:
    from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

    if name == "deepseek-r1" and "enable_thinking" in chat_template_kwargs:
        from tensorrt_llm.llmapi.reasoning_parser import DeepSeekR1Parser

        return DeepSeekR1Parser(
            reasoning_at_start=bool(chat_template_kwargs["enable_thinking"]),
            chat_template_kwargs=chat_template_kwargs,
        )

    return ReasoningParserFactory.create_reasoning_parser(name, chat_template_kwargs)


def _build_sampling_params(
    sampling_params_cls: Any,
    *,
    sampling_config: dict[str, Any],
    stop_token_ids: list[int] | None,
    max_tokens: int,
) -> Any:
    """Build the TRT-LLM sampling params for one HTTP rollout request.

    Mirrors the direct generate() path
    (``TrtllmAsyncGenerationWorkerImpl._build_sampling_params``) so both paths
    sample from the same distribution for a given generation config.

    Args:
        sampling_params_cls: ``tensorrt_llm.SamplingParams``, injected so this
            helper stays importable and testable without the TRT-LLM runtime.
        sampling_config: NeMo-RL generation config (temperature / top_p / top_k).
        stop_token_ids: Extra stop tokens from the generation config, if any.
        max_tokens: Output cap for this request, already clamped to the context window.

    Returns:
        A ``SamplingParams`` instance to hand to ``llm.generate_async``.
    """
    # TRT-LLM spells "no top-k restriction" as 0, the generation config as null.
    top_k_cfg = sampling_config["top_k"]
    stop_ids = list(stop_token_ids or [])
    return sampling_params_cls(
        temperature=float(sampling_config["temperature"]),
        top_p=float(sampling_config["top_p"]),
        top_k=int(top_k_cfg) if top_k_cfg is not None else 0,
        max_tokens=int(max_tokens),
        stop_token_ids=stop_ids or None,
        # Include generated stop tokens so the adapter can trim tokens and logprobs together.
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


def create_app(
    llm: Any,
    tokenizer: Any,
    model_name: str,
    *,
    max_seq_len: int,
    sampling_config: dict[str, Any],
    val_sampling_config: dict[str, Any] | None = None,
    stop_token_ids: list[int] | None = None,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> "FastAPI":
    """Build a FastAPI application backed by *llm* (``tensorrt_llm.LLM``).

    Args:
        llm: The ``tensorrt_llm.LLM`` engine to serve.
        tokenizer: Tokenizer matching *llm*, used for prompt construction.
        model_name: Model identifier echoed back on responses.
        max_seq_len: Context window of the engine being fronted.
        sampling_config: Train sampling profile, keyed
            ``temperature``/``top_p``/``top_k``.
        val_sampling_config: Validation sampling profile, same keys. ``None``
            (the default) accepts train sampling only, which is what a backend
            without separate validation sampling wants.
        stop_token_ids: Extra stop tokens to trim from generations.
        default_chat_template_kwargs: Server-side chat template defaults.
        tool_parser: Registered TRT-LLM tool parser name, or None to infer.
        reasoning_parser: Registered TRT-LLM reasoning parser name, or None.

    Returns:
        The configured FastAPI application.
    """
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

    if reasoning_parser is not None:
        _build_reasoning_parser(reasoning_parser, _server_template_kwargs)

    # Match TRT-LLM's effective EOS set so this adapter can trim every returned
    # stop token and its logprob together, preserving multi-turn continuity.
    _eos_token_ids: set[int] = set(stop_token_ids or [])

    def _add_eos_token_ids(token_ids: Any) -> None:
        if isinstance(token_ids, int):
            _eos_token_ids.add(token_ids)
        elif token_ids is not None:
            _eos_token_ids.update(
                token_id for token_id in token_ids if isinstance(token_id, int)
            )

    _add_eos_token_ids(tokenizer.eos_token_id)
    generation_config = getattr(llm, "_generation_config", None)
    generation_eos_token_ids = (
        generation_config.get("eos_token_id")
        if isinstance(generation_config, dict)
        else getattr(generation_config, "eos_token_id", None)
    )
    _add_eos_token_ids(generation_eos_token_ids)

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body: dict = await request.json()
        messages: list[dict] = body.get("messages", [])
        tools: list[dict] | None = body.get("tools")
        logprobs_requested = body.get("logprobs", False)

        # Under PD disaggregation a replica's OpenAIDisaggServer drives this
        # endpoint twice per request -- once context_only, once generation_only
        # -- carrying the handshake between the two. The wire model differs from
        # the engine one (opaque_state is bytes in the engine, base64 on the
        # wire), so use TRT-LLM's own converter rather than reproducing it.
        disagg_params = None
        if body.get("disaggregated_params") is not None:
            from tensorrt_llm.serve.openai_protocol import (
                DisaggregatedParams as WireDisaggregatedParams,
            )
            from tensorrt_llm.serve.openai_protocol import to_llm_disaggregated_params

            disagg_params = to_llm_disaggregated_params(
                WireDisaggregatedParams(**body["disaggregated_params"])
            )
        is_context_leg = (
            getattr(disagg_params, "request_type", None) == "context_only"
        )

        # The NeMo-RL generation config, not the request, is the source of truth
        # for sampling params: anything else would sample off-policy and destroy
        # training stability. Validation rollouts are the one exception -- they
        # are stamped with generation.val_temperature / val_top_p, which is
        # metric-only and safe to serve. Accept either profile and serve the one
        # the request pins, mirroring the vLLM server's is_train_sampling /
        # is_val_sampling check. Multi-turn agents issue their own requests, so
        # this handler is the single chokepoint they all pass.
        if _request_matches_profile(body, sampling_config):
            active_sampling_config = sampling_config
        elif val_sampling_config is not None and _request_matches_profile(
            body, val_sampling_config
        ):
            active_sampling_config = val_sampling_config
        else:
            raise AssertionError(
                f"request sampling (temperature={body.get('temperature')!r}, "
                f"top_p={body.get('top_p')!r}, top_k={body.get('top_k')!r}) "
                f"matches neither the train sampling params ({sampling_config}) "
                f"nor the validation sampling params ({val_sampling_config})"
            )

        # Request kwargs override server defaults.
        per_request_kwargs: dict[str, Any] = body.get("chat_template_kwargs") or {}
        effective_template_kwargs = {**_server_template_kwargs, **per_request_kwargs}

        _active_reasoning_parser = (
            _build_reasoning_parser(reasoning_parser, effective_template_kwargs)
            if reasoning_parser is not None
            else None
        )

        try:
            conversation, mm_coroutine, *_ = parse_chat_messages_coroutines(
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

        # On the generation leg the disagg server hands over the exact token ids
        # the context engine built KV for (openai_disagg_service._get_gen_request).
        # Rebuilding them from `messages` could yield a different sequence, which
        # would decode against mismatched KV -- and silently. Prefer what it sent.
        supplied = body.get("prompt_token_ids")
        if supplied is None and body.get("prompt_token_ids_b64"):
            # Same int32 buffer encoding openai_server.py uses on this hop.
            import base64

            import numpy as np

            supplied = np.frombuffer(
                base64.b64decode(body["prompt_token_ids_b64"]), dtype=np.int32
            ).tolist()
        if supplied:
            adj_prompt = list(supplied)

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

        # Serve whichever profile the request pinned (train or validation),
        # built through the shared helper so this path and the direct
        # generate() path keep sampling from the same distribution.
        sampling = _build_sampling_params(
            TrtSamplingParams,
            sampling_config=active_sampling_config,
            stop_token_ids=stop_token_ids,
            max_tokens=max_tokens,
        )
        try:
            output = await llm.generate_async(
                {"prompt_token_ids": adj_prompt},
                sampling_params=sampling,
                disaggregated_params=disagg_params,
            )
        except RequestError as e:
            err = str(e)
            if "max_seq_len" in err or "max_num_tokens" in err:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"context length exceeded: {err}"},
                )
            raise

        gen = output.outputs[0]

        if is_context_leg:
            # Prefill produced KV and at most one token. Everything downstream
            # -- reasoning parsing, tool parsing, stop-token trimming -- is for
            # the completed generation, so skip it and hand the disagg server
            # just what it needs to build the generation leg.
            return _context_leg_response(
                model_name, adj_prompt, gen, disagg_params
            )

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

        finish_reason = "stop"
        if gen.finish_reason is not None:
            fr = str(gen.finish_reason).lower()
            if "length" in fr:
                finish_reason = "length"

        # Split reasoning from answer, if a parser is configured.
        if _active_reasoning_parser is not None:
            parsed = _active_reasoning_parser.parse(gen_text)
            reasoning_content: str = parsed.reasoning_content
            answer_text: str = parsed.content
        else:
            reasoning_content = ""
            answer_text = gen_text

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
            }
            finish_reason = "tool_calls"
        else:
            msg_dict = {
                "role": "assistant",
                "content": answer_text,
                "reasoning_content": reasoning_content,
            }

        # NeMo-Gym reads the rollout fields off the *message*
        # (nemo_rl/environments/nemo_gym.py: a message without
        # generation_token_ids is skipped outright, so a miss loses the whole
        # turn's training data silently). Aggregated serving answers Gym
        # directly, so attach them here.
        #
        # Not under disaggregation: there the reply is re-validated by the
        # disagg server against ChatMessage, which is extra="forbid" and would
        # 400 on these. They ride the declared fields instead
        # (choices[].token_ids, prompt_token_ids, logprobs) and the disagg
        # server's outbound adaptor re-attaches them to the message before Gym
        # ever sees it -- see trtllm_disagg_server._attach_rollout_fields.
        if disagg_params is None:
            msg_dict["prompt_token_ids"] = adj_prompt
            msg_dict["generation_token_ids"] = gen_token_ids
            msg_dict["generation_log_probs"] = gen_logprobs

        response: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": msg_dict,
                    "finish_reason": finish_reason,
                    # Generated token ids. ChatCompletionResponseChoice needs
                    # the matching field upstream (CompletionResponseChoice
                    # already has it) or the disagg server rejects this.
                    "token_ids": gen_token_ids,
                }
            ],
            # Declared on ChatCompletionResponse precisely so a generation
            # server need not re-tokenize the prompt.
            "prompt_token_ids": adj_prompt,
            "usage": {
                "prompt_tokens": len(adj_prompt),
                "completion_tokens": len(gen_token_ids),
                "total_tokens": len(adj_prompt) + len(gen_token_ids),
            },
        }

        if logprobs_requested and gen_logprobs:
            # `token` carries the id rather than the decoded text when asked.
            # ChatCompletionResponseChoice has no token-id field upstream yet, so
            # this declared string field is how the ids survive the disagg
            # server's strict re-validation. Same encoding vLLM uses, which is
            # what NeMo-Gym already parses.
            as_ids = bool(body.get("return_tokens_as_token_ids"))
            response["choices"][0]["logprobs"] = {
                "content": [
                    {
                        "token": (
                            f"token_id:{tid}" if as_ids else tokenizer.decode([tid])
                        ),
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
    *,
    max_seq_len: int,
    sampling_config: dict[str, Any],
    val_sampling_config: dict[str, Any] | None = None,
    stop_token_ids: list[int] | None = None,
    host: str = "0.0.0.0",
    port: int = 0,
    default_chat_template_kwargs: dict[str, Any] | None = None,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> "tuple[threading.Thread, str, Any]":
    """Start the HTTP server in a daemon thread and return (thread, base_url, server).

    Args:
        llm: The ``tensorrt_llm.LLM`` engine to serve.
        tokenizer: Tokenizer matching *llm*.
        model_name: Model identifier echoed back on responses.
        max_seq_len: Context window of the engine being fronted.
        sampling_config: Train sampling profile.
        val_sampling_config: Validation sampling profile, or None to accept
            train sampling only.
        stop_token_ids: Extra stop tokens to trim from generations.
        host: Bind address.
        port: Bind port; 0 picks a free one.
        default_chat_template_kwargs: Server-side chat template defaults.
        tool_parser: Registered TRT-LLM tool parser name, or None to infer.
        reasoning_parser: Registered TRT-LLM reasoning parser name, or None.

    Returns:
        Tuple of (server thread, base URL, uvicorn server).
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

    app = create_app(
        llm,
        tokenizer,
        model_name,
        max_seq_len=max_seq_len,
        sampling_config=sampling_config,
        val_sampling_config=val_sampling_config,
        stop_token_ids=stop_token_ids,
        default_chat_template_kwargs=default_chat_template_kwargs,
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    logger.info("TRT-LLM HTTP server starting on %s", base_url)

    return thread, base_url, server
