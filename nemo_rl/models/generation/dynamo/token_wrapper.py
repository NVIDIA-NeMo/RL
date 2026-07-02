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

"""OpenAI-compatible token wrapper for NeMo-Gym traffic to Dynamo."""

import asyncio
import json
import threading
from copy import deepcopy
from typing import Any, Optional

from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.utils.prefix_reuse import (
    derive_required_prefix_token_ids,
    messages_to_last_assistant,
    replace_prefix_tokens,
)

_GYM_TOKEN_METADATA_FIELDS = (
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
)


def _coerce_token_id_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of token IDs.")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must contain only integer token IDs.") from e


def _strip_gym_token_metadata(messages: list[Any]) -> list[Any]:
    stripped_messages = deepcopy(messages)
    for message in stripped_messages:
        if isinstance(message, dict):
            for field in _GYM_TOKEN_METADATA_FIELDS:
                message.pop(field, None)
    return stripped_messages


def _chat_template_kwargs(
    request_body: dict[str, Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if tokenizer_chat_template_kwargs is not None:
        if not isinstance(tokenizer_chat_template_kwargs, dict):
            raise ValueError("tokenizer chat_template_kwargs must be a JSON object.")
        kwargs.update(tokenizer_chat_template_kwargs)

    request_kwargs = request_body.get("chat_template_kwargs")
    if request_kwargs is not None:
        if not isinstance(request_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a JSON object.")
        kwargs.update(request_kwargs)

    if "reasoning_effort" in request_body:
        kwargs["reasoning_effort"] = request_body["reasoning_effort"]
    return kwargs


def _request_add_generation_prompt(request_body: dict[str, Any]) -> bool:
    if "add_generation_prompt" in request_body:
        return bool(request_body["add_generation_prompt"])
    return not bool(request_body.get("continue_final_message", False))


def _render_prompt_token_ids(
    *,
    tokenizer: Any,
    request_body: dict[str, Any],
    messages: list[Any],
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
    add_generation_prompt: bool,
) -> list[int]:
    tools = request_body.get("tools")
    if request_body.get("tool_choice") == "none":
        tools = None

    apply_chat_template = type(tokenizer).apply_chat_template
    token_ids = apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        documents=request_body.get("documents"),
        chat_template=request_body.get("chat_template"),
        add_generation_prompt=add_generation_prompt,
        continue_final_message=bool(request_body.get("continue_final_message", False)),
        tokenize=True,
        return_tensors=None,
        return_dict=False,
        **_chat_template_kwargs(request_body, tokenizer_chat_template_kwargs),
    )

    if isinstance(token_ids, list) and (
        not token_ids or not isinstance(token_ids[0], list)
    ):
        return _coerce_token_id_list(token_ids, "prompt token IDs")
    if isinstance(token_ids, list) and len(token_ids) == 1:
        return _coerce_token_id_list(token_ids[0], "prompt token IDs")
    raise ValueError(
        "Dynamo token wrapper expected chat template rendering to return one "
        "list of prompt token IDs."
    )


def _validate_engine_data(response_body: dict[str, Any]) -> None:
    nvext = response_body.get("nvext")
    engine_data = nvext.get("engine_data") if isinstance(nvext, dict) else None
    if not isinstance(engine_data, dict):
        raise ValueError("Dynamo response did not include nvext.engine_data.")

    _coerce_token_id_list(
        engine_data.get("prompt_token_ids"),
        "nvext.engine_data.prompt_token_ids",
    )
    _coerce_token_id_list(
        engine_data.get("completion_token_ids"),
        "nvext.engine_data.completion_token_ids",
    )


def prepare_dynamo_chat_completion_request(
    request_body: dict[str, Any],
    *,
    tokenizer: Any,
    tokenizer_chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Prepare a NeMo-Gym chat-completion request for Dynamo token input."""
    if request_body.get("stream"):
        raise ValueError("Dynamo native token wrapper does not support stream=True.")

    n = request_body.get("n", 1)
    if n is not None and int(n) != 1:
        raise ValueError("Dynamo native token wrapper currently supports only n=1.")

    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Dynamo token wrapper requires chat-completion messages.")

    prepared_body = deepcopy(request_body)
    stripped_messages = _strip_gym_token_metadata(messages)
    prepared_body["messages"] = stripped_messages
    prepared_body.pop("required_prefix_token_ids", None)

    full_prompt_token_ids = _render_prompt_token_ids(
        tokenizer=tokenizer,
        request_body=prepared_body,
        messages=stripped_messages,
        tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
        add_generation_prompt=_request_add_generation_prompt(prepared_body),
    )

    required_prefix_token_ids = derive_required_prefix_token_ids(messages)
    if required_prefix_token_ids is not None:
        prefix_prompt_token_ids = _render_prompt_token_ids(
            tokenizer=tokenizer,
            request_body=prepared_body,
            messages=messages_to_last_assistant(stripped_messages),
            tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
            add_generation_prompt=False,
        )
        full_prompt_token_ids = replace_prefix_tokens(
            tokenizer,
            model_prefix_token_ids=required_prefix_token_ids,
            template_prefix_token_ids=prefix_prompt_token_ids,
            template_token_ids=full_prompt_token_ids,
        )

    nvext = prepared_body.get("nvext")
    if nvext is None:
        nvext = {}
    if not isinstance(nvext, dict):
        raise ValueError("nvext must be a JSON object.")
    nvext = dict(nvext)
    nvext["extra_fields"] = ["engine_data"]
    nvext["token_data"] = full_prompt_token_ids
    prepared_body["nvext"] = nvext

    return prepared_body


class DynamoTokenWrapperServer:
    """Small HTTP server that supplies tokenized chat prompts to Dynamo."""

    def __init__(
        self,
        *,
        dynamo_frontend_base_url: str,
        tokenizer: Any,
        tokenizer_chat_template_kwargs: Optional[dict[str, Any]],
        request_timeout_s: Optional[float],
    ) -> None:
        self.dynamo_frontend_base_url = dynamo_frontend_base_url
        self.tokenizer = tokenizer
        self.tokenizer_chat_template_kwargs = tokenizer_chat_template_kwargs
        self.request_timeout_s = request_timeout_s
        self.base_url: Optional[str] = None
        self.server: Any = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> str:
        """Start the wrapper in a background uvicorn thread."""
        import uvicorn
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {
                "status": "ok",
                "dynamo_frontend_base_url": self.dynamo_frontend_base_url,
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> JSONResponse:
            try:
                request_body = await request.json()
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail="Invalid JSON body.") from e
            if not isinstance(request_body, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Chat completion body must be a JSON object.",
                )

            try:
                prepared_body = await asyncio.to_thread(
                    prepare_dynamo_chat_completion_request,
                    request_body,
                    tokenizer=self.tokenizer,
                    tokenizer_chat_template_kwargs=self.tokenizer_chat_template_kwargs,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            status_code, response_body = await self._forward_chat_completion(
                prepared_body,
                authorization=request.headers.get("authorization"),
            )
            if 200 <= status_code < 300:
                try:
                    _validate_engine_data(response_body)
                except ValueError as e:
                    return JSONResponse(
                        content={"error": {"message": str(e)}},
                        status_code=502,
                    )
            return JSONResponse(content=response_body, status_code=status_code)

        node_ip = _get_node_ip_local()
        free_port = _get_free_port_local()
        self.base_url = f"http://{node_ip}:{free_port}/v1"

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=free_port,
            timeout_keep_alive=120,
        )
        self.server = uvicorn.Server(config=config)
        self.thread = threading.Thread(
            target=self.server.run,
            name="dynamo-token-wrapper",
            daemon=True,
        )
        self.thread.start()
        return self.base_url

    async def _forward_chat_completion(
        self,
        request_body: dict[str, Any],
        *,
        authorization: Optional[str],
    ) -> tuple[int, dict[str, Any]]:
        import aiohttp

        url = f"{self.dynamo_frontend_base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        timeout = (
            aiohttp.ClientTimeout(total=self.request_timeout_s)
            if self.request_timeout_s is not None
            else aiohttp.ClientTimeout(total=None)
        )
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    json=request_body,
                    headers=headers,
                ) as response:
                    response_text = await response.text()
                    if not response_text:
                        return response.status, {}
                    try:
                        response_body = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_body = {"raw": response_text}
                    if not isinstance(response_body, dict):
                        response_body = {"response": response_body}
                    return response.status, response_body
        except asyncio.TimeoutError:
            return 504, {"error": {"message": f"Timed out forwarding to {url}."}}
        except aiohttp.ClientError as e:
            return 502, {
                "error": {
                    "message": f"Failed to forward request to {url}: {type(e).__name__}: {e}"
                }
            }

    def shutdown(self) -> None:
        """Stop the background uvicorn server."""
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=10)
