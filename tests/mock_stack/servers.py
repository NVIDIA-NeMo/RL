# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException
from nemo_gym.token_id_capture.staging import RolloutTokenCapture, install_capture
from nemo_gym.token_id_capture.staging.records import CaptureAdmission
from pydantic import BaseModel

from nemo_rl.data_plane.interfaces import DataPlaneClient
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from tests.mock_stack.components import Generated, Turn
from tests.mock_stack.config import Prompt


@runtime_checkable
class TurnGenerator(Protocol):
    async def generate(self, turn: Turn) -> Generated: ...


class ChatRequest(BaseModel, extra="allow"):
    messages: list[dict]
    ng_capture: CaptureAdmission
    stream: Literal[False] = False


@dataclass(frozen=True)
class Call:
    prompt: str
    sibling: int
    turn: int
    capture_key: str
    started: float
    finished: float
    weight_digest: str


class GenerationServer:
    """OpenAI transport around replaceable computation and real Gym capture."""

    def __init__(self, generation: TurnGenerator, prompts: list[Prompt]):
        self.generation = generation
        self.prompts = {prompt.id: prompt for prompt in prompts}
        self.calls: list[Call] = []
        self.capture: RolloutTokenCapture | None = None
        self.source: TQTokenSource | None = None
        self.weight_version = 0
        self.app = FastAPI()
        self.app.post("/v1/chat/completions")(self.chat)
        self.app.get("/health")(lambda: {"status": "ok"})
        self.app.get("/v1/models")(
            lambda: {
                "object": "list",
                "data": [
                    {"id": "cpu", "object": "model", "created": 0, "owned_by": "test"}
                ],
            }
        )

    def install_token_capture(self, capture: RolloutTokenCapture) -> None:
        self.capture = capture

    def setup_token_capture(self, plane: DataPlaneClient, partition: str) -> None:
        self.source = TQTokenSource(plane, staging_partition=partition)
        install_capture(
            self,
            sink=TQTokenSink(plane, staging_partition=partition),
            weight_version_fn=lambda: self.weight_version,
        )

    def set_rollout_weight_version(self, version: int) -> None:
        self.weight_version = version

    async def chat(self, body: ChatRequest) -> dict:
        if self.capture is None or self.source is None:
            raise HTTPException(503, "Token capture is not installed")
        users = [m["content"] for m in body.messages if m["role"] == "user"]
        if not users:
            raise HTTPException(422, "A prompt ID is required")
        content = users[0]
        prompt_id = (
            content if isinstance(content, str) else "".join(p["text"] for p in content)
        )
        prompt = self.prompts.get(prompt_id)
        sibling_match = re.search(r"_g(\d+)(?:-a\d+)?$", body.ng_capture.rollout_id)
        if prompt is None or sibling_match is None:
            raise HTTPException(422, "Unknown prompt or sibling identity")
        sibling = int(sibling_match[1])
        turn_index = 1 + sum(m["role"] == "assistant" for m in body.messages)
        if sibling >= len(prompt.turn_seconds) or turn_index > prompt.turns:
            raise HTTPException(422, "Request exceeds the configured workload")
        admission = body.ng_capture
        prefix = (
            await asyncio.to_thread(
                self.source.fetch_prefix_token_ids, admission.staging_chain
            )
            if admission.staging_chain
            else list(admission.required_prefix_token_ids)
        )
        call = self.capture.begin_call(admission, prefix_token_ids=prefix)
        started = time.monotonic()
        generated = await self.generation.generate(
            Turn(prompt_id, sibling, turn_index, prompt.turn_seconds[sibling])
        )
        prompt_tokens = prefix + list(prompt_id.encode()) + [turn_index]
        coords = await asyncio.to_thread(
            self.capture.complete_call,
            call,
            prompt_token_ids=prompt_tokens,
            generated_token_ids=list(generated.token_ids),
            generated_logprobs=list(generated.logprobs),
        )
        self.calls.append(
            Call(
                prompt_id,
                sibling,
                turn_index,
                admission.rollout_id,
                started,
                time.monotonic(),
                generated.weight_digest,
            )
        )
        return {
            "id": admission.model_call_id,
            "object": "chat.completion",
            "created": 0,
            "model": "cpu",
            "ng_commit_coords": coords.model_dump(mode="json"),
            "prompt_token_ids": prompt_tokens,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"{prompt_id}_g{sibling}_t{turn_index}",
                                "type": "function",
                                "function": {
                                    "name": "increment_counter",
                                    "arguments": '{"count":1}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated.token_ids),
                "total_tokens": len(prompt_tokens) + len(generated.token_ids),
            },
        }
