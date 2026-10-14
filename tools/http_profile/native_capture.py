# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Production capture/codec benchmark adapter; only engine and TQ are simulated."""

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

from nemo_rl.data_plane.direct_capture import NativeCallStore
from nemo_rl.experience.direct_capture import (
    DirectCapture,
    DirectCaptureConfig,
    NativeCall,
)
from nemo_rl.experience.direct_http import install_direct_control
from nemo_rl.experience.vllm_http_client import _AsyncLLMHTTPClient


class _MemoryDP:
    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}

    def put_samples(self, *, sample_ids, partition_id, fields, tags) -> None:
        self.rows[sample_ids[0]] = fields

    def clear_samples(self, *, sample_ids, partition_id) -> None:
        for key in sample_ids:
            del self.rows[key]


class ProductionCapture:
    def __init__(self, app, fixtures, options, trace) -> None:
        self.memory = _MemoryDP()
        self.trace = trace
        store = NativeCallStore(self.memory, "benchmark", router_replay=False)

        def stage(call: NativeCall) -> dict[str, Any]:
            started = time.perf_counter()
            result = store.stage(call)
            trace.span(
                "native tensor staging",
                started,
                time.perf_counter(),
                threading.get_native_id(),
            )
            return result

        self.capture = DirectCapture(
            DirectCaptureConfig(
                workers=options.resolver_workers,
                max_pending_calls=options.queue_records,
            ),
            stage,
            lambda: 0,
            clear=store.clear,
        )
        install_direct_control(
            app,
            capture=lambda: self.capture,
            auth_token=lambda: "benchmark-control",
            timeout=lambda: 60,
        )
        self.outputs = {}
        for size, fixture in fixtures.items():
            for turn, tokens in enumerate(fixture.generated):
                self.outputs[size, turn] = SimpleNamespace(
                    prompt_token_ids=fixture.prompts[turn],
                    finished=True,
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            token_ids=tokens,
                            logprobs=[
                                {token: SimpleNamespace(logprob=-0.5)}
                                for token in tokens
                            ],
                            finish_reason="stop",
                            routed_experts=None,
                        )
                    ],
                )

    async def generate(self, size: int, turn: int, call_id: str) -> None:
        outputs = self.outputs

        class Engine:
            model_config = renderer = input_processor = vllm_config = None

            async def generate(self, prompt, sampling_params, request_id, **kwargs):
                yield outputs[prompt]

            async def abort(self, request_id):
                raise RuntimeError("unexpected benchmark abort")

        bridge = _AsyncLLMHTTPClient(Engine(), asyncio.get_running_loop())
        params = SimpleNamespace(
            logprobs=None, output_kind=SimpleNamespace(name="FINAL_ONLY")
        )
        async for _ in bridge.generate((size, turn), params, call_id):
            pass

    def drain(self) -> dict[str, Any]:
        started = time.perf_counter()
        receipts = [
            self.capture.finalize(key, timeout=60)
            for key in list(self.capture._rollouts)
        ]
        failures = [receipt["failure"] for receipt in receipts if receipt["failure"]]
        count = sum(len(receipt["manifest"]) for receipt in receipts)
        return {
            "submitted": count,
            "completed": count,
            "stored_calls": len(self.memory.rows),
            "pending": 0,
            "rejected": 0,
            "failures": failures,
            "drain_ms": (time.perf_counter() - started) * 1000,
        }

    def reset(self) -> None:
        stats = self.drain()
        if stats["failures"]:
            raise RuntimeError(stats)
        for key in list(self.capture._rollouts):
            self.capture.release(key)
        self.memory.rows.clear()
