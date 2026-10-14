# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Measure real Gym HTTP overhead against a deterministic CPU backend.

Run in an isolated CPU environment (no NeMo-RL project sync or GPU packages):
  uv run --no-project --python /tmp/nrl-http-profile-venv/bin/python \
    python -m tools.http_profile.benchmark --output /tmp/http-profile

Client, backend, and proxy run in separate processes over loopback TCP. The Gym
arms import the checkout's actual VLLMModel and external-capture middleware.
The backend simulates already-produced native token buffers: no model, tokenizer,
GPU transfer, Ray RPC, or TransferQueue. gym_capture uses the real capture core,
an in-memory staging sink, and the real fsync-backed Gym lineage ledger.
Other capture modes are proposed per-call tree capture, not equivalent to the
legacy terminal-chain training objective. Every HTTP response is checked, and
every queue is drained and checked before reporting a successful benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing as mp
import os
import platform
import pstats
import random
import socket
import subprocess
import sys
import tempfile
import time
from array import array
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import aiohttp
import orjson
import uvicorn
from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from tools.http_profile.capture import (
    CallRecord,
    CaptureQueue,
    PrefixTree,
    ShardedCaptureQueue,
)
from tools.http_profile.tracing import G_REQUEST_TRACK, Trace, TraceHTTPMiddleware

ROOT = Path(__file__).resolve().parents[2]
GYM_ROOT = ROOT / "3rdparty/Gym-workspace/Gym"
MODES = (
    "direct",
    "direct_native",
    "gym",
    "gym_capture",
    "opaque",
    "opaque_inline",
    "opaque_thread",
    "opaque_process",
)


@dataclass
class Options:
    sizes: list[int]
    concurrency: list[int]
    modes: list[str]
    turns: int
    generated_tokens: int
    added_tokens: int
    requests: int
    rounds: int
    backend_delay_ms: float
    resolver_delay_ms: float
    resolver_workers: int
    queue_records: int
    queue_mib: int
    stream: bool
    profile: bool
    profile_clock: str
    trace: bool
    output: str


@dataclass
class Fixture:
    bodies: list[bytes]
    prompts: list[list[int]]
    generated: list[list[int]]
    buffers: list[bytes]
    logprobs: bytes
    texts: list[str]


def make_fixture(size: int, options: Options) -> Fixture:
    source = "def run_test():\n    assert result == expected\n$ pytest -q\n"
    text = (source * (size * 4 // len(source) + 1))[: size * 4]
    messages = [{"role": "user", "content": f"profile:{size}\n{text}"}]
    prompt = [index % 32000 for index in range(size)]
    fixture = Fixture(
        [], [], [], [], array("f", [-0.5] * options.generated_tokens).tobytes(), []
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    for turn in range(options.turns):
        fixture.bodies.append(
            orjson.dumps(
                {
                    "model": "profile-model",
                    "messages": messages,
                    "tools": tools,
                    "max_tokens": options.generated_tokens,
                    "temperature": 1.0,
                    "stream": options.stream,
                }
            )
        )
        generated = [1000 + turn] * options.generated_tokens
        content = f"answer {turn}: " + "code " * options.generated_tokens
        fixture.prompts.append(prompt)
        fixture.generated.append(generated)
        fixture.buffers.append(array("I", prompt + generated).tobytes())
        fixture.texts.append(content)
        messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": f"tool result {turn}: " + "data" * options.added_tokens,
            },
        ]
        prompt = prompt + generated + [2000 + turn] * options.added_tokens
    return fixture


def _gym_import_path() -> None:
    if str(GYM_ROOT) not in sys.path:
        sys.path.insert(0, str(GYM_ROOT))


class LegacyCapture:
    """Real Gym capture core, replacing only distributed staging with memory."""

    def __init__(self) -> None:
        # Gym is optional and heavy; only the legacy baseline imports it.
        _gym_import_path()
        from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture

        self.records: dict[str, Any] = {}
        self.capture = RolloutTokenCapture(sink=self, weight_version_fn=lambda: 0)

    def stage(self, record: Any, *, attachments: Any = None) -> Any:
        # Optional Gym types are absent from the opaque/direct arms.
        from nemo_gym.token_id_capture.staging.records import StageResult

        self.records[record.staging_key] = record
        return StageResult(ok=True, staging_key=record.staging_key)

    def complete(
        self, admission_payload: dict[str, Any], fixture: Fixture, turn: int
    ) -> dict[str, Any]:
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        admission = CaptureAdmission.model_validate(admission_payload)
        prefix = [
            token
            for key in admission.staging_chain
            for token in self.records[key].token_ids_delta
        ]
        active = self.capture.begin_call(admission, prefix_token_ids=prefix)
        coords = self.capture.complete_call(
            active,
            prompt_token_ids=fixture.prompts[turn],
            generated_token_ids=fixture.generated[turn],
            generated_logprobs=[-0.5] * len(fixture.generated[turn]),
        )
        if coords.disposition == "capture_failed":
            raise RuntimeError("legacy capture failed")
        return coords.model_dump(mode="json")


def make_gym_app(
    mode: str, backend_url: str, ledger_dir: str, trace: Trace
) -> tuple[FastAPI, Any]:
    # Import real serving modules only in the Gym process, not the load client.
    # Use Gym's normal child-process config transport, avoiding Hydra parsing
    # this benchmark's unrelated command-line arguments.
    os.environ["NEMO_GYM_CONFIG_DICT"] = "{}"
    _gym_import_path()
    from omegaconf import OmegaConf
    from nemo_gym.base_responses_api_model import _CaptureMiddleware
    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import ServerClient
    from nemo_gym.token_id_capture.external_capture import VLLMWorkerCaptureHandler
    from nemo_gym.token_id_capture.lineage import FileLineageStore

    # Optional Gym checkout is added to sys.path above, outside package resolution.
    from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig  # pyrefly: ignore[missing-import]

    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
        global_config_dict=OmegaConf.create({}),
    )
    model = VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=1,
            entrypoint="",
            name="profile-model",
            base_url=f"{backend_url}/v1",
            api_key="benchmark-local-only",
            model="profile-model",
            return_token_id_information=False,
            supply_prefix_token_ids=False,
            uses_reasoning_parser=False,
            uses_interleaved_reasoning=False,
        ),
        server_client=server_client,
    )
    app = model.setup_webserver()
    ledger = None
    if mode == "gym_capture":
        model._external_capture_handler = VLLMWorkerCaptureHandler()
        if trace.enabled:

            class TracedLedger(FileLineageStore):
                def _append(self, rollout_id: str, record: dict, records: list) -> None:
                    started = time.perf_counter()
                    try:
                        super()._append(rollout_id, record, records)
                    finally:
                        trace.span(
                            "ledger append + fsync",
                            started,
                            time.perf_counter(),
                            G_REQUEST_TRACK.get(),
                        )

                def _resolve(self, rollout_id: str, request_items: list) -> Any:
                    started = time.perf_counter()
                    try:
                        return super()._resolve(rollout_id, request_items)
                    finally:
                        trace.span(
                            "ledger resolve",
                            started,
                            time.perf_counter(),
                            G_REQUEST_TRACK.get(),
                        )

            ledger = TracedLedger(ledger_dir)
        else:
            ledger = FileLineageStore(ledger_dir)
        app.add_middleware(
            _CaptureMiddleware,
            store=None,
            model_server_name="profile-model",
            lineage_store=ledger,
            external_staging=True,
            token_capture_enabled=True,
        )
    return app, ledger


def serve(
    sock: socket.socket, role: str, mode: str, backend_url: str, options: Options
) -> None:
    """Child process entry point; process startup is excluded from timings."""
    fixtures = {size: make_fixture(size, options) for size in options.sizes}
    trace = Trace(options.trace, f"{mode}/{role}")
    output_dir = Path(options.output)
    ledger_dir = tempfile.TemporaryDirectory(prefix="nrl-http-ledger-")
    ledger = None
    if role == "proxy" and mode in {"gym", "gym_capture"}:
        app, ledger = make_gym_app(mode, backend_url, ledger_dir.name, trace)
    else:
        app = FastAPI()
    if options.trace:
        app.add_middleware(TraceHTTPMiddleware, trace=trace)
    native_queue = None
    if role == "backend" and mode in {"opaque_thread", "opaque_process"}:
        if mode == "opaque_process" and options.resolver_workers > 1:
            native_queue = ShardedCaptureQueue(
                options.resolver_workers,
                max_records=options.queue_records,
                max_bytes=options.queue_mib * 1024**2,
                delay_s=options.resolver_delay_ms / 1000,
                tracing=options.trace,
            )
        else:
            native_queue = CaptureQueue(
                "thread" if mode == "opaque_thread" else "process",
                max_records=options.queue_records,
                max_bytes=options.queue_mib * 1024**2,
                delay_s=options.resolver_delay_ms / 1000,
                tracing=options.trace,
            )
    production = None
    if role == "backend" and mode == "direct_native":
        from tools.http_profile.native_capture import ProductionCapture

        production = ProductionCapture(app, fixtures, options, trace)
    legacy = LegacyCapture() if role == "backend" and mode == "gym_capture" else None
    inline_tree = PrefixTree()
    calls = 0
    cpu_start = time.process_time()
    case_name = "startup"
    profiling = False
    client: aiohttp.ClientSession | None = None

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal client
        client = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=120),
        )
        yield
        await client.close()
        if role == "proxy" and mode in {"gym", "gym_capture"}:
            from nemo_gym.server_utils import (
                get_global_aiohttp_client,
                is_global_aiohttp_client_setup,
            )

            if is_global_aiohttp_client_setup():
                await get_global_aiohttp_client().close()
        if native_queue is not None:
            await asyncio.to_thread(native_queue.close)
        if production is not None:
            await asyncio.to_thread(production.capture.shutdown)
        ledger_dir.cleanup()

    app.router.lifespan_context = lifespan

    @app.get("/bench/health")
    async def health() -> dict[str, bool]:
        return {"ready": True}

    @app.post("/bench/stop")
    async def stop() -> dict[str, bool]:
        server.should_exit = True
        return {"stopping": True}

    @app.post("/bench/reset")
    async def reset(request: Request) -> dict[str, bool]:
        nonlocal inline_tree, calls, cpu_start, case_name, profiling
        command = await request.json()
        if native_queue is not None:
            await asyncio.to_thread(native_queue.reset)
        if production is not None:
            await asyncio.to_thread(production.reset)
        inline_tree = PrefixTree()
        calls = 0
        if legacy is not None:
            legacy.records.clear()
        case_name = command["case_name"]
        trace.reset()
        if options.profile:
            # Profiling is a separate run; its overhead is not used for speedups.
            import yappi

            yappi.clear_stats()
            yappi.set_clock_type(options.profile_clock)
            yappi.start(builtins=True)
            profiling = True
        cpu_start = time.process_time()
        return {"ready": True}

    @app.get("/bench/stats")
    async def stats() -> dict[str, Any]:
        nonlocal profiling
        cpu_before_drain = time.process_time() - cpu_start
        queue_stats = (
            await asyncio.to_thread(native_queue.drain) if native_queue else {}
        )
        if production is not None:
            queue_stats = await asyncio.to_thread(production.drain)
        result = {
            "cpu_before_drain_s": cpu_before_drain,
            "cpu_including_drain_s": time.process_time() - cpu_start,
            "calls": calls,
            "queue": queue_stats,
            "inline_tree": inline_tree.stats(),
            "legacy_staged_records": len(legacy.records) if legacy else 0,
            "trace_events": trace.events,
        }
        if ledger is not None:
            # Inspect the actual ledger after load; never on the request path.
            rows = [
                row
                for _, _, records in ledger._ledger_cache.values()
                for row in records
            ]
            result["ledger_records"] = len(rows)
            result["ledger_failures"] = [row for row in rows if "failure_reason" in row]
        if profiling:
            import yappi

            yappi.stop()
            profile_path = output_dir / f"{mode}-{role}-{case_name}.pstat"
            yappi.get_func_stats().save(str(profile_path), type="pstat")
            stream = io.StringIO()
            pstats.Stats(str(profile_path), stream=stream).sort_stats(
                "cumulative"
            ).print_stats(45)
            pstats.Stats(str(profile_path), stream=stream).sort_stats(
                "tottime"
            ).print_stats(45)
            profile_path.with_suffix(".txt").write_text(stream.getvalue())
            profiling = False
        return result

    def capture_native(record: CallRecord) -> None:
        if mode not in {"opaque_inline", "opaque_thread", "opaque_process"}:
            return
        started = time.perf_counter()
        record = replace(record, submitted_at=started)
        tid = G_REQUEST_TRACK.get()
        if mode == "opaque_inline":
            if options.resolver_delay_ms:
                time.sleep(options.resolver_delay_ms / 1000)
            inline_tree.insert(record)
            trace.span("tree insert inline", started, time.perf_counter(), tid)
        elif native_queue is not None:
            trace.flow("capture handoff", record.trace_id, "s", started, tid)
            accepted = native_queue.submit(record)
            trace.span(
                "capture enqueue",
                started,
                time.perf_counter(),
                tid,
                bytes=record.nbytes,
                accepted=accepted,
            )

    if role == "backend":

        @app.post("/v1/chat/completions")
        async def complete(request: Request, body: dict = Body()) -> Response:
            nonlocal calls
            calls += 1
            size = int(
                body["messages"][0]["content"][:40].split("\n", 1)[0].split(":")[1]
            )
            turn = sum(message["role"] == "assistant" for message in body["messages"])
            fixture = fixtures[size]
            call_id = f"call-{calls}"
            record = CallRecord(
                request.headers.get("x-rollout-id", "uncorrelated"),
                call_id,
                fixture.buffers[turn],
                len(fixture.prompts[turn]),
                fixture.logprobs,
                time.perf_counter(),
                request.headers.get("x-profile-call", ""),
            )
            response = {
                "id": call_id,
                "object": "chat.completion",
                "created": 0,
                "model": "profile-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": fixture.texts[turn],
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(fixture.prompts[turn]),
                    "completion_tokens": options.generated_tokens,
                    "total_tokens": len(fixture.buffers[turn]) // 4,
                },
            }
            if body.get("stream"):

                async def events() -> AsyncIterator[bytes]:
                    content = fixture.texts[turn]
                    chunk_size = math.ceil(len(content) / 8)
                    for start in range(0, len(content), chunk_size):
                        if options.backend_delay_ms:
                            await asyncio.sleep(options.backend_delay_ms / 8000)
                        chunk = {
                            "id": call_id,
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": "profile-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": content[start : start + chunk_size]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield b"data: " + orjson.dumps(chunk) + b"\n\n"
                    if production is not None:
                        await production.generate(size, turn, call_id)
                    capture_native(record)
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(events(), media_type="text/event-stream")
            if options.backend_delay_ms:
                await asyncio.sleep(options.backend_delay_ms / 1000)
            if legacy is not None:
                capture_started = time.perf_counter()
                response["ng_commit_coords"] = await asyncio.to_thread(
                    legacy.complete, body["ng_capture"], fixture, turn
                )
                trace.span(
                    "legacy capture (awaited)",
                    capture_started,
                    time.perf_counter(),
                    G_REQUEST_TRACK.get(),
                )
            if production is not None:
                await production.generate(size, turn, call_id)
            capture_native(record)
            return Response(orjson.dumps(response), media_type="application/json")

    elif mode not in {"gym", "gym_capture"}:

        @app.post("/ng-rollout/{rollout_id}/training-token-capture/v1/chat/completions")
        async def proxy(request: Request, rollout_id: str) -> StreamingResponse:
            nonlocal calls
            calls += 1
            assert client is not None
            # Opaque request/response bodies; correlation is the only added header.
            upstream = await client.post(
                f"{backend_url}/v1/chat/completions",
                data=request.stream(),
                headers={
                    "content-type": request.headers["content-type"],
                    "x-rollout-id": rollout_id,
                    **(
                        {"x-profile-call": request.headers.get("x-profile-call", "")}
                        if options.trace
                        else {}
                    ),
                },
            )
            return StreamingResponse(
                upstream.content.iter_any(),
                status_code=upstream.status,
                headers={"content-type": upstream.headers["content-type"]},
                background=BackgroundTask(upstream.release),
            )

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            loop="uvloop",
            http="httptools",
            access_log=False,
            log_level="error",
            timeout_keep_alive=120,
        )
    )
    server.run(sockets=[sock])


async def start_server(
    role: str, mode: str, backend_url: str, options: Options
) -> tuple[Any, str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(256)
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    process = mp.get_context("spawn").Process(
        target=serve, args=(sock, role, mode, backend_url, options)
    )
    process.start()
    sock.close()
    async with aiohttp.ClientSession() as client:
        for _ in range(600):
            if not process.is_alive():
                raise RuntimeError(
                    f"{mode}/{role} failed to start (exit {process.exitcode})"
                )
            try:
                async with client.get(f"{url}/bench/health") as response:
                    if response.status == 200:
                        return process, url
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.1)
    process.terminate()
    process.join(timeout=10)
    raise TimeoutError(f"{mode}/{role} startup timed out")


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value / 100
    low = math.floor(index)
    return ordered[low] + (ordered[math.ceil(index)] - ordered[low]) * (index - low)


def response_content(payload: bytes, streaming: bool) -> str:
    if not streaming:
        return orjson.loads(payload)["choices"][0]["message"]["content"]
    text = []
    for line in payload.splitlines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        chunk = orjson.loads(line[6:])
        text.extend(
            choice["delta"].get("content", "") for choice in chunk.get("choices", [])
        )
    return "".join(text)


async def load(
    url: str,
    mode: str,
    fixture: Fixture,
    *,
    concurrency: int,
    episodes: int,
    tag: str,
    streaming: bool,
    trace: Trace | None = None,
) -> dict[str, Any]:
    latencies: list[float] = []
    first_bytes: list[float] = []
    response_bytes = 0

    async def lane(index: int) -> None:
        nonlocal response_bytes
        # Independent cookie jars and sessions match independent harnesses.
        async with aiohttp.ClientSession(
            cookie_jar=aiohttp.CookieJar(unsafe=True),
            timeout=aiohttp.ClientTimeout(total=120),
        ) as client:
            for episode in range(episodes):
                rollout = f"{tag}-{index}-{episode}"
                path = (
                    "/v1/chat/completions"
                    if mode in {"direct", "direct_native"}
                    else (
                        f"/ng-rollout/{rollout}/training-token-capture/v1/chat/completions"
                    )
                )
                if mode == "direct_native":
                    async with client.post(
                        url + "/nrl/capture/register",
                        json={
                            "api_key": rollout,
                            "rollout_id": rollout,
                            "temperature": 1,
                            "top_p": 1,
                        },
                        headers={"Authorization": "Bearer benchmark-control"},
                    ) as registered:
                        registered.raise_for_status()
                        await registered.read()
                for turn, body in enumerate(fixture.bodies):
                    started = time.perf_counter()
                    trace_key = f"{rollout}/{turn}"
                    tid = trace.track(trace_key) if trace else 0
                    if trace:
                        trace.flow("HTTP dispatch", trace_key, "s", started, tid)
                    chunks = []
                    first = None
                    async with client.post(
                        url + path,
                        data=body,
                        headers={
                            "content-type": "application/json",
                            "x-rollout-id": rollout,
                            **(
                                {"Authorization": f"Bearer {rollout}"}
                                if mode == "direct_native"
                                else {}
                            ),
                            **({"x-profile-call": trace_key} if trace else {}),
                        },
                    ) as response:
                        async for chunk in response.content.iter_any():
                            if first is None:
                                first = time.perf_counter()
                            chunks.append(chunk)
                        ended = time.perf_counter()
                        payload = b"".join(chunks)
                        if response.status != 200:
                            raise RuntimeError(
                                f"{mode} HTTP {response.status}: {payload[:2000]!r}"
                            )
                    if trace:
                        trace.span(
                            "client request",
                            started,
                            ended,
                            tid,
                            request_bytes=len(body),
                            response_bytes=len(payload),
                        )
                        trace.span(
                            "client until first body", started, first or ended, tid
                        )
                    if response_content(payload, streaming) != fixture.texts[turn]:
                        raise RuntimeError("response content changed")
                    if b"ng_commit_coords" in payload or b"prompt_token_ids" in payload:
                        raise RuntimeError(
                            "training metadata leaked into agent response"
                        )
                    latencies.append((ended - started) * 1000)
                    first_bytes.append(((first or ended) - started) * 1000)
                    response_bytes += len(payload)

    started = time.perf_counter()
    cpu_started = time.process_time()
    await asyncio.gather(*(lane(index) for index in range(concurrency)))
    duration = time.perf_counter() - started
    return {
        "requests": len(latencies),
        "wall_s": duration,
        "client_cpu_s": time.process_time() - cpu_started,
        "requests_per_s": len(latencies) / duration,
        "latency_ms": {f"p{p}": percentile(latencies, p) for p in (50, 95, 99)},
        "first_body_byte_ms": {
            f"p{p}": percentile(first_bytes, p) for p in (50, 95, 99)
        },
        "request_bytes_per_turn": [len(body) for body in fixture.bodies],
        "response_bytes": response_bytes,
        "samples_ms": latencies,
    }


def power_state() -> dict[str, str]:
    """Record raw power/thermal reports without changing the machine's settings."""
    if sys.platform != "darwin":
        return {}
    return {
        name: subprocess.run(
            ["pmset", "-g", argument], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        for name, argument in (("battery", "batt"), ("thermal", "therm"))
    }


async def run(options: Options) -> None:
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "options": asdict(options),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "power_at_start": power_state(),
        "benchmark_source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
        "commits": {
            name: subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            for name, path in (("nrl", ROOT), ("gym", GYM_ROOT))
        },
        "dependencies": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
        "limitations": [
            "Synthetic token counts; actual HTTP byte lengths are reported separately.",
            "No tokenizer, GPU generation, routed experts, TransferQueue, Ray RPC, or training.",
            "Native buffers are prebuilt, modeling capture after engine buffer ownership.",
            "Legacy and native capture have different downstream training representations.",
            "Closed-loop load on loopback TCP; not a production saturation or network estimate.",
        ],
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for repeat, mode in [
        (repeat, mode)
        for repeat in range(options.rounds)
        for mode in (
            options.modes if repeat % 2 == 0 else list(reversed(options.modes))
        )
    ]:
        processes = []
        try:
            backend, backend_url = await start_server("backend", mode, "", options)
            processes.append(backend)
            urls = [backend_url]
            if mode not in {"direct", "direct_native"}:
                proxy, proxy_url = await start_server(
                    "proxy", mode, backend_url, options
                )
                processes.append(proxy)
                urls.append(proxy_url)
            cases = [
                (size, concurrency, repeat)
                for size in options.sizes
                for concurrency in options.concurrency
            ]
            random.Random(17).shuffle(cases)
            async with aiohttp.ClientSession() as control:
                for size, concurrency, repeat in cases:
                    fixture = make_fixture(size, options)
                    tag = f"p{size}-c{concurrency}-r{repeat}"
                    await load(
                        urls[-1],
                        mode,
                        fixture,
                        concurrency=concurrency,
                        episodes=1,
                        tag=f"warm-{tag}",
                        streaming=options.stream,
                    )
                    for url in urls:
                        async with control.post(
                            f"{url}/bench/reset", json={"case_name": tag}
                        ) as response:
                            response.raise_for_status()
                            await response.read()
                    trace = Trace(options.trace, "load client")
                    case_power_before = power_state()
                    result = await load(
                        urls[-1],
                        mode,
                        fixture,
                        concurrency=concurrency,
                        episodes=max(
                            1,
                            math.ceil(options.requests / (concurrency * options.turns)),
                        ),
                        tag=tag,
                        streaming=options.stream,
                        trace=trace if options.trace else None,
                    )
                    servers = []
                    for url in urls:
                        async with control.get(f"{url}/bench/stats") as response:
                            response.raise_for_status()
                            servers.append(await response.json())
                    trace_events = list(trace.events)
                    for server_stats in servers:
                        trace_events.extend(server_stats.pop("trace_events"))
                        trace_events.extend(
                            server_stats["queue"].pop("trace_events", [])
                        )
                    if options.trace:
                        (output / f"{mode}-{tag}.trace.json").write_bytes(
                            orjson.dumps(
                                {"traceEvents": trace_events, "displayTimeUnit": "ms"}
                            )
                        )
                    result["power_before"] = case_power_before
                    result["power_after"] = power_state()
                    backend_stats = servers[0]
                    if backend_stats["calls"] != result["requests"]:
                        raise RuntimeError(
                            "backend call count differs from client completions"
                        )
                    queue_stats = backend_stats["queue"]
                    if queue_stats and (
                        queue_stats["submitted"] != result["requests"]
                        or queue_stats["rejected"]
                        or queue_stats["completed"] != result["requests"]
                        or queue_stats["stored_calls"] != result["requests"]
                        or queue_stats["failures"]
                        or queue_stats["pending"]
                    ):
                        failure_path = output / f"{mode}-{tag}.failure.json"
                        failure_path.write_text(
                            json.dumps({"load": result, "servers": servers}, indent=2)
                            + "\n"
                        )
                        raise RuntimeError(
                            f"incomplete capture; details: {failure_path}"
                        )
                    if mode == "gym_capture" and (
                        backend_stats["legacy_staged_records"] != result["requests"]
                        or servers[-1]["ledger_failures"]
                    ):
                        raise RuntimeError("legacy capture is incomplete")
                    total_cpu = sum(
                        server["cpu_including_drain_s"] for server in servers
                    )
                    if mode == "opaque_process":
                        total_cpu += queue_stats["resolver_total_cpu_s"]
                    row = {
                        "mode": mode,
                        "prompt_tokens": size,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        **result,
                        "servers": servers,
                        "server_cpu_ms_per_request": 1000
                        * total_cpu
                        / result["requests"],
                    }
                    rows.append(row)
                    with (output / "measurements.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(
                        f"{mode:15s} P={size:6d} C={concurrency:2d} r={repeat} "
                        f"p50={result['latency_ms']['p50']:8.3f}ms "
                        f"p95={result['latency_ms']['p95']:8.3f}ms "
                        f"RPS={result['requests_per_s']:8.1f} "
                        f"cpu={row['server_cpu_ms_per_request']:7.3f}ms "
                        f"drain={queue_stats.get('drain_ms', 0):7.2f}ms",
                        flush=True,
                    )
        finally:
            # Graceful exit lets multiprocessing release queue semaphores.
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as control:
                for url in reversed(urls if processes else []):
                    with contextlib.suppress(aiohttp.ClientError, TimeoutError):
                        async with control.post(f"{url}/bench/stop") as response:
                            await response.read()
            for process in reversed(processes):
                process.join(timeout=15)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--added-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--backend-delay-ms", type=float, default=0)
    parser.add_argument("--resolver-delay-ms", type=float, default=0)
    parser.add_argument(
        "--resolver-workers",
        type=int,
        default=1,
        help="rollout-sharded processes for opaque_process",
    )
    parser.add_argument("--queue-records", type=int, default=256)
    parser.add_argument("--queue-mib", type=int, default=64)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-clock", choices=["cpu", "wall"], default="cpu")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="write Perfetto-compatible traces; use a separate diagnostic run",
    )
    parser.add_argument("--output", required=True)
    options = Options(**vars(parser.parse_args()))
    if (
        min(
            *options.sizes,
            *options.concurrency,
            options.turns,
            options.generated_tokens,
            options.requests,
            options.rounds,
            options.queue_records,
            options.queue_mib,
            options.resolver_workers,
        )
        < 1
    ):
        parser.error("sizes, counts, and queue limits must be positive")
    if (
        min(options.added_tokens, options.backend_delay_ms, options.resolver_delay_ms)
        < 0
    ):
        parser.error("added tokens and delays must be nonnegative")
    if options.resolver_workers > options.queue_records:
        parser.error("each resolver must have at least one queue record of capacity")
    if (Path(options.output) / "measurements.jsonl").exists():
        parser.error("output already contains measurements; use a fresh directory")
    asyncio.run(run(options))


if __name__ == "__main__":
    main()
