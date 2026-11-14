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
"""The Dynamo frontend's render pool, over a tiny Qwen3 checkpoint.

The pool is spawned as in the frontend, so the spawn context, pickling and
parser registration are exercised; renders are checked against the
checkpoint's own chat template, replies against vLLM's Qwen3 parsers.
"""

import asyncio
import json
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest

pytestmark = pytest.mark.vllm
pytest.importorskip("vllm")

from transformers import AutoTokenizer  # noqa: E402

from nemo_rl.models.generation.dynamo.vllm.render_pool import (  # noqa: E402
    RenderPool,
    _wait_for_settings,
    _Worker,
)
from nemo_rl.models.generation.prefix_recovery.restore import (  # noqa: E402
    _classify_restored_prefix_terminal,
)
from tests.unit.models.generation.prefix_restore_cases import (  # noqa: E402
    RESTORE_CASES,
    RestoreCase,
)
from tests.unit.models.generation.test_prefix_restore_cases import (  # noqa: E402
    _engine_text,
    _ids,
    _tokenizer,
)

MAX_MODEL_LEN = 512
SPEC = {
    # The engine's own args, as it publishes them at registration.
    "engine_args": {"max_model_len": MAX_MODEL_LEN},
    "generation": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": None,
        "val_temperature": 0.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "vllm_cfg": {
            "http_server_serving_chat_kwargs": {
                "tool_parser": "hermes",
                "reasoning_parser": "qwen3",
                "enable_auto_tools": True,
            }
        },
    },
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
MESSAGES = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call-0", "content": "sunny"},
]


def _template_ids(tokenizer, messages, *, add_generation_prompt: bool) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _pool(spec: dict, model_dir: str) -> RenderPool:
    pool = RenderPool(workers=1)
    pool.build(spec, model_dir)
    return pool


def test_the_pool_renders_and_answers_with_the_checkpoints_template_and_parsers(
    tiny_qwen3_model_path,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_qwen3_model_path)
    reply_text = (
        "<think>\nchecking\n</think>\n\n<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Lyon"}}\n</tool_call>'
    )
    reply_ids = tokenizer.encode(reply_text, add_special_tokens=False)

    async def serve():
        pool = _pool(SPEC, tiny_qwen3_model_path)
        try:
            info = await pool.info()
            rendered = await pool.render(
                {
                    "model": "model",
                    "messages": MESSAGES,
                    "tools": TOOLS,
                    "max_tokens": 64,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    # Gym's echo of the last assistant turn, without capture.
                    "required_prefix_token_ids": [1, 2, 3],
                }
            )
            reply = await pool.respond(
                rendered.request,
                rendered.template_token_ids,
                {"text": reply_text, "token_ids": reply_ids, "finish_reason": "stop"},
            )
        finally:
            pool.shutdown()
        return info, rendered, reply

    info, rendered, reply = asyncio.run(serve())

    # Turns end on <|im_end|>, which the splice in the frontend cuts at.
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in info["turn_end_token_ids"]
    assert info["max_model_len"] == MAX_MODEL_LEN
    # The call, and the history up to its last assistant turn the splice cuts against.
    assert rendered.template_token_ids == _template_ids(
        tokenizer, MESSAGES, add_generation_prompt=True
    )
    assert rendered.history_token_ids == _template_ids(
        tokenizer, MESSAGES[:2], add_generation_prompt=False
    )
    assert rendered.required_prefix_token_ids == [1, 2, 3]
    assert rendered.request_max_tokens == 64
    assert "messages" not in rendered.request
    assert (rendered.admission, rendered.multimodal) == (None, None)
    # The reply through vLLM's own reasoning and tool parsers.
    (choice,) = reply["choices"]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning"].strip() == "checking"
    (tool_call,) = choice["message"]["tool_calls"]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Lyon"}


def test_workers_start_before_their_settings_and_build_when_they_arrive(
    tiny_qwen3_model_path,
) -> None:
    async def serve():
        pool = RenderPool(workers=2)
        try:
            start = asyncio.ensure_future(pool.start())
            # Both come up before any settings exist, as at the frontend's launch.
            async with asyncio.timeout(60):
                while len(pool._executor._processes) < 2:
                    await asyncio.sleep(0.01)
            assert not start.done()
            pool.build(SPEC, tiny_qwen3_model_path)
            await start
            return (await pool.info())["max_model_len"]
        finally:
            pool.shutdown()

    assert asyncio.run(serve()) == MAX_MODEL_LEN


def test_a_worker_waits_for_its_settings_until_its_pool_is_shut_down(tmp_path) -> None:
    path = tmp_path / "settings.json"
    with ThreadPoolExecutor(max_workers=1) as worker:
        waiting = worker.submit(_wait_for_settings, str(path))
        time.sleep(0.3)
        assert not waiting.done()
        path.write_text(json.dumps({"spec": SPEC, "model_dir": "/snapshots/model"}))
        assert waiting.result(timeout=10) == (SPEC, "/snapshots/model")
    # Shutdown removes the pool's directory; a worker left waiting would leak.
    with pytest.raises(SystemExit):
        _wait_for_settings(str(tmp_path / "removed" / "settings.json"))


def test_a_dead_worker_fails_only_its_calls(tiny_qwen3_model_path) -> None:
    body = {
        "model": "model",
        "messages": MESSAGES[:1],
        "max_tokens": 8,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    async def serve():
        pool = _pool(SPEC, tiny_qwen3_model_path)
        try:
            await pool.info()
            (pid,) = pool._executor._processes
            # As the OOM killer would.
            os.kill(pid, signal.SIGKILL)
            with pytest.raises(BrokenProcessPool):
                await pool.render(body)
            # The next call gets a fresh pool, not the broken one.
            return await pool.render(body)
        finally:
            pool.shutdown()

    rendered = asyncio.run(serve())

    assert rendered.template_token_ids


def test_workers_that_cannot_start_fail_the_call_by_name(tiny_qwen3_model_path):
    vllm_cfg = {"http_server_serving_chat_kwargs": {"reasoning_parser": "no_such"}}
    spec = {**SPEC, "generation": {**SPEC["generation"], "vllm_cfg": vllm_cfg}}

    async def serve():
        pool = _pool(spec, tiny_qwen3_model_path)
        try:
            await pool.info()
        finally:
            pool.shutdown()

    with pytest.raises(RuntimeError, match="before serving a call"):
        asyncio.run(serve())


def _nemotron_vl_loader() -> str:
    """Run in a render worker: the class vLLM decodes nemotron_vl videos with."""
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY

    return VIDEO_LOADER_REGISTRY.name2class["nemotron_vl"].__name__


def test_workers_decode_videos_with_nemo_rls_loader(tiny_qwen3_model_path) -> None:
    # The worker decodes a Gym call's videos, so it needs the engine's loader.
    vllm_cfg = {
        **SPEC["generation"]["vllm_cfg"],
        "video": {
            "sampling_style": "nemotron_vl",
            "num_frames": 8,
            "temporal_patch_size": 2,
        },
    }
    spec = {**SPEC, "generation": {**SPEC["generation"], "vllm_cfg": vllm_cfg}}

    async def serve():
        pool = _pool(spec, tiny_qwen3_model_path)
        try:
            return await pool._submit(_nemotron_vl_loader)
        finally:
            pool.shutdown()

    assert asyncio.run(serve()) == "TorchCodecNemotronVLVideoBackend"


class _RecordingParser:
    """vLLM's response parser: the content is the text, and the IDs are kept."""

    def __init__(self) -> None:
        self.token_ids: list[int] | None = None

    def parse(
        self, model_output, request, *, enable_auto_tools, model_output_token_ids
    ):
        self.token_ids = list(model_output_token_ids)
        return None, model_output, []

    def count_reasoning_tokens(self, token_ids) -> int:
        return 0

    def set_prompt_token_ids(self, prompt_token_ids) -> None:
        # vLLM hands the parser the prompt before it parses the reply.
        pass


@pytest.fixture(scope="module")
def reply_worker(tiny_qwen3_model_path) -> _Worker:
    """One render worker, in this process, to build replies with."""
    return _Worker(SPEC, tiny_qwen3_model_path)


@pytest.mark.parametrize("style", ["metaspace", "byte_level"])
@pytest.mark.parametrize("case", RESTORE_CASES, ids=lambda case: case.name)
def test_a_resumed_calls_reply_matches_the_uninterrupted_reply(
    reply_worker, monkeypatch, case: RestoreCase, style: str
) -> None:
    """The cases the native server passes in test_prefix_restore_cases.py."""
    tokenizer = _tokenizer(style)
    parser = _RecordingParser()
    monkeypatch.setattr(reply_worker, "tokenizer", tokenizer)
    # The serving chat's reply step decodes the saved tokens with its renderer's.
    monkeypatch.setattr(reply_worker.serving_chat.renderer, "tokenizer", tokenizer)
    monkeypatch.setattr(reply_worker, "_parser", lambda chat: parser)
    prompt = _ids(tokenizer, case.prompt)
    output = _ids(tokenizer, case.output)
    saved, tail = output[: case.saved], output[case.saved :]
    # The engine ran the saved rows as its prompt and decoded only the tail.
    terminal = _classify_restored_prefix_terminal(
        prompt_token_ids=prompt + saved,
        prefix_token_count=len(saved),
        requested_output_tokens=len(output) + 8,
        model_max_tokens=4096,
        terminal_finish_reason=case.finish_reason if not tail else None,
        terminal_stop_reason=case.stop_reason if not tail else None,
    )
    tail_text = ""
    if tail:
        tail_text = _engine_text(
            tokenizer,
            prompt + saved,
            tail,
            case,
            min_tokens=max(0, case.min_tokens - len(saved)),
        )

    reply = reply_worker.respond(
        {
            "model": "model",
            "min_tokens": case.min_tokens,
            "stop": list(case.stop) or None,
            "include_stop_str_in_output": case.include_stop_str_in_output,
        },
        # The prompt the frontend sent, which the engine swapped.
        prompt,
        {
            "text": tail_text,
            "token_ids": tail,
            "finish_reason": case.finish_reason,
            "stop_reason": case.stop_reason,
        },
        {
            "prompt_token_ids": prompt + saved,
            "token_ids": saved,
            "terminal": None
            if terminal is None
            else {
                "prefix_token_count": terminal.prefix_token_count,
                "reason": terminal.reason,
                "finish_reason": terminal.finish_reason,
                "stop_reason": terminal.stop_reason,
            },
        },
    )

    (choice,) = reply["choices"]
    assert choice["message"]["content"] == _engine_text(
        tokenizer, prompt, output, case, min_tokens=case.min_tokens
    )
    assert choice["finish_reason"] == case.finish_reason
    # Tool-call and reasoning parsers see the whole output, saved tokens first.
    assert parser.token_ids == output
    usage = reply["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (
        len(prompt),
        len(output),
    )
