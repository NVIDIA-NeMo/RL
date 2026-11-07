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

import pytest

pytestmark = pytest.mark.vllm
pytest.importorskip("vllm")

from transformers import AutoTokenizer  # noqa: E402

from nemo_rl.models.generation.dynamo.vllm.render_pool import RenderPool  # noqa: E402

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
        pool = RenderPool(SPEC, tiny_qwen3_model_path, workers=1)
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
