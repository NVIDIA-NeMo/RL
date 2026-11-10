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
"""Real-tokenizer continuation parity and Gym training-record compatibility."""

import asyncio
import hashlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from transformers import AutoTokenizer

from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens
from nemo_rl.models.generation.vllm.tito import (
    QWEN3_TEMPLATE_SHA256,
    CapturedQwen3Renderer,
    TitoNotApplicable,
    qwen3_suffix_text,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize(
    "suffix",
    [
        [{"role": "tool", "content": "tests passed", "tool_call_id": "call-1"}],
        [{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}],
        [{"role": "user", "content": "try another patch"}],
        [{"role": "tool", "content": ""}, {"role": "user", "content": "continue"}],
        [{"role": "user", "content": "<tool_response>result</tool_response>"}],
    ],
)
def test_real_tokenizer_matches_existing_splice(tokenizer, thinking, suffix):
    history = [
        {"role": "system", "content": "Fix the repository."},
        {"role": "user", "content": "Run the tests."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Inspect first.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pytest"}'},
                }
            ],
        },
    ]

    def render(items, generation):
        return tokenizer.apply_chat_template(
            items,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=thinking,
        )

    old_text = render(history, False)
    prefix = tokenizer.encode(old_text.removesuffix("\n"), add_special_tokens=False)
    # A captured reasoning span may differ from the text echoed by a harness.
    prefix[5:5] = tokenizer.encode("captured-only reasoning", add_special_tokens=False)
    expected = splice_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=prefix,
        template_prefix_token_ids=tokenizer.encode(old_text, add_special_tokens=False),
        template_token_ids=tokenizer.encode(
            render(history + suffix, True), add_special_tokens=False
        ),
    ).token_ids
    actual = prefix + tokenizer.encode(
        qwen3_suffix_text(suffix, enable_thinking=thinking), add_special_tokens=False
    )
    assert actual == expected
    assert actual[: len(prefix)] == prefix


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "x"}],
        [{"role": "tool", "content": None}],
        [{"role": "user", "content": []}],
    ],
)
def test_suffix_rejects_non_observations(messages):
    with pytest.raises(TitoNotApplicable):
        qwen3_suffix_text(messages, enable_thinking=True)


@pytest.fixture
def parser_stub(monkeypatch):
    # Only the optional vLLM parser/renderer boundary is substituted; capture
    # admission, tokenization, staging, and reconstruction use production code.
    chat = ModuleType("vllm.entrypoints.chat_utils")

    async def parse(messages, model_config, **kwargs):
        return [dict(m, content=m.get("content") or "") for m in messages], None, None

    chat.parse_chat_messages_async = parse
    hf = ModuleType("vllm.renderers.hf")
    hf.resolve_chat_template_content_format = lambda **kwargs: "string"
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.chat_utils", chat)
    monkeypatch.setitem(sys.modules, "vllm.renderers.hf", hf)
    return chat


def make_adapter(tokenizer, *, prefix, template=None):
    from nemo_gym.token_id_capture.replay import replay_context, summarize_replay
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

    history = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "run pytest"},
    ]
    messages = history + [{"role": "tool", "content": "1 passed"}]
    admission = CaptureAdmission(
        capture_key="swe",
        model_call_id="second",
        mode="candidate",
        parent_call_id="first",
        prev_len=len(prefix),
        parent_chain_hash="a" * 64,
        required_prefix_token_ids=prefix,
        requires_empty_tool_content_equivalence=False,
        candidate_replay=summarize_replay(
            replay_context(history, render_digest="b" * 64)
        ),
        request_replay=replay_context(messages, render_digest="b" * 64),
    )

    async def prepare(prompts, params, **kwargs):
        return prompts

    renderer = SimpleNamespace(
        tokenizer=tokenizer,
        model_config=SimpleNamespace(),
        render_cmpl_async=AsyncMock(side_effect=prepare),
    )
    worker = SimpleNamespace(_resolve_admission_prefix=lambda a: list(prefix))
    adapter = CapturedQwen3Renderer(renderer, worker, admission)
    params = SimpleNamespace(
        chat_template=template,
        chat_template_kwargs={},
        chat_template_content_format="string",
        media_io_kwargs=None,
        mm_processor_kwargs=None,
        get_apply_chat_template_kwargs=lambda: dict(
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    )
    return adapter, params, messages


def test_capture_adapter_encodes_only_suffix(tokenizer, parser_stub, monkeypatch):
    prefix = [42, tokenizer.eos_token_id]
    adapter, params, messages = make_adapter(tokenizer, prefix=prefix)
    encoded = []
    encode = tokenizer.encode

    def track(text, **kwargs):
        encoded.append(text)
        return encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", track)
    conversation, prompts = asyncio.run(
        adapter.render_chat_async(
            [messages],
            params,
            object(),
            prompt_extras={},
            skip_mm_cache=False,
        )
    )
    assert encoded == [qwen3_suffix_text(messages[-1:], enable_thinking=False)]
    assert prompts[0]["prompt_token_ids"][: len(prefix)] == prefix
    assert conversation[0] == messages
    assert adapter.decision.storage.mode == "token_in"
    assert adapter.decision.storage.parent_call_id == "first"
    assert adapter.prefix_token_ids == prefix


@pytest.mark.parametrize("case", ["template", "unfinished", "media"])
def test_capture_adapter_falls_back_without_staging(tokenizer, parser_stub, case):
    prefix = [42, tokenizer.eos_token_id] if case != "unfinished" else [42]
    adapter, params, messages = make_adapter(
        tokenizer,
        prefix=prefix,
        template="custom template" if case == "template" else None,
    )
    if case == "media":
        messages[-1]["content"] = [{"type": "image_url", "image_url": {"url": "x"}}]
    with pytest.raises(TitoNotApplicable):
        asyncio.run(
            adapter.render_chat_async(
                [messages],
                params,
                object(),
                prompt_extras={},
                skip_mm_cache=False,
            )
        )
    adapter.renderer.render_cmpl_async.assert_not_called()


def test_capture_storage_failure_is_not_a_fallback(tokenizer, parser_stub):
    adapter, params, messages = make_adapter(
        tokenizer, prefix=[42, tokenizer.eos_token_id]
    )

    def broken(admission):
        raise ValueError("capture digest mismatch")

    adapter.worker._resolve_admission_prefix = broken
    with pytest.raises(ValueError, match="digest mismatch"):
        asyncio.run(
            adapter.render_chat_async(
                [messages],
                params,
                object(),
                prompt_extras={},
                skip_mm_cache=False,
            )
        )


def test_qwen3_template_contract(tokenizer):
    # This guards the manual suffix adapter against an upstream template change.
    assert (
        hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        == QWEN3_TEMPLATE_SHA256
    )


def test_incremental_swe_tokens_round_trip_through_gym_capture(tokenizer):
    from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
    from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize
    from nemo_gym.token_id_capture.staging.records import (
        CallRecord,
        CaptureAdmission,
        RolloutReceipt,
        StagedCallBaseSnapshot,
        StageResult,
    )

    class Sink:
        def __init__(self):
            self.records = []

        def stage(self, record, *, attachments=None):
            self.records.append(record)
            return StageResult(ok=True, staging_key=record.model_call_id)

    sink = Sink()
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 7)
    prompt = tokenizer.encode("Fix the failing test", add_special_tokens=False)
    action = tokenizer.encode(
        '<tool_call>{"name":"bash","arguments":{"command":"pytest"}}</tool_call><|im_end|>',
        add_special_tokens=False,
    )
    first = capture.begin_call(
        CaptureAdmission(capture_key="swe", model_call_id="first", mode="text")
    )
    coords = capture.complete_call(
        first,
        prompt_token_ids=prompt,
        generated_token_ids=action,
        generated_logprobs=[-0.25] * len(action),
    )
    prefix = prompt + action
    suffix = tokenizer.encode(
        qwen3_suffix_text(
            [{"role": "tool", "content": "1 passed"}], enable_thinking=False
        ),
        add_special_tokens=False,
    )
    answer = tokenizer.encode("Fixed.<|im_end|>", add_special_tokens=False)
    second = capture.begin_call(
        CaptureAdmission(
            capture_key="swe",
            model_call_id="second",
            mode="token_in",
            parent_call_id="first",
            prev_len=len(prefix),
            parent_chain_hash=coords.chain_hash,
            required_prefix_token_ids=prefix,
        ),
        prefix_token_ids=prefix,
    )
    capture.complete_call(
        second,
        prompt_token_ids=prefix + suffix,
        generated_token_ids=answer,
        generated_logprobs=[-0.5] * len(answer),
    )
    snapshots, manifest = [], []
    for record in sink.records:
        payload = record.model_dump()
        snapshots.append(
            StagedCallBaseSnapshot(
                **{
                    k: v
                    for k, v in payload.items()
                    if k in StagedCallBaseSnapshot.model_fields
                }
            )
        )
        manifest.append(
            CallRecord(
                staging_key=record.model_call_id,
                response_id="response-" + record.model_call_id,
                **{
                    k: v
                    for k, v in payload.items()
                    if k in CallRecord.model_fields and k != "staging_key"
                },
            )
        )
    row = verify_and_linearize(
        RolloutReceipt(
            capture_key="swe",
            terminal_model_call_id="second",
            manifest=manifest,
            terminal_selection="declared",
        ),
        snapshots,
    )
    assert row.token_ids == prefix + suffix + answer
    assert row.token_mask == [0.0] * len(prompt) + [1.0] * len(action) + [0.0] * len(
        suffix
    ) + [1.0] * len(answer)
    assert row.logprobs == [0.0] * len(prompt) + [-0.25] * len(action) + [0.0] * len(
        suffix
    ) + [-0.5] * len(answer)
    assert row.prompt_len == len(prompt)
    assert row.model_call_ids == ["first", "second"]
