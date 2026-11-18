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
import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from transformers import AutoTokenizer

from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens
from nemo_rl.models.generation.vllm.tito import (
    QWEN3_TEMPLATE_SHA256,
    CapturedTitoRenderer,
    Qwen3IncrementalTokenizer,
    TitoNotApplicable,
    TitoTrace,
    preprocess_captured_tito,
    qwen3_suffix_text,
    resolve_captured_continuation,
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
    chat = ModuleType("vllm.renderers.chat_utils")

    async def parse(messages, model_config, **kwargs):
        return [dict(m, content=m.get("content") or "") for m in messages], None, None

    chat.parse_chat_messages_async = parse
    hf = ModuleType("vllm.renderers.hf")
    hf.resolve_chat_template = lambda tokenizer, **kwargs: tokenizer.get_chat_template(
        kwargs.get("chat_template"), tools=kwargs.get("tools")
    )
    hf.resolve_chat_template_content_format = lambda **kwargs: "string"
    monkeypatch.setitem(sys.modules, "vllm.renderers.chat_utils", chat)
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
    adapter = CapturedTitoRenderer(
        renderer, worker, admission, Qwen3IncrementalTokenizer(), TitoTrace()
    )
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


@pytest.mark.nemo_gym
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
    assert adapter.continuation.decision.storage.mode == "token_in"
    assert adapter.continuation.decision.storage.parent_call_id == "first"
    assert adapter.continuation.prefix_token_ids == prefix


@pytest.mark.parametrize(
    "case", ["template", "unfinished", "media", "generation_cut", "effective_template"]
)
@pytest.mark.nemo_gym
def test_capture_adapter_falls_back_without_staging(tokenizer, parser_stub, case):
    prefix = [42, tokenizer.eos_token_id] if case != "unfinished" else [42]
    adapter, params, messages = make_adapter(
        tokenizer,
        prefix=prefix,
        template="custom template" if case == "template" else None,
    )
    if case == "effective_template":
        sys.modules["vllm.renderers.hf"].resolve_chat_template = (
            lambda *args, **kwargs: "processor-specific template"
        )
    if case == "generation_cut":
        # Gym is installed only in the dedicated optional-dependency test lane.
        from nemo_gym.token_id_capture.staging.records import GenerationCutContinuation

        adapter.admission.generation_cut = GenerationCutContinuation(
            source_capture_key="saved-rollout",
            source_model_call_id="saved-call",
            staging_keys=("saved-cut",),
            prefix_token_count=1,
            prefix_digest="c" * 64,
            effective_output_limit=64,
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


@pytest.mark.nemo_gym
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


@pytest.fixture
def online_stub(monkeypatch, parser_stub):
    """Exercise request-local dispatch without importing optional engine binaries."""
    runtime = ModuleType("vllm")
    runtime.__version__ = "0.33.0"
    protocol = ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    protocol.ChatCompletionRequest = type(
        "ChatCompletionRequest", (SimpleNamespace,), {}
    )
    hf = sys.modules["vllm.renderers.hf"]
    hf.HfRenderer = type("HfRenderer", (SimpleNamespace,), {})
    online = ModuleType("vllm.renderers.online_renderer")
    online.OnlineRenderer = type("OnlineRenderer", (SimpleNamespace,), {})
    online.OnlineRenderer.preprocess_chat = AsyncMock()
    monkeypatch.setitem(sys.modules, "vllm", runtime)
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    monkeypatch.setitem(sys.modules, online.__name__, online)
    renderer = online.OnlineRenderer(
        renderer=hf.HfRenderer(),
        model_config=SimpleNamespace(enable_prompt_embeds=False),
    )
    return SimpleNamespace(
        request=protocol.ChatCompletionRequest(),
        worker=SimpleNamespace(_capture_admission=lambda request: None),
        online_renderer=renderer,
        native=online.OnlineRenderer.preprocess_chat,
    )


def run_preprocess(state, messages=None):
    return asyncio.run(
        preprocess_captured_tito(
            model="qwen3",
            worker=state.worker,
            online_renderer=state.online_renderer,
            request=state.request,
            messages=messages or [],
            default_template=None,
            default_template_content_format="auto",
            default_template_kwargs={"enable_thinking": False},
            tool_dicts=None,
            parser=None,
            skip_mm_cache=False,
        )
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("stream", True),
        ("echo", True),
        ("continue_final_message", True),
        ("return_assistant_tokens_mask", True),
        ("add_special_tokens", True),
        ("return_prompt_text", True),
        ("return_token_offsets", True),
        ("truncate_prompt_tokens", 0),
        ("truncate_prompt_tokens", 8),
        ("kv_transfer_params", {"remote_decode": True}),
        ("n", 2),
    ],
)
def test_preprocess_leaves_unsupported_requests_to_native(online_stub, option, value):
    setattr(online_stub.request, option, value)
    assert run_preprocess(online_stub).continuation is None
    online_stub.native.assert_not_called()
    assert getattr(online_stub.request, option) == value


@pytest.mark.parametrize(
    "case", ["request", "renderer", "embeds", "uncaptured", "root"]
)
def test_preprocess_requires_qualified_capture(online_stub, case):
    if case == "request":
        online_stub.request = object()
    elif case == "renderer":
        online_stub.online_renderer.renderer = object()
    elif case == "embeds":
        online_stub.online_renderer.model_config.enable_prompt_embeds = True
    elif case == "root":
        online_stub.worker._capture_admission = lambda request: SimpleNamespace(
            mode="text", root_reason="first_call"
        )
    assert run_preprocess(online_stub).continuation is None
    online_stub.native.assert_not_called()


def test_preprocess_rejects_unqualified_runtime(online_stub):
    sys.modules["vllm"].__version__ = "0.29.0"
    with pytest.raises(ValueError, match="requires vLLM 0.33.0"):
        run_preprocess(online_stub)
    online_stub.native.assert_not_called()


@pytest.mark.nemo_gym
@pytest.mark.parametrize("outcome", ["incremental", "fallback", "storage_error"])
def test_preprocess_is_request_local(tokenizer, online_stub, outcome):
    adapter, params, messages = make_adapter(
        tokenizer, prefix=[42, tokenizer.eos_token_id]
    )
    online_stub.worker = adapter.worker
    online_stub.worker._capture_admission = lambda request: adapter.admission
    renderer = online_stub.online_renderer.renderer
    renderer.__dict__.update(adapter.renderer.__dict__)
    if outcome == "fallback":
        params.chat_template = "custom template"
        params.get_apply_chat_template_kwargs = lambda: {
            "chat_template": "custom template",
            "add_generation_prompt": True,
        }
    elif outcome == "storage_error":

        def broken(admission):
            raise ValueError("capture digest mismatch")

        online_stub.worker._resolve_admission_prefix = broken

    async def native(local, request, messages, **kwargs):
        assert local is not online_stub.online_renderer
        assert isinstance(local.renderer, CapturedTitoRenderer)
        return await local.renderer.render_chat_async(
            [messages], params, object(), prompt_extras={}, skip_mm_cache=False
        )

    online_stub.native.side_effect = native
    if outcome == "storage_error":
        with pytest.raises(ValueError, match="capture digest mismatch"):
            run_preprocess(online_stub, messages)
    else:
        result = run_preprocess(online_stub, messages)
        if outcome == "fallback":
            assert result.continuation is None
            renderer.render_cmpl_async.assert_not_called()
        else:
            native_result, captured = result.preprocessed, result.continuation
            assert native_result[1][0]["prompt_token_ids"][:2] == [
                42,
                tokenizer.eos_token_id,
            ]
            assert captured.decision.storage.mode == "token_in"
    assert online_stub.online_renderer.renderer is renderer
    online_stub.native.assert_awaited_once()


@pytest.mark.nemo_gym
@pytest.mark.parametrize("retry", [False, True])
def test_incremental_swe_tokens_round_trip_through_gym_capture(tokenizer, retry):
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
    terminal = "second"
    if retry:
        terminal = "retry"
        suffix = Qwen3IncrementalTokenizer().encode_suffix(
            tokenizer,
            [{"role": "tool", "content": "different test result"}],
            enable_thinking=False,
        )
        answer = tokenizer.encode(
            "Try another fix.<|im_end|>", add_special_tokens=False
        )
        retried = capture.begin_call(
            CaptureAdmission(
                capture_key="swe",
                model_call_id=terminal,
                mode="token_in",
                parent_call_id="first",
                prev_len=len(prefix),
                parent_chain_hash=coords.chain_hash,
                required_prefix_token_ids=prefix,
            ),
            prefix_token_ids=prefix,
        )
        capture.complete_call(
            retried,
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
            terminal_model_call_id=terminal,
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
    assert row.model_call_ids == ["first", terminal]


@pytest.mark.parametrize(
    "reason,outcome",
    [
        ("first_call", "initial"),
        ("ambiguous_candidate", "no_history"),
        ("no_prefix_match", "no_history"),
    ],
)
def test_preprocess_reports_history_outcome(online_stub, caplog, reason, outcome):
    online_stub.worker._capture_admission = lambda request: SimpleNamespace(
        mode="text", root_reason=reason
    )
    with caplog.at_level(logging.DEBUG, logger="nemo_rl.models.generation.vllm.tito"):
        result = run_preprocess(online_stub)
    assert result.preprocessed is None and result.continuation is None
    event = json.loads(caplog.records[-1].getMessage().removeprefix("Captured TITO "))
    assert event["outcome"] == outcome
    assert event["reason"] == reason
    assert event["reused_tokens"] == event["suffix_tokens"] == 0
    online_stub.native.assert_not_called()


@pytest.mark.nemo_gym
def test_continuation_rejects_changed_boundary_before_read(tokenizer):
    adapter, _, messages = make_adapter(tokenizer, prefix=[42, tokenizer.eos_token_id])
    messages[-2]["role"] = "user"
    read = Mock()
    with pytest.raises(TitoNotApplicable, match="unverified_boundary"):
        asyncio.run(
            resolve_captured_continuation(
                adapter.admission,
                messages=messages,
                conversation=messages,
                resolve_prefix=read,
            )
        )
    read.assert_not_called()


@pytest.mark.nemo_gym
def test_continuation_rejects_truncated_storage(tokenizer, parser_stub):
    adapter, params, messages = make_adapter(
        tokenizer, prefix=[42, tokenizer.eos_token_id]
    )
    adapter.worker._resolve_admission_prefix = Mock(
        return_value=[tokenizer.eos_token_id]
    )
    with pytest.raises(ValueError, match="prefix length"):
        asyncio.run(
            adapter.render_chat_async(
                [messages], params, object(), prompt_extras={}, skip_mm_cache=False
            )
        )
    adapter.renderer.render_cmpl_async.assert_not_called()


@pytest.mark.nemo_gym
def test_continuation_reads_once_and_retains_request_owned_prefix(
    tokenizer, parser_stub
):
    prefix = [42, tokenizer.eos_token_id]
    adapter, params, messages = make_adapter(tokenizer, prefix=prefix)
    read = Mock(return_value=prefix)
    adapter.worker._resolve_admission_prefix = read
    _, prompts = asyncio.run(
        adapter.render_chat_async(
            [messages], params, object(), prompt_extras={}, skip_mm_cache=False
        )
    )
    read.assert_called_once()
    assert adapter.continuation.prefix_token_ids is prefix
    assert adapter.trace.reused_tokens == 2
    assert adapter.trace.suffix_tokens == len(prompts[0]["prompt_token_ids"]) - 2
    prompts[0]["prompt_token_ids"][0] = 99
    assert prefix == [42, tokenizer.eos_token_id]


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "scenario", ["latest", "retry", "fork", "ambiguous", "rewrite"]
)
def test_gym_history_selection_reaches_incremental_renderer(
    tokenizer, parser_stub, scenario
):
    from nemo_gym.token_id_capture.external_capture import VLLMWorkerCaptureHandler
    from nemo_gym.token_id_capture.lineage import InMemoryLineageStore
    from nemo_gym.token_id_capture.replay import (
        render_options_digest,
        replay_context,
        summarize_replay,
    )
    from nemo_gym.token_id_capture.sink import (
        CaptureContext,
        reset_token_sink,
        resolve_parent,
        set_token_sink,
    )
    from nemo_gym.token_id_capture.staging.records import CallRecord, RolloutManifest

    first = [
        {"role": "user", "content": "fix bug"},
        {"role": "assistant", "content": "run tests"},
    ]
    later = first + [
        {"role": "tool", "content": "failed"},
        {"role": "assistant", "content": "inspect file"},
    ]
    messages = (later if scenario == "latest" else first) + [
        {"role": "tool", "content": "new result"}
    ]
    if scenario == "rewrite":
        messages = [{"role": "user", "content": "compacted summary"}]
    payload = {"model": "Qwen/Qwen3-0.6B", "messages": messages}
    digest = render_options_digest(payload)
    prefixes = {
        "first": [41, tokenizer.eos_token_id],
        "later": [41, tokenizer.eos_token_id, 43, tokenizer.eos_token_id],
    }

    def record(call_id, history):
        prefix = prefixes[call_id]
        return CallRecord(
            model_call_id=call_id,
            prev_len=0,
            delta_len=len(prefix),
            cum_len=len(prefix),
            weight_version=0,
            digest="a" * 64,
            extras_digest="b" * 64,
            staging_key=call_id,
            mode="text",
            chain_hash="c" * 64,
            cumulative_hash="d" * 64,
            response_id="response-" + call_id,
            replay=summarize_replay(replay_context(history, render_digest=digest)),
        )

    records = [
        record("first", first),
        record("later", first if scenario == "ambiguous" else later),
    ]
    ledger = InMemoryLineageStore()
    ledger.manifest = AsyncMock(
        return_value=RolloutManifest(capture_key="swe", records=records).model_dump()
    )
    context = CaptureContext(
        capture_key="swe",
        model_call_id="next",
        token_sink=None,
        lineage_store=ledger,
        external_staging=True,
        parent_resolution_mode="source_history",
    )

    async def run():
        token = set_token_sink(context)
        try:
            await resolve_parent(messages)
            VLLMWorkerCaptureHandler().prepare_request(payload)
        finally:
            reset_token_sink(token)
        admission = context.capture_admission
        if scenario in ("ambiguous", "rewrite"):
            assert admission.mode == "text"
            assert admission.root_reason == (
                "ambiguous_candidate" if scenario == "ambiguous" else "no_prefix_match"
            )
            return
        parent = "later" if scenario == "latest" else "first"
        assert admission.parent_call_id == parent
        adapter, params, _ = make_adapter(tokenizer, prefix=prefixes[parent])
        adapter.admission = admission
        read = Mock(side_effect=lambda selected: prefixes[selected.parent_call_id])
        adapter.worker._resolve_admission_prefix = read
        _, prompts = await adapter.render_chat_async(
            [messages], params, object(), prompt_extras={}, skip_mm_cache=False
        )
        assert (
            prompts[0]["prompt_token_ids"][: len(prefixes[parent])] == prefixes[parent]
        )
        assert adapter.continuation.decision.storage.parent_call_id == parent
        read.assert_called_once()
        if scenario == "fork":
            branch_messages = messages[:-1] + [
                {"role": "user", "content": "try another approach"}
            ]
            branch_admission = admission.model_copy(
                update={
                    "model_call_id": "branch",
                    "request_replay": replay_context(
                        branch_messages, render_digest=digest
                    ),
                }
            )
            branch, _, _ = make_adapter(tokenizer, prefix=prefixes[parent])
            branch.admission = branch_admission
            _, other = await branch.render_chat_async(
                [branch_messages],
                params,
                object(),
                prompt_extras={},
                skip_mm_cache=False,
            )
            assert (
                other[0]["prompt_token_ids"][: len(prefixes[parent])]
                == prefixes[parent]
            )
            assert other[0]["prompt_token_ids"] != prompts[0]["prompt_token_ids"]

    asyncio.run(run())
