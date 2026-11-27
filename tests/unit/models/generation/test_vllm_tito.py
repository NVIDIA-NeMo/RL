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
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from transformers import AutoTokenizer

from nemo_rl.models.generation import tito as shared_tito
from nemo_rl.models.generation.openai_server_utils import splice_prefix_tokens
from nemo_rl.models.generation.tito import (
    QWEN3_TEMPLATE_SHA256,
    Qwen3IncrementalTokenizer,
    TitoNotApplicable,
    qualify_incremental_tokenizer,
    qwen3_suffix_text,
)
from nemo_rl.models.generation.vllm import tito
from nemo_rl.models.generation.vllm.tito import (
    TitoTrace,
    plan_incremental_continuation,
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


def make_request(**options):
    return SimpleNamespace(**{"add_generation_prompt": True, **options})


def make_candidate(tokenizer, *, prefix):
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
    worker = SimpleNamespace(
        _capture_admission=lambda request: admission,
        _resolve_admission_prefix=Mock(side_effect=lambda selected: list(prefix)),
    )
    return worker, admission, messages


def run_plan(tokenizer, worker, messages, request=None, template_kwargs=None):
    return asyncio.run(
        plan_incremental_continuation(
            incremental_tokenizer=Qwen3IncrementalTokenizer(),
            tokenizer=tokenizer,
            worker=worker,
            request=request or make_request(),
            messages=messages,
            default_template_kwargs=(
                {"enable_thinking": False}
                if template_kwargs is None
                else template_kwargs
            ),
        )
    )


@pytest.mark.nemo_gym
def test_plan_encodes_only_suffix(tokenizer, monkeypatch):
    prefix = [42, tokenizer.eos_token_id]
    worker, _, messages = make_candidate(tokenizer, prefix=prefix)
    encoded = []
    encode = tokenizer.encode

    def track(text, **kwargs):
        encoded.append(text)
        return encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", track)
    continuation = run_plan(tokenizer, worker, messages)
    assert encoded == [qwen3_suffix_text(messages[-1:], enable_thinking=False)]
    assert continuation.parent_token_ids == prefix
    assert continuation.prompt_token_ids == prefix + continuation.suffix_token_ids
    assert continuation.decision.storage.mode == "token_in"
    assert continuation.decision.storage.parent_call_id == "first"
    worker._resolve_admission_prefix.assert_called_once()


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "request_kwargs,default_kwargs,thinking",
    [
        ({}, {}, True),
        ({}, {"enable_thinking": False}, False),
        ({"enable_thinking": True}, {"enable_thinking": False}, True),
        ({"enable_thinking": False}, {"enable_thinking": True}, False),
    ],
)
def test_plan_uses_the_request_thinking_setting(
    tokenizer, request_kwargs, default_kwargs, thinking
):
    prefix = [42, tokenizer.eos_token_id]
    worker, _, messages = make_candidate(tokenizer, prefix=prefix)
    continuation = run_plan(
        tokenizer,
        worker,
        messages,
        request=make_request(chat_template_kwargs=request_kwargs),
        template_kwargs=default_kwargs,
    )
    assert continuation.suffix_token_ids == tokenizer.encode(
        qwen3_suffix_text(messages[-1:], enable_thinking=thinking),
        add_special_tokens=False,
    )


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "case,reads",
    [
        ("unfinished", 1),
        ("media", 0),
        ("generation_cut", 0),
        ("changed_boundary", 0),
        ("empty_tool_equivalence", 0),
    ],
)
def test_plan_leaves_ineligible_continuations_to_the_bridge(tokenizer, case, reads):
    prefix = [42, tokenizer.eos_token_id] if case != "unfinished" else [42]
    worker, admission, messages = make_candidate(tokenizer, prefix=prefix)
    if case == "generation_cut":
        # Gym is installed only in the dedicated optional-dependency test lane.
        from nemo_gym.token_id_capture.staging.records import GenerationCutContinuation

        admission.generation_cut = GenerationCutContinuation(
            source_capture_key="saved-rollout",
            source_model_call_id="saved-call",
            staging_keys=("saved-cut",),
            prefix_token_count=1,
            prefix_digest="c" * 64,
            effective_output_limit=64,
        )
    elif case == "media":
        messages[-1]["content"] = [{"type": "image_url", "image_url": {"url": "x"}}]
    elif case == "changed_boundary":
        messages[-2]["role"] = "user"
    elif case == "empty_tool_equivalence":
        admission.requires_empty_tool_content_equivalence = True
    assert run_plan(tokenizer, worker, messages) is None
    assert worker._resolve_admission_prefix.call_count == reads


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
        ("chat_template", "custom template"),
        ("reasoning_effort", "low"),
        ("add_generation_prompt", False),
        ("n", 2),
    ],
)
def test_plan_leaves_unsupported_requests_to_the_bridge(tokenizer, option, value):
    worker = SimpleNamespace(_capture_admission=Mock())
    request = make_request(**{option: value})
    assert run_plan(tokenizer, worker, [], request=request) is None
    worker._capture_admission.assert_not_called()
    assert getattr(request, option) == value


@pytest.mark.nemo_gym
def test_capture_storage_failure_is_not_a_fallback(tokenizer):
    worker, _, messages = make_candidate(tokenizer, prefix=[42, tokenizer.eos_token_id])
    worker._resolve_admission_prefix = Mock(
        side_effect=ValueError("capture digest mismatch")
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        run_plan(tokenizer, worker, messages)


@pytest.mark.nemo_gym
def test_plan_rejects_truncated_storage(tokenizer):
    worker, _, messages = make_candidate(tokenizer, prefix=[42, tokenizer.eos_token_id])
    worker._resolve_admission_prefix = Mock(return_value=[tokenizer.eos_token_id])
    with pytest.raises(ValueError, match="resolved to 1 tokens"):
        run_plan(tokenizer, worker, messages)


def test_qwen3_template_contract(tokenizer):
    # This guards the manual suffix adapter against an upstream template change.
    assert (
        hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        == QWEN3_TEMPLATE_SHA256
    )


def test_qualification_accepts_official_template(tokenizer):
    qualify_incremental_tokenizer(
        "qwen3", tokenizer=tokenizer, chat_template=tokenizer.chat_template
    )


class _ExtraSuffixToken(Qwen3IncrementalTokenizer):
    def qualify_template(self, template):
        pass

    def encode_suffix(self, tokenizer, messages, *, enable_thinking):
        suffix = super().encode_suffix(
            tokenizer, messages, enable_thinking=enable_thinking
        )
        return suffix + [tokenizer.eos_token_id]


class _MissingSeparator(_ExtraSuffixToken):
    def encode_suffix(self, tokenizer, messages, *, enable_thinking):
        return Qwen3IncrementalTokenizer.encode_suffix(
            self, tokenizer, messages, enable_thinking=enable_thinking
        )[1:]


@pytest.mark.parametrize(
    "case,match",
    [
        ("template", "does not support this server's chat template"),
        ("missing_template", "does not support this server's chat template"),
        ("suffix", "differs from the chat template rendering"),
        ("boundary", "does not end the assistant turn"),
    ],
)
def test_qualification_rejects_unqualified_server(tokenizer, monkeypatch, case, match):
    template = tokenizer.chat_template
    if case == "template":
        template += " "
    elif case == "missing_template":
        template = None
    elif case == "suffix":
        monkeypatch.setitem(
            shared_tito.INCREMENTAL_TOKENIZERS, "qwen3", _ExtraSuffixToken()
        )
    elif case == "boundary":
        monkeypatch.setitem(
            shared_tito.INCREMENTAL_TOKENIZERS, "qwen3", _MissingSeparator()
        )
    with pytest.raises(ValueError, match=match):
        qualify_incremental_tokenizer(
            "qwen3", tokenizer=tokenizer, chat_template=template
        )


def test_outcomes_are_summarized_at_the_interval(monkeypatch, caplog):
    monkeypatch.setattr(tito, "_OUTCOME_SUMMARY_INTERVAL", 2)
    outcomes = tito.TitoOutcomes()
    with caplog.at_level(logging.INFO, logger="nemo_rl.models.generation.vllm.tito"):
        outcomes.record(TitoTrace(outcome="incremental", reason="matched"))
        assert not caplog.records
        outcomes.record(TitoTrace(outcome="bridge", reason="stream"))
    summary = caplog.records[-1].getMessage()
    assert summary.startswith("Captured TITO outcomes over 2 requests: ")
    assert json.loads(summary.split(": ", 1)[1]) == {
        "bridge/stream": 1,
        "incremental/matched": 1,
    }


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
    "admission,reason",
    [
        (None, "uncaptured"),
        (SimpleNamespace(mode="text", root_reason="first_call"), "first_call"),
        (
            SimpleNamespace(mode="text", root_reason="no_prefix_match"),
            "no_prefix_match",
        ),
        (SimpleNamespace(mode="token_in", root_reason=None), "non_candidate"),
    ],
)
def test_plan_reports_why_a_request_is_bridged(tokenizer, caplog, admission, reason):
    worker = SimpleNamespace(_capture_admission=lambda request: admission)
    with caplog.at_level(logging.DEBUG, logger="nemo_rl.models.generation.vllm.tito"):
        assert run_plan(tokenizer, worker, []) is None
    event = json.loads(caplog.records[-1].getMessage().removeprefix("Captured TITO "))
    assert event["outcome"] == "bridge"
    assert event["reason"] == reason
    assert event["reused_tokens"] == event["suffix_tokens"] == 0


@pytest.mark.nemo_gym
def test_plan_reports_reused_and_appended_tokens(tokenizer, caplog):
    prefix = [42, tokenizer.eos_token_id]
    worker, _, messages = make_candidate(tokenizer, prefix=prefix)
    with caplog.at_level(logging.DEBUG, logger="nemo_rl.models.generation.vllm.tito"):
        continuation = run_plan(tokenizer, worker, messages)
    event = json.loads(caplog.records[-1].getMessage().removeprefix("Captured TITO "))
    assert event["outcome"] == "incremental"
    assert event["reused_tokens"] == 2
    assert event["suffix_tokens"] == len(continuation.suffix_token_ids)
    continuation.prompt_token_ids[0] = 99
    assert continuation.parent_token_ids == prefix


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "scenario", ["latest", "retry", "fork", "ambiguous", "rewrite"]
)
def test_gym_history_selection_reaches_incremental_tokenizer(tokenizer, scenario):
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
        read = Mock(side_effect=lambda selected: prefixes[selected.parent_call_id])

        async def plan(selected, request_messages):
            return await plan_incremental_continuation(
                incremental_tokenizer=Qwen3IncrementalTokenizer(),
                tokenizer=tokenizer,
                worker=SimpleNamespace(
                    _capture_admission=lambda request: selected,
                    _resolve_admission_prefix=read,
                ),
                request=make_request(),
                messages=request_messages,
                default_template_kwargs={"enable_thinking": False},
            )

        continuation = await plan(admission, messages)
        assert continuation.parent_token_ids == prefixes[parent]
        assert continuation.decision.storage.parent_call_id == parent
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
            other = await plan(branch_admission, branch_messages)
            assert other.parent_token_ids == prefixes[parent]
            assert other.prompt_token_ids != continuation.prompt_token_ids

    asyncio.run(run())
