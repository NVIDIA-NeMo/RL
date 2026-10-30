# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source rewrites and unsupported replay must be distinguished before splice."""

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.token_id_capture.replay import replay_context, summarize_replay
from nemo_gym.token_id_capture.staging.records import CaptureAdmission
from pydantic import ValidationError

from nemo_rl.models.generation.capture_context import decide_capture_input

pytestmark = pytest.mark.nemo_gym

HISTORY = [
    {"role": "user", "content": "original question"},
    {"role": "assistant", "content": "<think>private work</think>answer"},
]
OBSERVATION = {"role": "user", "content": "next observation"}


def candidate(items=None, **overrides):
    values = dict(
        capture_key="attempt",
        model_call_id="next",
        mode="candidate",
        requires_empty_tool_content_equivalence=False,
        parent_call_id="previous",
        prev_len=3,
        required_prefix_token_ids=[10, 11, 2],
        parent_chain_hash="a" * 64,
        candidate_replay=summarize_replay(
            replay_context(HISTORY, render_digest="b" * 64)
        ),
        request_replay=replay_context(
            HISTORY + [OBSERVATION] if items is None else items, render_digest="b" * 64
        ),
    )
    values.update(overrides)
    return CaptureAdmission(**values)


def test_compatible_request_preserves_prefix_and_links_delta_capture():
    decision = decide_capture_input(candidate(), messages=HISTORY + [OBSERVATION])
    assert decision.storage is decision.serving
    assert decision.storage.mode == "token_in"
    assert decision.storage.prev_len == 3
    assert decision.serving.parent_call_id == "previous"
    assert decision.serving.required_prefix_token_ids == [10, 11, 2]
    assert decision.verify_retained_media


def test_an_unproven_response_boundary_roots_a_new_segment():
    decision = decide_capture_input(
        candidate(), messages=HISTORY + [OBSERVATION, HISTORY[-1]]
    )
    assert decision.storage.mode == "text"
    assert decision.serving is None


def test_ordinary_admission_is_unchanged():
    admission = CaptureAdmission(capture_key="a", model_call_id="c", mode="text")
    decision = decide_capture_input(admission, messages=[])
    assert decision.storage is admission
    assert decision.serving is admission
    assert not decision.verify_retained_media


def test_previous_array_format_and_unknown_normalization_fail_closed():
    with pytest.raises(ValidationError):
        candidate(candidate_replay=replay_context(HISTORY).model_dump())
    summary = summarize_replay(replay_context(HISTORY)).model_dump()
    summary["normalization_version"] = 2
    with pytest.raises(ValidationError):
        candidate(candidate_replay=summary)


def tool_history(content=None):
    return [
        HISTORY[0],
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "check", "arguments": '{"x":1}'},
                }
            ],
        },
    ]


@pytest.mark.parametrize("before,after", [(None, ""), ("", None)])
@pytest.mark.parametrize("qualified", [False, True])
def test_empty_tool_content_needs_renderer_proof(before, after, qualified):
    items = tool_history(after) + [OBSERVATION]
    admission = candidate(
        items,
        requires_empty_tool_content_equivalence=True,
        candidate_replay=summarize_replay(
            replay_context(tool_history(before), render_digest="b" * 64)
        ),
    )
    decision = decide_capture_input(
        admission, messages=items, allow_empty_tool_content=qualified
    )
    assert decision.storage.mode == ("token_in" if qualified else "text")
    if qualified:
        assert decision.serving.required_prefix_token_ids == [10, 11, 2]
        assert decision.storage.parent_call_id == "previous"


@pytest.mark.parametrize("proof", [None, "false", 0])
def test_missing_or_nonboolean_source_proof_fails_closed(proof):
    with pytest.raises(ValidationError):
        candidate(requires_empty_tool_content_equivalence=proof)


def test_candidate_requires_parent_and_root_cannot_carry_candidate_proof():
    with pytest.raises(ValidationError):
        candidate(
            parent_call_id=None,
            prev_len=0,
            required_prefix_token_ids=[],
            parent_chain_hash=None,
        )
    with pytest.raises(ValidationError):
        CaptureAdmission(
            capture_key="a",
            model_call_id="b",
            mode="text",
            requires_empty_tool_content_equivalence=False,
        )


def test_managed_root_preserves_media_verification_without_serving_prefix():
    admission = CaptureAdmission(
        capture_key="a",
        model_call_id="b",
        mode="text",
        request_replay=replay_context(HISTORY, render_digest="b" * 64),
    )
    decision = decide_capture_input(admission, messages=HISTORY)
    assert decision.serving is None
    assert decision.storage.mode == "text"
    assert decision.storage.parent_call_id is None
    assert decision.verify_retained_media
    with pytest.raises(ValueError, match="Missing current"):
        decide_capture_input(
            admission.model_copy(update={"request_replay": replay_context(HISTORY)}),
            messages=HISTORY,
        )


def test_exact_source_match_does_not_require_renderer_equivalence():
    items = tool_history(None) + [OBSERVATION]
    admission = candidate(
        items,
        candidate_replay=summarize_replay(
            replay_context(tool_history(None), render_digest="b" * 64)
        ),
    )
    assert (
        decide_capture_input(
            admission, messages=items, allow_empty_tool_content=False
        ).storage.mode
        == "token_in"
    )


@pytest.mark.parametrize("root_cause", ["source", "options", "boundary"])
def test_paired_root_append_root_append_keeps_exact_tokens(
    monkeypatch, tmp_path, root_cause
):
    from responses_api_models.vllm_model.tests import test_framework_context as gym

    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
    from tests.unit.experience.test_logical_owner_finalization import (
        PublicationDataPlane,
    )

    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    fetches = []
    decisions = []

    def fetch(keys):
        fetches.append(list(keys))
        return source.fetch_prefix_chains(keys).expanded

    def decide(admission, *, messages):
        if root_cause == "boundary" and len(decisions) == 2:
            messages = messages + [
                {"role": "assistant", "content": "conversion moved the cut"}
            ]
        result = decide_capture_input(admission, messages=messages)
        decisions.append(result)
        return result

    harness = gym.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=fetch,
        root_prompt=[10, 11],
        decide_input=decide,
    )
    items = [{"role": "user", "content": "task"}]
    for turn in range(4):
        if turn == 2:
            if root_cause == "source":
                items = [{"role": "user", "content": "summary"}]
            elif root_cause == "options":
                harness.model.config.model = "changed-model"
        response = gym.assert_clean(harness.post("attempt", items))
        items += response["output"] + [{"role": "user", "content": "next"}]
    manifest = harness.manifest("attempt")
    records = manifest.records
    assert not manifest.failures and not manifest.pending_call_ids
    assert [r.parent_call_id for r in records] == [
        None,
        records[0].model_call_id,
        None,
        records[2].model_call_id,
    ]
    assert [r.prev_len for r in records] == [0, 3, 0, 3]
    assert [r.cum_len for r in records] == [3, 6, 3, 6]
    assert all(r.replay and r.replay.render_digest for r in records)
    # Direct roots and worker downgrades never fetch a discarded parent prefix.
    assert fetches == [[records[0].staging_key], [records[2].staging_key]]
    assert all(d.verify_retained_media for d in decisions)
    assert source.fetch_prefix_chains(
        [r.staging_key for r in records[:2]]
    ).expanded == [
        10,
        11,
        1001,
        20,
        21,
        1002,
    ]
    assert source.fetch_prefix_chains(
        [r.staging_key for r in records[2:]]
    ).expanded == [
        10,
        11,
        1003,
        40,
        41,
        1004,
    ]
    assert harness.worker_calls[2][0].mode == (
        "candidate" if root_cause == "boundary" else "text"
    )


@pytest.mark.asyncio
async def test_worker_managed_root_clears_untrusted_prefix_and_keeps_media_check(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )
    from tests.unit.models.generation.test_vllm_chat_template_wiring import (
        _build_server,
        _OnlineRenderer,
    )

    renderer = _build_server(monkeypatch, {})[0][0]
    renderer.renderer = SimpleNamespace()
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "token_capture", object(), raising=False
    )
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "_capture_media", True, raising=False
    )
    capture_media = MagicMock(return_value=None)
    begin = MagicMock()
    fetch = MagicMock(side_effect=AssertionError("root must not fetch a prefix"))
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "_capture_request_media", capture_media
    )
    monkeypatch.setattr(VllmAsyncGenerationWorkerImpl, "_begin_request_capture", begin)
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "_resolve_admission_prefix", fetch
    )
    prompt = {"prompt_token_ids": [10, 11]}

    async def render(self, **kwargs):
        assert kwargs["skip_mm_cache"]
        return HISTORY, [prompt]

    monkeypatch.setattr(_OnlineRenderer, "preprocess_chat", render, raising=False)
    admission = CaptureAdmission(
        capture_key="a",
        model_call_id="b",
        mode="text",
        request_replay=replay_context(HISTORY, render_digest="b" * 64),
    )
    request = SimpleNamespace(
        ng_capture=admission.model_dump(), required_prefix_token_ids=[999]
    )
    result = await renderer.preprocess_chat(request, HISTORY, None, "string", {})
    assert request.required_prefix_token_ids is None
    assert result[1][0] is prompt
    fetch.assert_not_called()
    capture_media.assert_called_once()
    assert capture_media.call_args.kwargs["verify_retained_media"] is True
    assert capture_media.call_args.kwargs["admission"].mode == "text"
    begin.assert_called_once()
    assert begin.call_args.args[1] == [10, 11]
