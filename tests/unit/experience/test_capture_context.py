# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source rewrites and unsupported replay must be distinguished before splice."""

import logging
from copy import deepcopy

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
        rollout_id="attempt",
        model_call_id="next",
        mode="candidate",
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


@pytest.mark.parametrize("index, replacement", [(0, "summary"), (1, "answer")])
def test_edited_history_or_removed_exposed_reasoning_renders_without_splice(
    index, replacement
):
    items = [dict(item) for item in HISTORY] + [OBSERVATION]
    items[index]["content"] = replacement
    decision = decide_capture_input(candidate(items), messages=items)
    assert decision.serving is None
    assert decision.storage.mode == "text"


def test_missing_render_evidence_is_not_a_rewrite():
    admission = candidate(candidate_replay=summarize_replay(replay_context(HISTORY)))
    with pytest.raises(ValueError, match="Missing predecessor"):
        decide_capture_input(admission, messages=HISTORY + [OBSERVATION])


def test_render_option_change_is_a_rewrite():
    admission = candidate(
        request_replay=replay_context(HISTORY + [OBSERVATION], render_digest="c" * 64)
    )
    assert (
        decide_capture_input(admission, messages=HISTORY + [OBSERVATION]).serving
        is None
    )


@pytest.mark.parametrize("suffix", [[], [{"role": "assistant", "content": "prefill"}]])
def test_no_supported_observation_or_appended_assistant_roots_a_new_segment(
    suffix, caplog
):
    with caplog.at_level(
        logging.WARNING, logger="nemo_rl.models.generation.capture_context"
    ):
        decision = decide_capture_input(
            candidate(HISTORY + suffix), messages=HISTORY + suffix
        )
    assert decision.storage.mode == "text"
    assert decision.storage.parent_call_id is None
    assert decision.serving is None
    assert decision.verify_retained_media
    assert "rooting a new segment" in caplog.text


def test_an_unproven_response_boundary_roots_a_new_segment():
    decision = decide_capture_input(
        candidate(), messages=HISTORY + [OBSERVATION, HISTORY[-1]]
    )
    assert decision.storage.mode == "text"
    assert decision.serving is None


def test_ordinary_admission_is_unchanged():
    admission = CaptureAdmission(rollout_id="a", model_call_id="c", mode="text")
    decision = decide_capture_input(admission, messages=[])
    assert decision.storage is admission
    assert decision.serving is admission
    assert not decision.verify_retained_media


@pytest.mark.parametrize(
    "items", [[], HISTORY[:1], list(reversed(HISTORY)) + [OBSERVATION]]
)
def test_shortened_or_reordered_history_starts_a_root(items):
    decision = decide_capture_input(candidate(items), messages=items)
    assert decision.storage.mode == "text"
    assert decision.serving is None


@pytest.mark.parametrize("edited", [False, True])
def test_long_history_checks_early_media_and_last_response(edited):
    history = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://example.test/a.png"}
            ],
        },
        HISTORY[1],
    ] + [item for _ in range(127) for item in HISTORY]
    current = [dict(item) for item in history] + [OBSERVATION]
    if edited:
        current[0] = {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://example.test/b.png"}
            ],
        }
    admission = candidate(
        current,
        candidate_replay=summarize_replay(
            replay_context(history, render_digest="b" * 64)
        ),
    )
    decision = decide_capture_input(admission, messages=current)
    assert decision.storage.mode == ("text" if edited else "token_in")


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


@pytest.mark.parametrize(
    "edit",
    ["text", "whitespace", "tool_id", "arguments", "reasoning", "render", "order"],
)
def test_empty_content_equivalence_does_not_hide_real_edits(edit):
    items = tool_history("") + [OBSERVATION]
    original = deepcopy(items)
    if edit in ("text", "whitespace"):
        items[1]["content"] = "changed" if edit == "text" else " "
    elif edit == "tool_id":
        items[1]["tool_calls"][0]["id"] = "different"
    elif edit == "arguments":
        items[1]["tool_calls"][0]["function"]["arguments"] = '{"x":2}'
    elif edit == "reasoning":
        items[1]["reasoning"] = "new reasoning"
    elif edit == "order":
        items[:2] = reversed(items[:2])
    admission = candidate(
        items,
        candidate_replay=summarize_replay(
            replay_context(tool_history(None), render_digest="b" * 64)
        ),
    )
    if edit == "render":
        admission.request_replay.render_digest = "c" * 64
    assert (
        decide_capture_input(
            admission, messages=items, allow_empty_tool_content=True
        ).storage.mode
        == "text"
    )
    assert original[1]["content"] == ""


def test_old_strict_evidence_remains_usable_but_cannot_authorize_equivalence():
    summary = summarize_replay(
        replay_context(tool_history(None), render_digest="b" * 64)
    )
    summary.empty_tool_content_digest = None
    for content, expected in [(None, "token_in"), ("", "text")]:
        items = tool_history(content) + [OBSERVATION]
        decision = decide_capture_input(
            candidate(items, candidate_replay=summary),
            messages=items,
            allow_empty_tool_content=True,
        )
        assert decision.storage.mode == expected


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_empty_equivalence_does_not_normalize_other_messages(role):
    history = [{"role": role, "content": None}, HISTORY[1]]
    items = [{"role": role, "content": ""}, HISTORY[1], OBSERVATION]
    admission = candidate(
        items,
        candidate_replay=summarize_replay(
            replay_context(history, render_digest="b" * 64)
        ),
    )
    assert (
        decide_capture_input(
            admission, messages=items, allow_empty_tool_content=True
        ).storage.mode
        == "text"
    )
