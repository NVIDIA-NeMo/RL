# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests for the shared on-policy prefix splice used by the vLLM/TRT-LLM OpenAI servers."""

from copy import deepcopy

import pytest

from nemo_rl.models.generation.openai_server_utils import (
    escape_prefix_media_markers,
    replace_prefix_tokens,
)


def test_prefix_media_escape_preserves_source_and_observations():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "image"}}],
        },
        {
            "role": "assistant",
            "content": "<think>Count the <image> tags</think>",
            "reasoning": "An <image> and a <video>",
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "echo '<image>'"},
                    }
                }
            ],
        },
        {"role": "tool", "content": "<image> is a literal in a new observation"},
        {"role": "user", "content": [{"type": "text", "text": "<image>"}]},
    ]
    original = deepcopy(messages)
    result = escape_prefix_media_markers(messages)
    assert messages == original
    assert result[0] == messages[0]
    assert result[2:] == messages[2:]
    assert result[1]["content"] == "<think>Count the < image > tags</think>"
    assert result[1]["reasoning"] == "An < image > and a < video >"
    assert (
        result[1]["tool_calls"][0]["function"]["arguments"]["command"]
        == "echo '< image >'"
    )


@pytest.mark.parametrize("marker", ["<image>", "<video>"])
def test_prefix_media_escape_handles_content_parts_and_serialized_arguments(marker):
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": marker}],
            "tool_calls": [
                {"function": {"name": "bash", "arguments": '{"text":"' + marker + '"}'}}
            ],
        }
    ]
    result = escape_prefix_media_markers(messages)
    assert marker not in str(result)
    assert messages[0]["content"][0]["text"] == marker


def test_prefix_media_escape_returns_original_when_only_observations_have_markers():
    messages = [
        {"role": "assistant", "content": "Ready"},
        {"role": "user", "content": "Look at this <image>"},
    ]
    assert escape_prefix_media_markers(messages) is messages


def test_prefix_media_escape_preserves_exact_model_tokens_after_splice():
    messages = [{"role": "assistant", "content": "A literal <image>"}]
    escaped = escape_prefix_media_markers(messages)
    # A toy renderer gives the harmless replacement ordinary text tokens. The
    # final prompt must still contain the sampled literal's exact token ID 18.
    rendered_assistant = [70, 50, 51, 52, 2]
    assert escaped[0]["content"] == "A literal < image >"
    model_prefix = [70, 18, 2]
    suffix = [80, 18, 18, 81]
    result = replace_prefix_tokens(
        tokenizer=None,
        eos_token_id=2,
        model_prefix_token_ids=model_prefix,
        template_prefix_token_ids=rendered_assistant,
        template_token_ids=rendered_assistant + suffix,
    )
    assert result == model_prefix + suffix


def test_replace_prefix_tokens_empty_model_prefix_returns_template():
    """Turn 1 has no prior model output; the full template is returned unchanged."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = []
    template_prefix_token_ids = [9, 2]
    template_token_ids = [9, 2, 33, 44]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == template_token_ids


def test_replace_prefix_tokens_missing_eos_in_template_prefix_raises():
    """A template prefix with no EOS has no valid splice boundary and must raise."""

    class _T:
        eos_token_id = 2

        def decode(self, *args, **kwargs):
            pass

    tokenizer = _T()
    model_prefix_token_ids = [7, 2]
    template_prefix_token_ids = [9, 9, 9]
    template_token_ids = [9, 9, 9, 2, 10]
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )


def test_replace_prefix_tokens_tokenizer_without_eos_raises():
    """A tokenizer that has no EOS token cannot locate the splice boundary."""

    class _T:
        eos_token_id = None

    tokenizer = _T()
    with pytest.raises(AssertionError):
        replace_prefix_tokens(
            tokenizer=tokenizer,
            model_prefix_token_ids=[1],
            template_prefix_token_ids=[1, 2],
            template_token_ids=[1, 2],
        )


def test_replace_prefix_tokens_without_tokenizer_uses_explicit_eos():
    """Callers that only hold token ids (the Megatron prompt preparer) pass eos_token_id."""
    result = replace_prefix_tokens(
        tokenizer=None,
        model_prefix_token_ids=[100, 2],
        template_prefix_token_ids=[9, 2],
        template_token_ids=[9, 2, 77, 88],
        eos_token_id=2,
    )
    assert result == [100, 2, 77, 88]


def test_replace_prefix_tokens_without_tokenizer_reports_missing_eos_without_decoding():
    """The failure message must not require a tokenizer to decode."""
    with pytest.raises(AssertionError, match="EOS token #1 not found"):
        replace_prefix_tokens(
            tokenizer=None,
            model_prefix_token_ids=[100, 2],
            template_prefix_token_ids=[9, 2],
            template_token_ids=[9, 77],
            eos_token_id=2,
        )


def test_replace_prefix_tokens_uses_last_eos_in_template_prefix():
    """When the prefix contains multiple EOS tokens, the splice cuts at the last one."""

    class _T:
        eos_token_id = 2

    tokenizer = _T()
    model_prefix_token_ids = [100, 2]
    template_prefix_token_ids = [9, 2, 9, 2]
    template_token_ids = [9, 2, 9, 2, 77, 88]
    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )
    assert result == [100, 2, 77, 88]


def test_replace_prefix_tokens_qwen3_think_shift_picks_assistant_eos_not_user_eos():
    """Non-strict-prefix: Qwen3 strips <think> from history when the last message is a
    user turn, so the template's prefix region is shorter and a later user-turn EOS
    lands within the first len(template_prefix) positions. The count-based algorithm
    must cut at the assistant EOS and preserve the intervening user turn.
    """

    class _T:
        eos_token_id = 2

        def decode(self, ids, **kwargs):
            return " ".join(str(i) for i in ids)

    tokenizer = _T()
    model_prefix_token_ids = [11, 12, 99, 99, 99, 55, 2]
    template_prefix_token_ids = [11, 12, 88, 88, 88, 56, 2]
    template_token_ids = [11, 12, 56, 2, 70, 71, 2, 40, 41]

    result = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix_token_ids,
        template_prefix_token_ids=template_prefix_token_ids,
        template_token_ids=template_token_ids,
    )

    assert result == [11, 12, 99, 99, 99, 55, 2, 70, 71, 2, 40, 41]
    assert 70 in result and 71 in result
