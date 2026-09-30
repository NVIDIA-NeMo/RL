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

import pytest

from nemo_rl.models.generation.openai_server_utils import (
    PrefixSplice,
    replace_prefix_tokens,
    splice_prefix_tokens,
)


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


@pytest.mark.parametrize(
    ("model_prefix", "template_prefix", "template", "eos_token_id", "expected"),
    [
        # splice_prefix_tokens accepts eos_token_id itself, not only via
        # replace_prefix_tokens.
        pytest.param(
            [100, 2],
            [9, 2],
            [9, 2, 77, 88],
            2,
            PrefixSplice(
                token_ids=[100, 2, 77, 88], model_cut_end=1, template_cut_start=1
            ),
            id="single-eos-id",
        ),
        # Megatron-LM ships the model's full EOS set; any member marks a turn
        # boundary. Here the model stopped on the template's own terminator ...
        pytest.param(
            [100, 2],
            [9, 2],
            [9, 2, 77, 88],
            [2, 11],
            PrefixSplice(
                token_ids=[100, 2, 77, 88], model_cut_end=1, template_cut_start=1
            ),
            id="eos-set",
        ),
        # ... and here on a different declared EOS: its exact id survives the
        # splice, so the prompt still starts with the model's tokens.
        pytest.param(
            [100, 11],
            [9, 2],
            [9, 2, 77, 88],
            [2, 11],
            PrefixSplice(
                token_ids=[100, 11, 77, 88], model_cut_end=1, template_cut_start=1
            ),
            id="eos-set-other-member",
        ),
        # A template prefix whose turns end on different EOS ids counts all of them.
        pytest.param(
            [5, 11, 6, 2],
            [5, 11, 7, 2],
            [5, 11, 7, 2, 77],
            [2, 11],
            PrefixSplice(
                token_ids=[5, 11, 6, 2, 77], model_cut_end=3, template_cut_start=3
            ),
            id="eos-set-counts-every-member",
        ),
        # A prefix cut by max_tokens has no EOS to keep; the template's boundary is used.
        pytest.param(
            [100, 101],
            [9, 2],
            [9, 2, 77],
            [2, 11],
            PrefixSplice(
                token_ids=[100, 101, 2, 77], model_cut_end=2, template_cut_start=1
            ),
            id="no-trailing-eos",
        ),
    ],
)
def test_splice_prefix_tokens_without_tokenizer_honors_explicit_eos_ids(
    model_prefix, template_prefix, template, eos_token_id, expected
):
    result = splice_prefix_tokens(
        tokenizer=None,
        model_prefix_token_ids=model_prefix,
        template_prefix_token_ids=template_prefix,
        template_token_ids=template,
        eos_token_id=eos_token_id,
    )
    assert result == expected
    assert result.token_ids[: len(model_prefix)] == model_prefix


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
