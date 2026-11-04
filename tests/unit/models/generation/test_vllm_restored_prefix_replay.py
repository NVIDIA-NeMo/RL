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
"""A restored generation prefix reads exactly as vLLM first produced it.

Every expected response here is vLLM's own detokenization of the
uninterrupted call, so the tests track vLLM rather than hand-written strings.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.vllm

detokenizer = pytest.importorskip("vllm.v1.engine.detokenizer")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    _detokenize_restored_prefix,
    _RestoredPrefixTerminal,
)

_SPECIAL = ["<unk>", "<s>", "</s>", "<|stop|>"]
# Word pieces, written with a plain leading space where the token carries one.
_PIECES = [" Q", ":", " Hello", " world", "!", " answer", " 4", "</", "answer", ">"]
_PROMPT = ["<s>", " Q", ":"]
_RESPONSE = [" Hello", " world", "!", " answer", ":", " 4"]
_STOP_STRING_RESPONSE = [" answer", ":", " 4", "</", "answer", ">"]


def _tokenizer(style: str) -> PreTrainedTokenizerFast:
    if style == "metaspace":
        # SentencePiece (Llama 2, Mistral): decoding drops the leading space of
        # the first token, so a prefix decoded on its own loses it.
        space = "▁"
        pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
        decoder = decoders.Metaspace(prepend_scheme="always")
    else:
        # Byte-level BPE (Qwen, Llama 3).
        space = "Ġ"
        pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        decoder = decoders.ByteLevel()
    vocab = {
        token: token_id
        for token_id, token in enumerate(
            _SPECIAL + [piece.replace(" ", space) for piece in _PIECES]
        )
    }
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizer
    raw.decoder = decoder
    raw.add_special_tokens(_SPECIAL)
    return PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )


def _ids(tokenizer: PreTrainedTokenizerFast, pieces: list[str]) -> list[int]:
    return [
        tokenizer.convert_tokens_to_ids(
            piece if piece in _SPECIAL else piece.replace(" ", _space(tokenizer))
        )
        for piece in pieces
    ]


def _space(tokenizer: PreTrainedTokenizerFast) -> str:
    return "▁" if "▁Q" in tokenizer.get_vocab() else "Ġ"


def _vllm_response(
    tokenizer: PreTrainedTokenizerFast,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    *,
    stop_token_terminated: bool = False,
    stop: list[str] | None = None,
    min_tokens: int = 0,
    include_stop_str_in_output: bool = False,
) -> tuple[str, int]:
    """Detokenize a call one step at a time, as vLLM's output processor does.

    Returns the response text and how many tokens the call generated before a
    stop string ended it.
    """
    state = detokenizer.IncrementalDetokenizer.from_new_request(
        tokenizer,
        SimpleNamespace(
            request_id="uninterrupted",
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=None,
            sampling_params=SimpleNamespace(
                stop=stop,
                min_tokens=min_tokens,
                include_stop_str_in_output=include_stop_str_in_output,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            ),
        ),
    )
    generated = 0
    for token_id in output_token_ids:
        generated += 1
        last = generated == len(output_token_ids)
        if state.update([token_id], stop_token_terminated and last) is not None:
            break
    return state.get_next_output_text(finished=True, delta=False), generated


def _replay(
    tokenizer: PreTrainedTokenizerFast,
    prompt_token_ids: list[int],
    prefix_token_ids: list[int],
    *,
    terminal: _RestoredPrefixTerminal | None = None,
    stop: list[str] | None = None,
    min_tokens: int = 0,
    include_stop_str_in_output: bool = False,
) -> str:
    return _detokenize_restored_prefix(
        tokenizer,
        original_prompt_token_ids=prompt_token_ids,
        prefix_token_ids=prefix_token_ids,
        terminal=terminal,
        stop=stop,
        min_tokens=min_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )


def _terminal(
    prompt_token_ids: list[int],
    prefix_token_ids: list[int],
    finish_reason: str,
    stop_reason: str | int | None = None,
) -> _RestoredPrefixTerminal:
    return _RestoredPrefixTerminal(
        prompt_token_ids=tuple(prompt_token_ids + prefix_token_ids),
        prefix_token_count=len(prefix_token_ids),
        reason=f"observed_{finish_reason}",
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


@pytest.mark.parametrize("style", ["metaspace", "byte_level"])
def test_resumed_response_matches_the_uninterrupted_one_at_every_cut(style):
    tokenizer = _tokenizer(style)
    prompt = _ids(tokenizer, _PROMPT)
    response = _ids(tokenizer, _RESPONSE + ["</s>"])
    expected, _ = _vllm_response(
        tokenizer, prompt, response, stop_token_terminated=True
    )

    for cut in range(1, len(response)):
        prefix, tail = response[:cut], response[cut:]
        # The resumed engine call sees the prefix as part of its prompt.
        tail_text, _ = _vllm_response(
            tokenizer, prompt + prefix, tail, stop_token_terminated=True
        )
        prefix_text = _replay(tokenizer, prompt, prefix)

        assert prefix_text + tail_text == expected, f"cut after {cut} tokens"


def test_standalone_sentencepiece_decode_drops_the_space_the_replay_keeps():
    """Guards the test above: it covers a tokenizer that needs the prompt."""
    tokenizer = _tokenizer("metaspace")
    prompt = _ids(tokenizer, _PROMPT)
    prefix = _ids(tokenizer, _RESPONSE[:2])

    assert tokenizer.decode(prefix, skip_special_tokens=True) == "Hello world"
    assert _replay(tokenizer, prompt, prefix) == " Hello world"


@pytest.mark.parametrize("style", ["metaspace", "byte_level"])
# A stop_token_ids entry need not be a special token.
@pytest.mark.parametrize("final_token", ["</s>", "<|stop|>", " world"])
@pytest.mark.parametrize("include_stop_str_in_output", [False, True])
def test_call_stopped_by_a_token_before_the_cut_restores_its_response(
    style, final_token, include_stop_str_in_output
):
    tokenizer = _tokenizer(style)
    prompt = _ids(tokenizer, _PROMPT)
    response = _ids(tokenizer, _RESPONSE + [final_token])
    # vLLM reports EOS with no stop reason and a stop_token_ids entry by its ID.
    stop_reason = None if final_token == "</s>" else response[-1]
    expected, _ = _vllm_response(
        tokenizer,
        prompt,
        response,
        stop_token_terminated=True,
        include_stop_str_in_output=include_stop_str_in_output,
    )

    restored = _replay(
        tokenizer,
        prompt,
        response,
        terminal=_terminal(prompt, response, "stop", stop_reason),
        include_stop_str_in_output=include_stop_str_in_output,
    )

    assert restored == expected


@pytest.mark.parametrize("style", ["metaspace", "byte_level"])
@pytest.mark.parametrize(
    ("min_tokens", "include_stop_str_in_output"),
    [(0, False), (0, True), (len(_STOP_STRING_RESPONSE), False)],
)
def test_call_stopped_by_a_stop_string_before_the_cut_restores_its_response(
    style, min_tokens, include_stop_str_in_output
):
    tokenizer = _tokenizer(style)
    prompt = _ids(tokenizer, _PROMPT)
    stop = ["</answer>"]
    # During min_tokens a stop string is ignored, so the call runs on to the
    # next one: only that one was stripped.
    response = _ids(tokenizer, _STOP_STRING_RESPONSE * 2)
    expected, generated = _vllm_response(
        tokenizer,
        prompt,
        response,
        stop=stop,
        min_tokens=min_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
    )
    prefix = response[:generated]

    restored = _replay(
        tokenizer,
        prompt,
        prefix,
        terminal=_terminal(prompt, prefix, "stop", "</answer>"),
        stop=stop,
        min_tokens=min_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
    )

    assert restored == expected
    assert expected.count("</answer>") == (
        bool(min_tokens) + include_stop_str_in_output
    )


@pytest.mark.parametrize("style", ["metaspace", "byte_level"])
def test_call_cut_off_by_its_length_limit_keeps_every_token(style):
    tokenizer = _tokenizer(style)
    prompt = _ids(tokenizer, _PROMPT)
    response = _ids(tokenizer, _RESPONSE)
    expected, _ = _vllm_response(tokenizer, prompt, response)

    restored = _replay(
        tokenizer,
        prompt,
        response,
        terminal=_terminal(prompt, response, "length"),
    )

    assert restored == expected
    assert restored.endswith(" 4")
