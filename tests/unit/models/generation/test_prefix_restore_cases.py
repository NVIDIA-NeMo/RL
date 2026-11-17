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

"""The native vLLM restore reproduces every shared restore case.

For each case the reply to the resumed call (its text, the token IDs vLLM's
parsers see, and its usage) must equal the reply to the uninterrupted call.
"""

from types import SimpleNamespace

import pytest

detokenizer = pytest.importorskip("vllm.v1.engine.detokenizer")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from nemo_rl.models.generation.prefix_recovery.restore import (  # noqa: E402
    _classify_restored_prefix_terminal,
    plan_restore,
    restore_reply_text,
    restore_reply_usage,
    wrap_parser_for_restore,
)
from tests.unit.models.generation.prefix_restore_cases import (  # noqa: E402
    RESTORE_CASES,
    RestoreCase,
    case_pieces,
)

pytestmark = pytest.mark.vllm

_SPECIAL = ["<unk>", "<s>", "</s>"]


def _tokenizer(style: str) -> PreTrainedTokenizerFast:
    if style == "metaspace":
        # SentencePiece: a prefix decoded on its own loses its leading space.
        space = "▁"
        pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
        decoder = decoders.Metaspace(prepend_scheme="always")
    else:
        space = "Ġ"
        pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        decoder = decoders.ByteLevel()
    pieces = [piece for piece in case_pieces() if piece not in _SPECIAL]
    vocab = {
        token: token_id
        for token_id, token in enumerate(
            _SPECIAL + [piece.replace(" ", space) for piece in pieces]
        )
    }
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizer
    raw.decoder = decoder
    raw.add_special_tokens(_SPECIAL)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    tokenizer.space = space
    return tokenizer


def _ids(tokenizer, pieces) -> list[int]:
    return [
        tokenizer.convert_tokens_to_ids(
            piece if piece in _SPECIAL else piece.replace(" ", tokenizer.space)
        )
        for piece in pieces
    ]


def _engine_text(tokenizer, prompt_ids, output_ids, case, *, min_tokens) -> str:
    """Detokenize one engine call step by step, as vLLM's output processor does."""
    state = detokenizer.IncrementalDetokenizer.from_new_request(
        tokenizer,
        SimpleNamespace(
            request_id="call",
            prompt_token_ids=list(prompt_ids),
            prompt_embeds=None,
            sampling_params=SimpleNamespace(
                stop=list(case.stop) or None,
                min_tokens=min_tokens,
                include_stop_str_in_output=case.include_stop_str_in_output,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            ),
        ),
    )
    for position, token_id in enumerate(output_ids, 1):
        last = position == len(output_ids)
        if state.update([token_id], case.stop_token_terminated and last) is not None:
            break
    return state.get_next_output_text(finished=True, delta=False)


class _RecordingParser:
    def __init__(self) -> None:
        self.token_ids: list[int] | None = None

    def parse(
        self, model_output, request, *, enable_auto_tools, model_output_token_ids
    ):
        self.token_ids = list(model_output_token_ids)
        return model_output


@pytest.mark.parametrize("style", ["metaspace", "byte_level"])
@pytest.mark.parametrize("case", RESTORE_CASES, ids=lambda case: case.name)
def test_resumed_reply_matches_the_uninterrupted_reply(case: RestoreCase, style):
    tokenizer = _tokenizer(style)
    prompt = _ids(tokenizer, case.prompt)
    output = _ids(tokenizer, case.output)
    saved, tail = output[: case.saved], output[case.saved :]
    limit = len(output) + 8
    expected_text = _engine_text(
        tokenizer, prompt, output, case, min_tokens=case.min_tokens
    )

    # The cut saved a root text call: its prompt (not generated), then its tokens.
    plan = plan_restore(
        prefix_token_ids=[],
        saved=SimpleNamespace(
            token_ids_delta=prompt + saved,
            token_mask_delta=[0.0] * len(prompt) + [1.0] * len(saved),
        ),
        continuation=SimpleNamespace(
            effective_output_limit=limit, prefix_token_count=len(saved)
        ),
        min_tokens=case.min_tokens,
    )
    assert plan.engine_prefix_token_ids == prompt + saved
    assert list(plan.resumed_generation_token_ids) == saved
    terminal = _classify_restored_prefix_terminal(
        prompt_token_ids=plan.engine_prefix_token_ids,
        prefix_token_count=len(saved),
        requested_output_tokens=limit,
        model_max_tokens=4096,
        terminal_finish_reason=case.finish_reason if not tail else None,
        terminal_stop_reason=case.stop_reason if not tail else None,
    )
    assert (terminal is not None) == (not tail)

    # The engine decodes only the tail, primed on the restored prompt.
    request = SimpleNamespace(
        _restored_prefix_terminal=terminal,
        _original_min_tokens=case.min_tokens,
        min_tokens=plan.min_tokens or 0,
        stop=list(case.stop) or None,
        include_stop_str_in_output=case.include_stop_str_in_output,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )
    reply = SimpleNamespace(
        text=_engine_text(
            tokenizer, prompt + saved, tail, case, min_tokens=plan.min_tokens or 0
        )
        if tail
        else ""
    )
    restore_reply_text(
        request,
        SimpleNamespace(outputs=[reply]),
        prefix_token_ids=saved,
        engine_prompt_token_ids=prompt + saved,
        tokenizer=tokenizer,
    )
    assert reply.text == expected_text

    # Tool-call and reasoning parsers see the whole output, so they parse it
    # exactly as they would the uninterrupted one.
    parser = _RecordingParser()
    wrap_parser_for_restore(parser, saved).parse(
        reply.text, request, enable_auto_tools=True, model_output_token_ids=tail
    )
    assert parser.token_ids == output

    # vLLM counts the restored tokens as prompt; the reply counts them as output.
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=len(prompt) + len(saved),
            completion_tokens=len(tail),
            total_tokens=len(prompt) + len(output),
            prompt_tokens_details=None,
        ),
        prompt_token_ids=prompt + saved,
    )
    assert restore_reply_usage(
        response, restored_token_count=len(saved), terminal=terminal
    )
    usage = response.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        len(prompt),
        len(output),
        len(prompt) + len(output),
    )
    assert list(response.prompt_token_ids) == prompt
