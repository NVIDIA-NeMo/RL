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
    assert_splice_tokenizer_invariants,
    replace_prefix_tokens,
    splice_prefix_into_rendered_prompt,
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


def _toy_encode(text: str) -> list[int]:
    """One token per character, with the EOS string "|" mapped to id 2."""
    return [2 if ch == "|" else ord(ch) for ch in text]


def test_splice_prefix_into_rendered_prompt_tokenizes_only_the_suffix():
    encoded: list[str] = []

    def encode(text: str) -> list[int]:
        encoded.append(text)
        return _toy_encode(text)

    # Model prefix: "s|" + generated "ab" + EOS; the template re-renders it as "AB".
    model_prefix_token_ids = _toy_encode("s|") + _toy_encode("ab|")
    result = splice_prefix_into_rendered_prompt(
        model_prefix_token_ids,
        "s|AB|tool|>",
        prefix_text="s|AB|",
        eos_token="|",
        eos_token_id=2,
        encode=encode,
    )
    assert result == _toy_encode("s|ab|tool|>")
    assert encoded == ["|tool|>"]


def test_splice_prefix_into_rendered_prompt_returns_none_without_enough_eos():
    model_prefix_token_ids = _toy_encode("s|ab|c|")
    assert (
        splice_prefix_into_rendered_prompt(
            model_prefix_token_ids,
            "s|AB|",
            prefix_text="s|AB|c|",
            eos_token="|",
            eos_token_id=2,
            encode=_toy_encode,
        )
        is None
    )


def _render(tokenizer, messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )


def _encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _legacy_and_incremental(tokenizer, messages, model_prefix):
    """The legacy and incremental splices for one request, side by side."""
    rendered = _render(tokenizer, messages, add_generation_prompt=True)
    full = _encode(tokenizer, rendered)
    last_assistant = max(
        i for i, m in enumerate(messages) if m["role"] == "assistant"
    )
    prefix_text = _render(tokenizer, messages[: last_assistant + 1], False)
    legacy = replace_prefix_tokens(
        tokenizer=tokenizer,
        model_prefix_token_ids=model_prefix,
        template_prefix_token_ids=_encode(tokenizer, prefix_text),
        template_token_ids=full,
    )
    incremental = splice_prefix_into_rendered_prompt(
        model_prefix,
        rendered,
        prefix_text=prefix_text,
        eos_token=tokenizer.eos_token,
        eos_token_id=tokenizer.eos_token_id,
        encode=lambda text: _encode(tokenizer, text),
    )
    return legacy, incremental


@pytest.mark.parametrize("generation_ends_with_eos", [True, False])
def test_incremental_splice_matches_legacy_on_qwen3(
    generation_ends_with_eos,
):
    """The incremental splice agrees with legacy on a multi-turn Qwen3 conversation.

    Covers whitespace the template re-renders differently, merged consecutive
    tool messages, a mid-rollout user turn (which strips earlier reasoning), and
    generations that stopped without EOS.
    """
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    eos = tokenizer.eos_token_id

    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug."},
    ]
    model_prefix = None
    for turn in range(4):
        rendered = _render(tokenizer, messages, add_generation_prompt=True)
        full = _encode(tokenizer, rendered)
        if model_prefix is None:
            prompt = full
        else:
            legacy, incremental = _legacy_and_incremental(
                tokenizer, messages, model_prefix
            )
            assert incremental == legacy
            prompt = legacy

        content = f"<think>\nstep {turn}\n</think>\n\n Running ls.  \n"
        generation = _encode(tokenizer, content)
        if generation_ends_with_eos:
            generation.append(eos)
        model_prefix = prompt + generation
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "tool", "content": f"out {turn}"})
        messages.append({"role": "tool", "content": f"err {turn}"})
        if turn == 1:
            messages.append({"role": "user", "content": "Keep going."})


def _qwen3_tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def _ordinary(tokenizer, text):
    """Tokenize text the way a model would spell it: no special-token matching."""
    return tokenizer(text, add_special_tokens=False, split_special_tokens=True)[
        "input_ids"
    ]


def _two_turn_history(tokenizer):
    """One clean assistant/tool exchange; returns (messages, model_prefix, prompt)."""
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug."},
    ]
    prompt = _encode(tokenizer, _render(tokenizer, messages, True))
    generation = _ordinary(tokenizer, "ls") + [tokenizer.eos_token_id]
    model_prefix = prompt + generation
    messages.append({"role": "assistant", "content": "ls"})
    messages.append({"role": "tool", "content": "f.py"})
    return messages, model_prefix, prompt


def test_incremental_matches_legacy_when_model_spells_eos():
    """A generation may spell the EOS string with ordinary tokens (review A/B).

    The echoed prefix ids then hold fewer EOS ids than the re-rendered history;
    a boundary count anchored on the prefix ids cuts one turn early and
    duplicates content, while counting the rendered prefix matches legacy.
    """
    tokenizer = _qwen3_tokenizer()
    eos_s = tokenizer.eos_token
    im_start = eos_s.replace("im_end", "im_start")
    messages, model_prefix, prompt = _two_turn_history(tokenizer)

    # Turn 2's prompt: the first turn was clean, so the template ids are the
    # model ids. The model then spells a fake turn boundary with ordinary
    # tokens.
    turn2_prompt = _encode(tokenizer, _render(tokenizer, messages, True))
    spelled = "x" + eos_s + "\n" + im_start + "user\nhi"
    generation = _ordinary(tokenizer, spelled) + [tokenizer.eos_token_id]
    model_prefix = turn2_prompt + generation
    messages.append(
        {
            "role": "assistant",
            "content": tokenizer.decode(generation, skip_special_tokens=True),
        }
    )
    messages.append({"role": "tool", "content": "real tool output"})

    legacy, incremental = _legacy_and_incremental(tokenizer, messages, model_prefix)
    assert incremental == legacy
    decoded = tokenizer.decode(legacy)
    # The first turn survived; the real tool turn survived; the spelled fake
    # turn was not duplicated.
    assert "f.py" in decoded
    assert "real tool output" in decoded
    assert decoded.count("user\nhi") == 1


def test_incremental_matches_legacy_on_double_counted_echo():
    """Review case I: the echoed prompt ids already contain the generation.

    Neither path repairs the duplication (the model prefix is kept verbatim);
    the incremental splice must reproduce legacy exactly, including keeping the
    new tool turn.
    """
    tokenizer = _qwen3_tokenizer()
    messages, model_prefix, prompt = _two_turn_history(tokenizer)

    turn2_prompt = _encode(tokenizer, _render(tokenizer, messages, True))
    generation = _ordinary(tokenizer, "cat f.py") + [tokenizer.eos_token_id]
    # Double-counted echo: prompt ids already end with the generation.
    model_prefix = turn2_prompt + generation + generation
    messages.append(
        {
            "role": "assistant",
            "content": tokenizer.decode(
                prompt[len(prompt):] + generation, skip_special_tokens=True
            ),
        }
    )
    messages.append({"role": "tool", "content": "42"})

    legacy, incremental = _legacy_and_incremental(tokenizer, messages, model_prefix)
    assert incremental == legacy
    assert "42" in tokenizer.decode(legacy)


def test_incremental_matches_legacy_when_reasoning_strip_drops_a_literal_eos():
    """Review case F: a literal EOS inside reasoning the template strips.

    Legacy over-counts here (the prefix render keeps the reasoning, the full
    render strips it) and drops the tool response. The incremental splice
    inherits legacy's behaviour exactly -- parity, not repair.
    """
    tokenizer = _qwen3_tokenizer()
    eos_s = tokenizer.eos_token
    messages, model_prefix, prompt = _two_turn_history(tokenizer)

    turn2_prompt = _encode(tokenizer, _render(tokenizer, messages, True))
    think = "<think>\nthe " + eos_s + " token\n</think>\n\nok"
    generation = _ordinary(tokenizer, think) + [tokenizer.eos_token_id]
    model_prefix = turn2_prompt + generation
    messages.append(
        {
            "role": "assistant",
            "content": tokenizer.decode(generation, skip_special_tokens=True),
        }
    )
    messages.append({"role": "tool", "content": "o2"})
    messages.append({"role": "user", "content": "Keep going."})

    legacy, incremental = _legacy_and_incremental(tokenizer, messages, model_prefix)
    assert incremental == legacy


def test_assert_splice_tokenizer_invariants_qwen3_passes():
    assert_splice_tokenizer_invariants(_qwen3_tokenizer())


def test_assert_splice_tokenizer_invariants_rejects_split_eos():
    """A tokenizer that does not treat the EOS string as special must fail."""

    class _SplitEos:
        eos_token = "<" + "|im_end|>"
        eos_token_id = 99

        def __call__(self, text, add_special_tokens=False):
            # Char-level: the EOS string is not special and merges freely.
            return {"input_ids": [ord(c) for c in text]}

    with pytest.raises(RuntimeError, match="prefix_splice"):
        assert_splice_tokenizer_invariants(_SplitEos())
