from __future__ import annotations

import pytest

from nemo_rl.models.generation.vllm.incremental_tokenizer import (
    ExactIncrementalTokenizer,
    ExactIncrementalTokenizerSessionManager,
    IncrementalTokenizerError,
    IncrementalTokenizerSessionNotFoundError,
)


class _Encoding:
    def __init__(self, text: str) -> None:
        self.ids = [ord(char) for char in text]
        self.offsets = [(index, index + 1) for index in range(len(text))]


class _Normalizer:
    def normalize_str(self, text: str) -> str:
        return text


class _BackendNormalizer(_Normalizer):
    pass


class NFC(_BackendNormalizer):
    pass


class _PreTokenizer:
    def pre_tokenize_str(self, text: str):
        if not text:
            return []
        parts = []
        start = 0
        for index, char in enumerate(text):
            if char.isspace():
                if start < index:
                    parts.append((text[start:index], (start, index)))
                parts.append((char, (index, index + 1)))
                start = index + 1
        if start < len(text):
            parts.append((text[start:], (start, len(text))))
        return parts


class _Backend:
    normalizer = NFC()
    pre_tokenizer = _PreTokenizer()

    def encode(self, text: str, *, add_special_tokens: bool) -> _Encoding:
        assert not add_special_tokens
        return _Encoding(text)


class _CharTokenizer:
    is_fast = True
    backend_tokenizer = _Backend()

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def get_added_vocab(self):
        return {"<special>": 1000}


def _encode(tokenizer: _CharTokenizer, prompt: str) -> list[int]:
    return list(
        tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )["input_ids"]
    )


def test_incremental_tokenizer_reencodes_only_mutable_tail():
    tokenizer = _CharTokenizer()
    initial = "immutable\nabc\nsuffix"
    updated = "immutable\nabcdef\nsuffix"
    encoder = ExactIncrementalTokenizer(
        tokenizer=tokenizer,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )

    step = encoder.append(updated)

    assert encoder.token_ids == _encode(tokenizer, updated)
    assert step.encoded_chars < len(updated)
    assert step.reused_tokens > 0
    assert step.rollback_tokens > 0


def test_incremental_tokenizer_does_not_call_mutating_wrapper():
    class _WrapperForbiddenTokenizer(_CharTokenizer):
        def __call__(self, *args, **kwargs):
            raise AssertionError("transformers tokenizer wrapper must not be called")

    tokenizer = _WrapperForbiddenTokenizer()
    initial = "immutable\nabc\nsuffix"
    updated = "immutable\nabcdef\nsuffix"
    encoder = ExactIncrementalTokenizer(
        tokenizer=tokenizer,
        rendered_prompt=initial,
        authoritative_token_ids=[ord(char) for char in initial],
    )

    encoder.append(updated)

    assert encoder.token_ids == [ord(char) for char in updated]


def test_incremental_tokenizer_revises_middle_truncated_final_output():
    tokenizer = _CharTokenizer()
    raw_output = "0123456789" * 4_000
    initial = f"prefix\n{raw_output[:32_000]}\nsuffix"
    final_output = (
        raw_output[:15_000]
        + "\n[... Observation truncated due to length ...]\n"
        + raw_output[-15_000:]
    )
    final = f"prefix\n{final_output}\nsuffix"
    encoder = ExactIncrementalTokenizer(
        tokenizer=tokenizer,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )

    encoder.append(final)

    assert encoder.token_ids == _encode(tokenizer, final)


def test_incremental_tokenizer_preserves_authoritative_prefix_replacement():
    tokenizer = _CharTokenizer()
    initial = "template history\nabc\nsuffix"
    updated = "template history\nabcdef\nsuffix"
    template_tokens = _encode(tokenizer, initial)
    authoritative = [9001, 9002] + template_tokens[8:]
    encoder = ExactIncrementalTokenizer(
        tokenizer=tokenizer,
        rendered_prompt=initial,
        authoritative_token_ids=authoritative,
    )

    encoder.append(updated)

    assert encoder.token_ids[:2] == [9001, 9002]
    assert encoder.token_ids[-len("abcdef\nsuffix") :] == _encode(
        tokenizer, "abcdef\nsuffix"
    )


def test_incremental_tokenizer_rolls_back_partial_added_token():
    tokenizer = _CharTokenizer()
    initial = "prefix <spec\nsuffix"
    updated = "prefix <special>\nsuffix"
    encoder = ExactIncrementalTokenizer(
        tokenizer=tokenizer,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )

    step = encoder.append(updated)

    assert encoder.token_ids == _encode(tokenizer, updated)
    assert step.rollback_tokens >= len("<spec")


def test_incremental_tokenizer_manager_detects_checkpoint_mismatch():
    tokenizer = _CharTokenizer()
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=2,
        session_ttl_seconds=30,
        checkpoint_interval=2,
    )
    initial = "prefix a suffix"
    updated = "prefix ab suffix"
    manager.start(
        session_id="session",
        sequence_no=0,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )

    result = manager.append(
        session_id="session",
        sequence_no=1,
        rendered_prompt=updated,
        authoritative_token_ids=[999],
    )

    assert result.checkpoint_mismatches == 1
    assert not result.incremental_valid
    assert manager.requires_checkpoint(session_id="session", sequence_no=2)


def test_incremental_tokenizer_manager_finalizes_with_incremental_tokens():
    tokenizer = _CharTokenizer()
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=2,
        session_ttl_seconds=30,
        checkpoint_interval=8,
    )
    initial = "prefix a suffix"
    final = "prefix abc suffix"
    manager.start(
        session_id="session",
        sequence_no=0,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )

    result = manager.finalize(
        session_id="session",
        sequence_no=1,
        rendered_prompt=final,
    )

    assert result.incremental_valid
    assert result.checkpoint_count == 0
    assert result.checkpoint_mismatches == 0
    assert result.tokens == _encode(tokenizer, final)
    with pytest.raises(IncrementalTokenizerSessionNotFoundError):
        manager.requires_checkpoint(session_id="session", sequence_no=2)


def test_incremental_tokenizer_manager_finalizes_with_checkpoint_tokens():
    tokenizer = _CharTokenizer()
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=2,
        session_ttl_seconds=30,
        checkpoint_interval=2,
    )
    initial = "prefix a suffix"
    middle = "prefix ab suffix"
    final = "prefix abc suffix"
    manager.start(
        session_id="session",
        sequence_no=0,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )
    manager.append(
        session_id="session",
        sequence_no=1,
        rendered_prompt=middle,
    )

    assert manager.requires_checkpoint(session_id="session", sequence_no=2)
    result = manager.finalize(
        session_id="session",
        sequence_no=2,
        rendered_prompt=final,
        authoritative_token_ids=_encode(tokenizer, final),
    )

    assert result.incremental_valid
    assert result.checkpoint_count == 1
    assert result.checkpoint_mismatches == 0
    assert result.tokens == _encode(tokenizer, final)


def test_incremental_tokenizer_manager_rejects_unchecked_invalid_final():
    tokenizer = _CharTokenizer()
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=2,
        session_ttl_seconds=30,
        checkpoint_interval=2,
    )
    initial = "prefix a suffix"
    manager.start(
        session_id="session",
        sequence_no=0,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )
    manager.append(
        session_id="session",
        sequence_no=1,
        rendered_prompt="prefix ab suffix",
        authoritative_token_ids=[999],
    )

    with pytest.raises(
        IncrementalTokenizerError,
        match="disabled incremental session requires a full checkpoint",
    ):
        manager.finalize(
            session_id="session",
            sequence_no=2,
            rendered_prompt="prefix abc suffix",
        )


def test_incremental_tokenizer_manager_sequence_and_idempotency():
    tokenizer = _CharTokenizer()
    manager = ExactIncrementalTokenizerSessionManager(
        tokenizer=tokenizer,
        max_sessions=2,
        session_ttl_seconds=30,
        checkpoint_interval=8,
    )
    initial = "prefix a suffix"
    first = manager.start(
        session_id="session",
        sequence_no=0,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )

    repeated_start = manager.start(
        session_id="session",
        sequence_no=0,
        rendered_prompt=initial,
        authoritative_token_ids=_encode(tokenizer, initial),
    )
    assert repeated_start == first
    assert not manager.requires_checkpoint(session_id="session", sequence_no=0)

    repeated = manager.append(
        session_id="session",
        sequence_no=0,
        rendered_prompt=initial,
    )
    assert repeated.sequence_no == 0
    with pytest.raises(IncrementalTokenizerError):
        manager.append(
            session_id="session",
            sequence_no=0,
            rendered_prompt="different",
        )
