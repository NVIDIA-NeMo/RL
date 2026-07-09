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

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

from tokenizers import Tokenizer


class IncrementalTokenizerError(RuntimeError):
    """Base error for an exact incremental-tokenizer session."""


class IncrementalTokenizerSessionNotFoundError(IncrementalTokenizerError):
    """Raised when an incremental-tokenizer session is absent or expired."""


@dataclass(frozen=True)
class IncrementalTokenizationStep:
    """Metrics and tokens produced by one incremental-tokenizer operation."""

    sequence_no: int
    token_count: int
    encoded_chars: int
    encoded_tokens: int
    reused_tokens: int
    rollback_tokens: int
    checkpoint_count: int
    checkpoint_tokens: int
    checkpoint_mismatches: int
    incremental_valid: bool
    tokens: list[int] | None = None


class ExactIncrementalTokenizer:
    """Incrementally encode a Qwen byte-level-BPE rendered prompt exactly.

    The complete rendered prompt is normalized to find the exact changed
    frontier. Only the final mutable pre-tokenizer segment and the changed tail
    are BPE encoded again. Full-tokenizer checkpoints remain authoritative.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        rendered_prompt: str,
        authoritative_token_ids: list[int],
        backend_tokenizer: Any | None = None,
        added_tokens: tuple[str, ...] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._backend_tokenizer = (
            backend_tokenizer
            if backend_tokenizer is not None
            else tokenizer.backend_tokenizer
        )
        self._validate_tokenizer()
        self._added_tokens = (
            added_tokens
            if added_tokens is not None
            else tuple(tokenizer.get_added_vocab())
        )
        self.reset(
            rendered_prompt=rendered_prompt,
            authoritative_token_ids=authoritative_token_ids,
        )

    def _validate_tokenizer(self) -> None:
        if not getattr(self._tokenizer, "is_fast", False):
            raise IncrementalTokenizerError("a fast tokenizer is required")
        backend = self._backend_tokenizer
        if backend is None or backend.pre_tokenizer is None:
            raise IncrementalTokenizerError(
                "a backend pre-tokenizer with exact offsets is required"
            )
        normalizer = backend.normalizer
        if normalizer is not None and type(normalizer).__name__ != "NFC":
            raise IncrementalTokenizerError(
                f"unsupported tokenizer normalizer: {type(normalizer).__name__}"
            )

    def _normalize(self, text: str) -> str:
        normalizer = self._backend_tokenizer.normalizer
        if normalizer is None:
            return text
        return normalizer.normalize_str(text)

    def _encode_with_offsets(
        self, text: str
    ) -> tuple[list[int], list[tuple[int, int]]]:
        # The transformers fast-tokenizer wrapper mutates backend truncation and
        # padding settings on every call. Concurrent vLLM HTTP requests can then
        # fail with `RuntimeError: Already borrowed`. The backend encode path is
        # immutable for this no-special-token operation and returns the same IDs
        # and exact offsets without touching those shared settings.
        encoding = self._backend_tokenizer.encode(text, add_special_tokens=False)
        token_ids = list(encoding.ids)
        offsets = [tuple(offset) for offset in encoding.offsets]
        if len(token_ids) != len(offsets):
            raise IncrementalTokenizerError(
                "token IDs and offset mapping have different lengths"
            )
        return token_ids, offsets

    def _pretoken_offsets(self, text: str) -> list[tuple[int, int]]:
        return [
            tuple(offset)
            for _, offset in self._backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                text
            )
        ]

    @staticmethod
    def _common_suffix_length(left: list[int], right: list[int]) -> int:
        length = 0
        for left_token, right_token in zip(reversed(left), reversed(right)):
            if left_token != right_token:
                break
            length += 1
        return length

    @staticmethod
    def _common_prefix_chars(left: str, right: str) -> int:
        for index, (left_char, right_char) in enumerate(zip(left, right)):
            if left_char != right_char:
                return index
        return min(len(left), len(right))

    def _align_authoritative_offsets(
        self,
        *,
        template_token_ids: list[int],
        template_offsets: list[tuple[int, int]],
        authoritative_token_ids: list[int],
    ) -> list[tuple[int, int] | None]:
        common_suffix = self._common_suffix_length(
            template_token_ids, authoritative_token_ids
        )
        if not common_suffix:
            raise IncrementalTokenizerError(
                "authoritative tokens do not share a template-token suffix"
            )
        return [None] * (len(authoritative_token_ids) - common_suffix) + list(
            template_offsets[len(template_token_ids) - common_suffix :]
        )

    def reset(
        self,
        *,
        rendered_prompt: str,
        authoritative_token_ids: list[int],
    ) -> None:
        """Reset mutable state from a complete authoritative checkpoint."""
        normalized_prompt = self._normalize(rendered_prompt)
        template_token_ids, template_offsets = self._encode_with_offsets(
            normalized_prompt
        )
        self._prompt = normalized_prompt
        self._token_ids = list(authoritative_token_ids)
        self._offsets = self._align_authoritative_offsets(
            template_token_ids=template_token_ids,
            template_offsets=template_offsets,
            authoritative_token_ids=authoritative_token_ids,
        )
        self._pretoken_offsets_cache = self._pretoken_offsets(normalized_prompt)

    def _added_token_repair_start(self, common_prefix_chars: int) -> int:
        repair_start = common_prefix_chars
        prompt_prefix = self._prompt[:common_prefix_chars]
        for added_token in self._added_tokens:
            max_prefix = min(len(added_token) - 1, len(prompt_prefix))
            for prefix_length in range(1, max_prefix + 1):
                if prompt_prefix.endswith(added_token[:prefix_length]):
                    repair_start = min(
                        repair_start, common_prefix_chars - prefix_length
                    )
        return repair_start

    def _pretoken_repair_start(self, frontier: int) -> int:
        repair_start = 0
        for start, end in self._pretoken_offsets_cache:
            if start < frontier <= end:
                repair_start = start
                break
            if end <= frontier:
                repair_start = start
            elif start >= frontier:
                break
        return repair_start

    def _repair_start(self, common_prefix_chars: int) -> int:
        if common_prefix_chars <= 0:
            raise IncrementalTokenizerError(
                "the rendered prompt changed before any stable character"
            )
        repair_start = self._pretoken_repair_start(common_prefix_chars)
        repair_start = min(
            repair_start,
            self._added_token_repair_start(common_prefix_chars),
        )
        while True:
            retained_tokens = self._retained_token_count(repair_start)
            first_mutable_offset = self._offsets[retained_tokens]
            if first_mutable_offset is None:
                raise IncrementalTokenizerError(
                    "repair frontier crosses an authoritative unmapped token"
                )
            if first_mutable_offset[0] >= repair_start:
                return repair_start
            aligned_start = self._pretoken_repair_start(first_mutable_offset[0])
            if aligned_start >= repair_start:
                raise IncrementalTokenizerError(
                    "token offsets cannot be aligned to a pre-token boundary"
                )
            repair_start = aligned_start

    def _retained_token_count(self, repair_start: int) -> int:
        retained = 0
        mapped_offset_seen = False
        for offset in self._offsets:
            if offset is None:
                if mapped_offset_seen:
                    raise IncrementalTokenizerError(
                        "unmapped authoritative token after mapped prompt tokens"
                    )
                retained += 1
                continue
            mapped_offset_seen = True
            if offset[1] <= repair_start:
                retained += 1
                continue
            break
        if retained == len(self._offsets):
            raise IncrementalTokenizerError(
                "repair frontier did not include a mutable token"
            )
        return retained

    def append(self, rendered_prompt: str) -> IncrementalTokenizationStep:
        """Encode only the exact mutable tail of a revised rendered prompt."""
        normalized_prompt = self._normalize(rendered_prompt)
        if normalized_prompt == self._prompt:
            return IncrementalTokenizationStep(
                sequence_no=-1,
                token_count=len(self._token_ids),
                encoded_chars=0,
                encoded_tokens=0,
                reused_tokens=len(self._token_ids),
                rollback_tokens=0,
                checkpoint_count=0,
                checkpoint_tokens=0,
                checkpoint_mismatches=0,
                incremental_valid=True,
            )

        common_prefix_chars = self._common_prefix_chars(self._prompt, normalized_prompt)
        repair_start = self._repair_start(common_prefix_chars)
        retained_tokens = self._retained_token_count(repair_start)
        old_token_count = len(self._token_ids)
        encoded_text = normalized_prompt[repair_start:]
        tail_token_ids, tail_offsets = self._encode_with_offsets(encoded_text)
        adjusted_tail_offsets = [
            (start + repair_start, end + repair_start) for start, end in tail_offsets
        ]

        self._prompt = normalized_prompt
        self._token_ids = self._token_ids[:retained_tokens] + tail_token_ids
        self._offsets = self._offsets[:retained_tokens] + adjusted_tail_offsets
        stable_pretokens = [
            offset
            for offset in self._pretoken_offsets_cache
            if offset[1] <= repair_start
        ]
        tail_pretokens = [
            (start + repair_start, end + repair_start)
            for start, end in self._pretoken_offsets(encoded_text)
        ]
        self._pretoken_offsets_cache = stable_pretokens + tail_pretokens

        return IncrementalTokenizationStep(
            sequence_no=-1,
            token_count=len(self._token_ids),
            encoded_chars=len(encoded_text),
            encoded_tokens=len(tail_token_ids),
            reused_tokens=retained_tokens,
            rollback_tokens=old_token_count - retained_tokens,
            checkpoint_count=0,
            checkpoint_tokens=0,
            checkpoint_mismatches=0,
            incremental_valid=True,
        )

    @property
    def token_ids(self) -> list[int]:
        """Return the current incrementally constructed prompt token IDs."""
        return list(self._token_ids)


@dataclass
class _IncrementalTokenizerSession:
    encoder: ExactIncrementalTokenizer | None
    sequence_no: int
    expires_at: float
    incremental_valid: bool = True
    last_prompt: str = ""
    last_result: IncrementalTokenizationStep | None = None


class ExactIncrementalTokenizerSessionManager:
    """Bounded, ordered sessions for exact suffix-only tokenization."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        max_sessions: int,
        session_ttl_seconds: float,
        checkpoint_interval: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        self._tokenizer = tokenizer
        # Isolate incremental encoding from the transformers/vLLM tokenizer.
        # Its wrapper mutates truncation and padding state while serving normal
        # generation requests, which is unsafe to share with this HTTP path.
        source_backend = tokenizer.backend_tokenizer
        self._backend_tokenizer = (
            Tokenizer.from_str(source_backend.to_str())
            if hasattr(source_backend, "to_str")
            else deepcopy(source_backend)
        )
        self._added_tokens = tuple(tokenizer.get_added_vocab())
        self._max_sessions = max_sessions
        self._session_ttl_seconds = session_ttl_seconds
        self._checkpoint_interval = checkpoint_interval
        self._clock = clock
        self._sessions: OrderedDict[str, _IncrementalTokenizerSession] = OrderedDict()

    def _expire_sessions(self) -> None:
        now = self._clock()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _get_session(self, session_id: str) -> _IncrementalTokenizerSession:
        self._expire_sessions()
        session = self._sessions.get(session_id)
        if session is None:
            raise IncrementalTokenizerSessionNotFoundError(
                f"incremental tokenizer session not found: {session_id}"
            )
        return session

    def _refresh(self, session_id: str, session: _IncrementalTokenizerSession) -> None:
        session.expires_at = self._clock() + self._session_ttl_seconds
        self._sessions.move_to_end(session_id)

    def start(
        self,
        *,
        session_id: str,
        sequence_no: int,
        rendered_prompt: str,
        authoritative_token_ids: list[int],
    ) -> IncrementalTokenizationStep:
        """Start a session from one complete tokenizer checkpoint."""
        self._expire_sessions()
        if sequence_no != 0:
            raise IncrementalTokenizerError(
                f"first sequence must be 0, got {sequence_no}"
            )
        existing_session = self._sessions.get(session_id)
        if existing_session is not None:
            if (
                existing_session.sequence_no == sequence_no
                and existing_session.last_prompt == rendered_prompt
                and existing_session.encoder is not None
                and existing_session.encoder.token_ids == authoritative_token_ids
                and existing_session.last_result is not None
            ):
                self._refresh(session_id, existing_session)
                return existing_session.last_result
            raise IncrementalTokenizerError(
                f"incremental tokenizer session already exists: {session_id}"
            )
        if len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)
        session = _IncrementalTokenizerSession(
            encoder=ExactIncrementalTokenizer(
                tokenizer=self._tokenizer,
                rendered_prompt=rendered_prompt,
                authoritative_token_ids=authoritative_token_ids,
                backend_tokenizer=self._backend_tokenizer,
                added_tokens=self._added_tokens,
            ),
            sequence_no=sequence_no,
            expires_at=self._clock() + self._session_ttl_seconds,
            last_prompt=rendered_prompt,
        )
        result = IncrementalTokenizationStep(
            sequence_no=sequence_no,
            token_count=len(authoritative_token_ids),
            encoded_chars=0,
            encoded_tokens=0,
            reused_tokens=0,
            rollback_tokens=0,
            checkpoint_count=1,
            checkpoint_tokens=len(authoritative_token_ids),
            checkpoint_mismatches=0,
            incremental_valid=True,
        )
        session.last_result = result
        self._sessions[session_id] = session
        return result

    def requires_checkpoint(self, *, session_id: str, sequence_no: int) -> bool:
        """Return whether an append must include authoritative full tokens."""
        session = self._get_session(session_id)
        if sequence_no == session.sequence_no:
            return False
        if sequence_no != session.sequence_no + 1:
            raise IncrementalTokenizerError(
                f"expected sequence {session.sequence_no + 1}, got {sequence_no}"
            )
        return (
            not session.incremental_valid
            or sequence_no % self._checkpoint_interval == 0
        )

    def append(
        self,
        *,
        session_id: str,
        sequence_no: int,
        rendered_prompt: str,
        authoritative_token_ids: list[int] | None = None,
    ) -> IncrementalTokenizationStep:
        """Append one ordered prompt revision and optionally verify it."""
        session = self._get_session(session_id)
        if sequence_no == session.sequence_no:
            if rendered_prompt != session.last_prompt or session.last_result is None:
                raise IncrementalTokenizerError(
                    "same sequence number was retried with a different prompt"
                )
            return session.last_result
        if sequence_no != session.sequence_no + 1:
            raise IncrementalTokenizerError(
                f"expected sequence {session.sequence_no + 1}, got {sequence_no}"
            )
        if not session.incremental_valid and authoritative_token_ids is None:
            raise IncrementalTokenizerError(
                "disabled incremental session requires a full checkpoint"
            )

        if session.incremental_valid:
            assert session.encoder is not None
            incremental_step = session.encoder.append(rendered_prompt)
        else:
            incremental_step = IncrementalTokenizationStep(
                sequence_no=sequence_no,
                token_count=len(authoritative_token_ids or []),
                encoded_chars=0,
                encoded_tokens=0,
                reused_tokens=0,
                rollback_tokens=0,
                checkpoint_count=0,
                checkpoint_tokens=0,
                checkpoint_mismatches=0,
                incremental_valid=False,
            )
        checkpoint_count = int(authoritative_token_ids is not None)
        checkpoint_tokens = (
            len(authoritative_token_ids) if authoritative_token_ids is not None else 0
        )
        checkpoint_mismatches = 0
        if authoritative_token_ids is not None:
            if (
                session.encoder is None
                or session.encoder.token_ids != authoritative_token_ids
            ):
                checkpoint_mismatches = 1
                session.incremental_valid = False
            if session.encoder is not None:
                try:
                    session.encoder.reset(
                        rendered_prompt=rendered_prompt,
                        authoritative_token_ids=authoritative_token_ids,
                    )
                except IncrementalTokenizerError:
                    session.encoder = None
                    session.incremental_valid = False

        if authoritative_token_ids is not None:
            token_count = len(authoritative_token_ids)
        else:
            assert session.encoder is not None
            token_count = len(session.encoder.token_ids)
        result = IncrementalTokenizationStep(
            sequence_no=sequence_no,
            token_count=token_count,
            encoded_chars=incremental_step.encoded_chars,
            encoded_tokens=incremental_step.encoded_tokens,
            reused_tokens=incremental_step.reused_tokens,
            rollback_tokens=incremental_step.rollback_tokens,
            checkpoint_count=checkpoint_count,
            checkpoint_tokens=checkpoint_tokens,
            checkpoint_mismatches=checkpoint_mismatches,
            incremental_valid=session.incremental_valid,
        )
        session.sequence_no = sequence_no
        session.last_prompt = rendered_prompt
        session.last_result = result
        self._refresh(session_id, session)
        return result

    def finalize(
        self,
        *,
        session_id: str,
        sequence_no: int,
        rendered_prompt: str,
        authoritative_token_ids: list[int] | None = None,
    ) -> IncrementalTokenizationStep:
        """Finalize with exact incremental tokens or a requested checkpoint."""
        try:
            result = self.append(
                session_id=session_id,
                sequence_no=sequence_no,
                rendered_prompt=rendered_prompt,
                authoritative_token_ids=authoritative_token_ids,
            )
            session = self._get_session(session_id)
            if authoritative_token_ids is not None:
                final_token_ids = authoritative_token_ids
            else:
                if not session.incremental_valid or session.encoder is None:
                    raise IncrementalTokenizerError(
                        "finalization without a checkpoint requires a valid "
                        "incremental session"
                    )
                final_token_ids = session.encoder.token_ids
            return IncrementalTokenizationStep(
                sequence_no=result.sequence_no,
                token_count=len(final_token_ids),
                encoded_chars=result.encoded_chars,
                encoded_tokens=result.encoded_tokens,
                reused_tokens=result.reused_tokens,
                rollback_tokens=result.rollback_tokens,
                checkpoint_count=result.checkpoint_count,
                checkpoint_tokens=result.checkpoint_tokens,
                checkpoint_mismatches=result.checkpoint_mismatches,
                incremental_valid=result.incremental_valid,
                tokens=list(final_token_ids),
            )
        finally:
            self._sessions.pop(session_id, None)

    def abort(self, session_id: str) -> bool:
        """Remove a session, returning whether it existed."""
        return self._sessions.pop(session_id, None) is not None
