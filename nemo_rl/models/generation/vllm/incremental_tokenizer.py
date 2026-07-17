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
from dataclasses import dataclass, replace
from time import monotonic
from typing import Any, Callable

from tokenizers import Tokenizer


class IncrementalTokenizerError(RuntimeError):
    """Base error for an exact incremental-tokenizer session."""


class IncrementalTokenizerSessionNotFoundError(IncrementalTokenizerError):
    """Raised when an incremental-tokenizer session is absent or expired."""


class IncrementalTokenizerPrefixSeedError(IncrementalTokenizerError):
    """Raised when an authoritative-prefix fast seed cannot be proven exact."""


class IncrementalTokenizerStablePrefixError(IncrementalTokenizerError):
    """Raised when a stable token prefix cannot be proven from two renders."""


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


@dataclass(frozen=True)
class IncrementalTokenizerPrefixSeed:
    """Work performed while seeding from prior authoritative model tokens."""

    reused_tokens: int
    encoded_chars: int
    encoded_tokens: int


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
        self._initialize_tokenizer(
            tokenizer=tokenizer,
            backend_tokenizer=backend_tokenizer,
            added_tokens=added_tokens,
        )
        self.reset(
            rendered_prompt=rendered_prompt,
            authoritative_token_ids=authoritative_token_ids,
        )

    def _initialize_tokenizer(
        self,
        *,
        tokenizer: Any,
        backend_tokenizer: Any | None,
        added_tokens: tuple[str, ...] | None,
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

    @classmethod
    def from_authoritative_prefix(
        cls,
        *,
        tokenizer: Any,
        rendered_prompt: str,
        template_prefix_prompt: str,
        authoritative_prefix_token_ids: list[int],
        backend_tokenizer: Any | None = None,
        added_tokens: tuple[str, ...] | None = None,
    ) -> tuple[ExactIncrementalTokenizer, IncrementalTokenizerPrefixSeed]:
        """Seed an exact session without encoding the immutable conversation.

        The last chat-template EOS is an added-token boundary. Consequently,
        tokenization after that boundary is independent of the preceding model
        output. The prior model tokens remain authoritative and only the short
        template suffix containing the tool response is encoded.
        """
        encoder = cls.__new__(cls)
        encoder._initialize_tokenizer(
            tokenizer=tokenizer,
            backend_tokenizer=backend_tokenizer,
            added_tokens=added_tokens,
        )
        seed = encoder._reset_from_authoritative_prefix(
            rendered_prompt=rendered_prompt,
            template_prefix_prompt=template_prefix_prompt,
            authoritative_prefix_token_ids=authoritative_prefix_token_ids,
        )
        return encoder, seed

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

    @staticmethod
    def _common_suffix_chars(
        left: str,
        right: str,
        *,
        common_prefix_chars: int,
    ) -> int:
        max_suffix_chars = min(
            len(left) - common_prefix_chars,
            len(right) - common_prefix_chars,
        )
        suffix_chars = 0
        while (
            suffix_chars < max_suffix_chars
            and left[-suffix_chars - 1] == right[-suffix_chars - 1]
        ):
            suffix_chars += 1
        return suffix_chars

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
        self._mapped_prompt_start = next(
            offset[0] for offset in self._offsets if offset is not None
        )
        self._pretoken_offsets_cache = [
            offset
            for offset in self._pretoken_offsets(normalized_prompt)
            if offset[1] > self._mapped_prompt_start
        ]

    def _reset_from_authoritative_prefix(
        self,
        *,
        rendered_prompt: str,
        template_prefix_prompt: str,
        authoritative_prefix_token_ids: list[int],
    ) -> IncrementalTokenizerPrefixSeed:
        if not authoritative_prefix_token_ids:
            raise IncrementalTokenizerPrefixSeedError(
                "an authoritative token prefix is required"
            )
        eos_token = getattr(self._tokenizer, "eos_token", None)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if not isinstance(eos_token, str) or eos_token_id is None:
            raise IncrementalTokenizerPrefixSeedError(
                "tokenizer EOS text and token ID are required"
            )

        normalized_prompt = self._normalize(rendered_prompt)
        normalized_template_prefix = self._normalize(template_prefix_prompt)
        normalized_eos_token = self._normalize(eos_token)
        template_cut_start = normalized_template_prefix.rfind(normalized_eos_token)
        if template_cut_start < 0:
            raise IncrementalTokenizerPrefixSeedError(
                "the rendered assistant prefix has no EOS boundary"
            )
        template_cut_end = template_cut_start + len(normalized_eos_token)
        if (
            normalized_prompt[:template_cut_end]
            != normalized_template_prefix[:template_cut_end]
        ):
            raise IncrementalTokenizerPrefixSeedError(
                "the full prompt changed before the assistant EOS boundary"
            )

        encoded_suffix = normalized_prompt[template_cut_start:]
        suffix_token_ids, suffix_offsets = self._encode_with_offsets(encoded_suffix)
        if (
            not suffix_token_ids
            or suffix_token_ids[0] != eos_token_id
            or suffix_offsets[0] != (0, len(normalized_eos_token))
        ):
            raise IncrementalTokenizerPrefixSeedError(
                "the assistant EOS is not an isolated tokenizer boundary"
            )

        model_cut_end = len(authoritative_prefix_token_ids)
        if authoritative_prefix_token_ids[-1] == eos_token_id:
            model_cut_end -= 1
        reused_prefix_token_ids = authoritative_prefix_token_ids[:model_cut_end]
        adjusted_suffix_offsets = [
            (start + template_cut_start, end + template_cut_start)
            for start, end in suffix_offsets
        ]
        self._prompt = normalized_prompt
        self._token_ids = reused_prefix_token_ids + suffix_token_ids
        self._offsets = [None] * len(reused_prefix_token_ids) + adjusted_suffix_offsets
        self._mapped_prompt_start = template_cut_start
        self._pretoken_offsets_cache = [
            (start + template_cut_start, end + template_cut_start)
            for start, end in self._pretoken_offsets(encoded_suffix)
        ]
        return IncrementalTokenizerPrefixSeed(
            reused_tokens=len(reused_prefix_token_ids),
            encoded_chars=len(encoded_suffix),
            encoded_tokens=len(suffix_token_ids),
        )

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
        repair_start = self._mapped_prompt_start
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
        if common_prefix_chars <= self._mapped_prompt_start:
            raise IncrementalTokenizerError(
                "the rendered prompt changed before the mapped suffix"
            )
        repair_start = self._pretoken_repair_start(common_prefix_chars)
        repair_start = max(
            self._mapped_prompt_start,
            min(
                repair_start,
                self._added_token_repair_start(common_prefix_chars),
            ),
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

    def stable_prefix_token_ids_before_alternate_suffix(
        self,
        alternate_rendered_prompt: str,
    ) -> list[int]:
        """Return tokens proven stable before a shared rendered suffix.

        ``alternate_rendered_prompt`` must render the same chat request with
        only the final tool output replaced. The common suffix identifies the
        chat-template text after that output. The returned prefix ends before
        the pre-tokenizer segment that a future tool-output append can revise.
        """
        normalized_alternate = self._normalize(alternate_rendered_prompt)
        if normalized_alternate == self._prompt:
            raise IncrementalTokenizerStablePrefixError(
                "the alternate rendering contains no nonempty tool output"
            )
        common_prefix_chars = self._common_prefix_chars(
            self._prompt,
            normalized_alternate,
        )
        common_suffix_chars = self._common_suffix_chars(
            self._prompt,
            normalized_alternate,
            common_prefix_chars=common_prefix_chars,
        )
        if not common_suffix_chars:
            raise IncrementalTokenizerStablePrefixError(
                "the alternate rendering has no shared template suffix"
            )

        stable_output_end = len(self._prompt) - common_suffix_chars
        if stable_output_end <= common_prefix_chars:
            raise IncrementalTokenizerStablePrefixError(
                "the alternate rendering contains no nonempty tool output"
            )
        try:
            repair_start = self._repair_start(stable_output_end)
            retained_tokens = self._retained_token_count(repair_start)
        except IncrementalTokenizerError as error:
            raise IncrementalTokenizerStablePrefixError(str(error)) from error
        if not retained_tokens:
            raise IncrementalTokenizerStablePrefixError(
                "the stable tool-output boundary retains no tokens"
            )
        return list(self._token_ids[:retained_tokens])

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
        self._aborted_sessions: OrderedDict[str, float] = OrderedDict()

    def _expire_sessions(self) -> None:
        now = self._clock()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
        while self._aborted_sessions:
            session_id, expires_at = next(iter(self._aborted_sessions.items()))
            if expires_at > now:
                break
            self._aborted_sessions.pop(session_id)

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
        if session_id in self._aborted_sessions:
            raise IncrementalTokenizerError(
                f"incremental tokenizer session was aborted: {session_id}"
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

    def start_from_authoritative_prefix(
        self,
        *,
        session_id: str,
        sequence_no: int,
        rendered_prompt: str,
        template_prefix_prompt: str,
        authoritative_prefix_token_ids: list[int],
    ) -> tuple[IncrementalTokenizationStep, list[int]]:
        """Start from prior model tokens plus an exactly tokenized suffix."""
        self._expire_sessions()
        if sequence_no != 0:
            raise IncrementalTokenizerError(
                f"first sequence must be 0, got {sequence_no}"
            )
        if session_id in self._aborted_sessions:
            raise IncrementalTokenizerError(
                f"incremental tokenizer session was aborted: {session_id}"
            )

        encoder, seed = ExactIncrementalTokenizer.from_authoritative_prefix(
            tokenizer=self._tokenizer,
            rendered_prompt=rendered_prompt,
            template_prefix_prompt=template_prefix_prompt,
            authoritative_prefix_token_ids=authoritative_prefix_token_ids,
            backend_tokenizer=self._backend_tokenizer,
            added_tokens=self._added_tokens,
        )
        authoritative_token_ids = encoder.token_ids
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
                return existing_session.last_result, authoritative_token_ids
            raise IncrementalTokenizerError(
                f"incremental tokenizer session already exists: {session_id}"
            )
        if len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)
        session = _IncrementalTokenizerSession(
            encoder=encoder,
            sequence_no=sequence_no,
            expires_at=self._clock() + self._session_ttl_seconds,
            last_prompt=rendered_prompt,
        )
        result = IncrementalTokenizationStep(
            sequence_no=sequence_no,
            token_count=len(authoritative_token_ids),
            encoded_chars=seed.encoded_chars,
            encoded_tokens=seed.encoded_tokens,
            reused_tokens=seed.reused_tokens,
            rollback_tokens=0,
            checkpoint_count=0,
            checkpoint_tokens=0,
            checkpoint_mismatches=0,
            incremental_valid=True,
        )
        session.last_result = result
        self._sessions[session_id] = session
        return result, authoritative_token_ids

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

    def finalize_current(self, session_id: str) -> IncrementalTokenizationStep:
        """Finalize an exact prefix-seeded session without another snapshot."""
        session = self._get_session(session_id)
        try:
            if not session.incremental_valid or session.encoder is None:
                raise IncrementalTokenizerError(
                    "finalization requires a valid incremental session"
                )
            if session.last_result is None:
                raise IncrementalTokenizerError(
                    "finalization requires a completed tokenizer step"
                )
            return replace(
                session.last_result,
                tokens=list(session.encoder.token_ids),
            )
        finally:
            self._sessions.pop(session_id, None)

    def current_token_ids(self, session_id: str) -> list[int]:
        """Return the exact tokens for an active, valid session."""
        session = self._get_session(session_id)
        if not session.incremental_valid or session.encoder is None:
            raise IncrementalTokenizerError(
                "current tokens require a valid incremental session"
            )
        return session.encoder.token_ids

    def stable_prefix_token_ids_before_alternate_suffix(
        self,
        *,
        session_id: str,
        alternate_rendered_prompt: str,
    ) -> list[int]:
        """Return a proven stable prefix for an active exact session."""
        session = self._get_session(session_id)
        if not session.incremental_valid or session.encoder is None:
            raise IncrementalTokenizerStablePrefixError(
                "stable prefix requires a valid incremental session"
            )
        return session.encoder.stable_prefix_token_ids_before_alternate_suffix(
            alternate_rendered_prompt
        )

    def abort(self, session_id: str) -> bool:
        """Cancel present or late-arriving work for one session ID."""
        self._expire_sessions()
        existed = self._sessions.pop(session_id, None) is not None
        self._aborted_sessions.pop(session_id, None)
        self._aborted_sessions[session_id] = self._clock() + self._session_ttl_seconds
        while len(self._aborted_sessions) > self._max_sessions:
            self._aborted_sessions.popitem(last=False)
        return existed
