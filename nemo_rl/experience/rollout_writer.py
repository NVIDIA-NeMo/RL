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
"""Contracts and cursor state for direct NeMo Gym rollout writes."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import ray
import torch

from nemo_rl.data_plane.interfaces import DataPlaneClient
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

ROLLOUT_CONTEXT_VERSION = 1
HASH_VERSION = 1
EMPTY_PREFIX_HASH = hashlib.sha256(b"nemo-rl-prefix-v1").hexdigest()
TOKEN_ID_PAYLOAD_KEYS = frozenset(
    {
        "prompt_token_ids",
        "generation_token_ids",
        "token_ids",
        # vLLM's /tokenize response uses `tokens`, while chat responses use
        # the fields above. Include both so the legacy side channel is
        # represented in the same token-payload accounting.
        "tokens",
    }
)
LOGPROB_PAYLOAD_KEYS = frozenset(
    {
        "generation_log_probs",
        "generation_logprobs",
        "logprobs",
        "top_logprobs",
    }
)


class RolloutContextError(ValueError):
    """Raised when a signed rollout context is invalid."""


class CursorError(RuntimeError):
    """Base class for cursor state-machine failures."""


class CursorConflictError(CursorError):
    """Raised for concurrent or stale reservations."""


class DuplicateRequestError(CursorError):
    """Raised when a completed inference request is retried."""


class CursorFailedError(CursorError):
    """Raised when a sample has already entered a terminal failed state."""


def _strip_payload_keys(value: Any, excluded: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_payload_keys(item, excluded)
            for key, item in value.items()
            if key not in excluded
        }
    if isinstance(value, list):
        return [_strip_payload_keys(item, excluded) for item in value]
    return value


def encoded_response_payload_sizes(payload: dict[str, Any]) -> dict[str, int]:
    """Measure an OpenAI JSON response and its token/logprob-free variants.

    The encoding matches Starlette's compact ``JSONResponse`` settings, so the
    total is the exact uncompressed HTTP response-body length.
    """

    def encoded_size(value: Any) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    without_token_ids = _strip_payload_keys(payload, TOKEN_ID_PAYLOAD_KEYS)
    without_logprobs = _strip_payload_keys(payload, LOGPROB_PAYLOAD_KEYS)
    without_token_ids_or_logprobs = _strip_payload_keys(
        without_token_ids, LOGPROB_PAYLOAD_KEYS
    )
    return {
        "encoded_response_bytes": encoded_size(payload),
        "encoded_response_bytes_without_token_ids": encoded_size(without_token_ids),
        "encoded_response_bytes_without_logprobs": encoded_size(without_logprobs),
        "encoded_response_base_bytes": encoded_size(without_token_ids_or_logprobs),
    }


def _encode_bytes(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _encode_text(value: str) -> bytes:
    return _encode_bytes(value.encode("utf-8"))


def encode_token_ids(token_ids: list[int]) -> bytes:
    """Return a stable, length-delimited big-endian token encoding."""
    encoded = bytearray(struct.pack(">BQ", HASH_VERSION, len(token_ids)))
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError(f"token IDs must be non-negative, got {token_id}")
        encoded.extend(struct.pack(">Q", token_id))
    return bytes(encoded)


def hash_token_ids(token_ids: list[int]) -> str:
    """Hash an exact token sequence using the rollout hash encoding."""
    if not token_ids:
        return EMPTY_PREFIX_HASH
    return hashlib.sha256(b"nemo-rl-prefix" + encode_token_ids(token_ids)).hexdigest()


def derive_request_nonce(sample_id: str, prompt_token_ids: list[int]) -> str:
    """Derive the idempotency key for one inference request."""
    payload = (
        struct.pack(">B", HASH_VERSION)
        + _encode_text(sample_id)
        + _encode_bytes(encode_token_ids(prompt_token_ids))
    )
    return hashlib.sha256(b"nemo-rl-request" + payload).hexdigest()


def mint_rollout_id() -> str:
    """Mint the canonical identity for one generation attempt.

    One ``rollout_id`` names one Gym execution, its staging rows, and its
    canonical train row. It is opaque, URL-safe, carries 128 random bits, and
    stays valid under Gym's rollout-id charset so the same string can ride a
    ``/ng-rollout/{rid}/v1`` base URL. ``group_id`` is deliberately not
    embedded: sibling grouping stays a driver-side mapping and never travels
    through model requests or storage keys.
    """
    return secrets.token_hex(16)


def build_staging_delta(
    *,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    generated_logprobs: list[float],
    prev_len: int,
) -> tuple[list[int], list[float], list[float]]:
    """Slice one full request/response into the next cursor delta."""
    if prev_len < 0 or prev_len > len(prompt_token_ids):
        raise ValueError(
            f"prev_len={prev_len} is outside prompt length {len(prompt_token_ids)}"
        )
    if len(generated_token_ids) != len(generated_logprobs):
        raise ValueError(
            "generated token and log-probability lengths differ: "
            f"{len(generated_token_ids)} != {len(generated_logprobs)}"
        )
    prompt_delta = prompt_token_ids[prev_len:]
    token_ids_delta = prompt_delta + generated_token_ids
    token_mask_delta = [0.0] * len(prompt_delta) + [1.0] * len(generated_token_ids)
    logprobs_delta = [0.0] * len(prompt_delta) + generated_logprobs
    if not token_ids_delta:
        raise ValueError("staging delta must contain at least one token")
    return token_ids_delta, token_mask_delta, logprobs_delta


def extract_generation_token_info(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]]:
    """Read token IDs/logprobs from either supported vLLM response shape."""
    message = choice.get("message") or {}
    if "generation_token_ids" in message and "generation_log_probs" in message:
        raw_ids = message["generation_token_ids"]
        raw_logprobs = message["generation_log_probs"]
    else:
        content_logprobs = (choice.get("logprobs") or {}).get("content")
        if content_logprobs is None:
            raise ValueError(
                "vLLM response contained neither message token fields nor "
                "choice.logprobs.content"
            )
        raw_ids = [item["token"] for item in content_logprobs]
        raw_logprobs = [item["logprob"] for item in content_logprobs]
    token_ids = [int(str(token_id).removeprefix("token_id:")) for token_id in raw_ids]
    logprobs = [float(value) for value in raw_logprobs]
    if len(token_ids) != len(logprobs):
        raise ValueError(
            "generated token and log-probability lengths differ: "
            f"{len(token_ids)} != {len(logprobs)}"
        )
    return token_ids, logprobs


@dataclass(frozen=True)
class RolloutContext:
    """Signed identity forwarded opaquely through NeMo Gym.

    ``sample_id`` is a migration alias for the canonical ``rollout_id``: the
    value carried here is always the id minted by :func:`mint_rollout_id`, and
    any request that carries both this context and a bare ``nemo_rl_rollout_id``
    field is accepted only when the two agree.
    """

    sample_id: str
    group_id: str
    weight_version: int
    nonce: str
    issued_at: float
    expires_at: float
    signature: str
    version: int = ROLLOUT_CONTEXT_VERSION

    def unsigned_bytes(self) -> bytes:
        return b"".join(
            (
                struct.pack(">B", self.version),
                _encode_text(self.sample_id),
                _encode_text(self.group_id),
                struct.pack(">q", self.weight_version),
                _encode_text(self.nonce),
                struct.pack(">dd", self.issued_at, self.expires_at),
            )
        )

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "version": self.version,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "weight_version": self.weight_version,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RolloutContext":
        try:
            return cls(
                version=int(payload["version"]),
                sample_id=str(payload["sample_id"]),
                group_id=str(payload["group_id"]),
                weight_version=int(payload["weight_version"]),
                nonce=str(payload["nonce"]),
                issued_at=float(payload["issued_at"]),
                expires_at=float(payload["expires_at"]),
                signature=str(payload["signature"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RolloutContextError("malformed rollout context") from error


def mint_rollout_context(
    *,
    sample_id: str,
    group_id: str,
    weight_version: int,
    secret: bytes,
    ttl_s: float,
    now: float | None = None,
) -> RolloutContext:
    """Create and HMAC-sign a short-lived rollout context."""
    issued_at = time.time() if now is None else now
    unsigned = RolloutContext(
        sample_id=sample_id,
        group_id=group_id,
        weight_version=weight_version,
        nonce=secrets.token_hex(16),
        issued_at=issued_at,
        expires_at=issued_at + ttl_s,
        signature="",
    )
    signature = hmac.new(secret, unsigned.unsigned_bytes(), hashlib.sha256).hexdigest()
    return RolloutContext(
        sample_id=unsigned.sample_id,
        group_id=unsigned.group_id,
        weight_version=unsigned.weight_version,
        nonce=unsigned.nonce,
        issued_at=unsigned.issued_at,
        expires_at=unsigned.expires_at,
        signature=signature,
        version=unsigned.version,
    )


def validate_rollout_context(
    context: RolloutContext,
    *,
    secret: bytes,
    now: float | None = None,
) -> None:
    """Validate version, lifetime, and signature of a rollout context."""
    current_time = time.time() if now is None else now
    if context.version != ROLLOUT_CONTEXT_VERSION:
        raise RolloutContextError(
            f"unsupported rollout context version: {context.version}"
        )
    if context.expires_at <= context.issued_at:
        raise RolloutContextError("rollout context expiry must follow issue time")
    if current_time > context.expires_at:
        raise RolloutContextError("rollout context has expired")
    expected = hmac.new(secret, context.unsigned_bytes(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(context.signature, expected):
        raise RolloutContextError("rollout context signature is invalid")


TurnState = Literal["reserved", "committed", "failed"]


@dataclass
class TurnRecord:
    """Registry record for one reserved inference request."""

    turn: int
    request_nonce: str
    lease: str
    prev_len: int
    prev_hash: str
    state: TurnState
    updated_at: float
    staging_key: str | None = None
    new_len: int | None = None
    new_hash: str | None = None
    failure_reason: str | None = None
    group_id: str | None = None
    weight_version: int | None = None


@dataclass(frozen=True)
class TurnReservation:
    """Reservation returned to a vLLM worker."""

    turn: int
    request_nonce: str
    lease: str
    prev_len: int
    prev_hash: str


@dataclass(frozen=True)
class RolloutIdentity:
    """The verified write identity for one inference request.

    Two provenances produce it: the native path validates a signed
    ``RolloutContext`` (``rollout_id`` = the context's ``sample_id`` alias,
    ``group_id`` from the context, ``call_id`` absent), and the gateway path
    accepts ``nemo_rl_rollout_id``/``nemo_rl_call_id`` forwarded by a trusted
    Gym gate (``group_id`` stays empty — sibling grouping never travels
    through model requests; ``weight_version`` is the worker's configured
    step version).
    """

    rollout_id: str
    group_id: str
    weight_version: int
    call_id: str | None = None


@dataclass(frozen=True)
class RolloutRequestState:
    """Request-scoped state shared by vLLM preprocessing and serialization."""

    identity: RolloutIdentity
    reservation: TurnReservation
    prompt_token_ids: list[int]


def strip_direct_response_logprobs(response: dict[str, Any]) -> dict[str, Any]:
    """Remove sampling log probabilities while retaining token IDs."""
    for choice in response.get("choices", []):
        choice.pop("logprobs", None)
        message = choice.get("message", {})
        message.pop("generation_log_probs", None)
        message.pop("generation_logprobs", None)
    return response


@dataclass
class SampleCursor:
    """All registry state for an active rollout sample."""

    next_turn: int
    committed_length: int
    prefix_hash: str
    updated_at: float
    turns: dict[int, TurnRecord] = field(default_factory=dict)
    nonce_to_turn: dict[str, int] = field(default_factory=dict)
    active_turn: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class FinalizationManifest:
    """Immutable cursor view consumed by the trusted finalizer."""

    sample_id: str
    committed_length: int
    prefix_hash: str
    turns: tuple[TurnRecord, ...]
    failure_reason: str | None


@dataclass(frozen=True)
class FinalizedRolloutBatch:
    """Canonical direct-writer candidate and its cleanup metadata."""

    bulk_batch: BatchedDataDict
    driver_carry: BatchedDataDict
    staging_keys: tuple[str, ...]
    manifest_rows: tuple[dict[str, Any], ...]


class RolloutCursorStateMachine:
    """Single-threaded rollout cursor implementation independent of Ray."""

    def __init__(self, *, lease_ttl_s: float, cursor_ttl_s: float) -> None:
        if lease_ttl_s <= 0 or cursor_ttl_s <= 0:
            raise ValueError("cursor and lease TTLs must be positive")
        self.lease_ttl_s = lease_ttl_s
        self.cursor_ttl_s = cursor_ttl_s
        self.samples: dict[str, SampleCursor] = {}

    def reserve_turn(
        self, sample_id: str, request_nonce: str, *, now: float | None = None
    ) -> TurnReservation:
        timestamp = time.time() if now is None else now
        cursor = self.samples.get(sample_id)
        if cursor is None:
            cursor = SampleCursor(
                next_turn=0,
                committed_length=0,
                prefix_hash=EMPTY_PREFIX_HASH,
                updated_at=timestamp,
            )
            self.samples[sample_id] = cursor
        if cursor.failure_reason is not None:
            raise CursorFailedError(cursor.failure_reason)

        existing_turn = cursor.nonce_to_turn.get(request_nonce)
        if existing_turn is not None:
            record = cursor.turns[existing_turn]
            if record.state == "committed":
                cursor.failure_reason = "duplicate_request"
                cursor.updated_at = timestamp
                raise DuplicateRequestError(
                    f"completed request retried for sample {sample_id}"
                )
            if record.state == "failed":
                raise CursorFailedError(record.failure_reason or "turn_failed")
            if timestamp - record.updated_at > self.lease_ttl_s:
                record.lease = uuid.uuid4().hex
                record.updated_at = timestamp
            cursor.updated_at = timestamp
            return self._reservation(record)

        if cursor.active_turn is not None:
            active = cursor.turns[cursor.active_turn]
            cursor.failure_reason = "concurrent_request"
            cursor.updated_at = timestamp
            active.state = "failed"
            active.failure_reason = cursor.failure_reason
            raise CursorConflictError(
                f"sample {sample_id} already has active turn {active.turn}"
            )

        turn = cursor.next_turn
        record = TurnRecord(
            turn=turn,
            request_nonce=request_nonce,
            lease=uuid.uuid4().hex,
            prev_len=cursor.committed_length,
            prev_hash=cursor.prefix_hash,
            state="reserved",
            updated_at=timestamp,
        )
        cursor.turns[turn] = record
        cursor.nonce_to_turn[request_nonce] = turn
        cursor.active_turn = turn
        cursor.updated_at = timestamp
        return self._reservation(record)

    def commit_turn(
        self,
        sample_id: str,
        lease: str,
        *,
        staging_key: str,
        new_len: int,
        new_hash: str,
        group_id: str,
        weight_version: int,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        cursor, record = self._active_record(sample_id, lease)
        if new_len <= record.prev_len:
            self.fail_turn(
                sample_id,
                lease,
                reason="non_growing_cursor",
                now=timestamp,
            )
            raise CursorConflictError(
                f"new length {new_len} must exceed {record.prev_len}"
            )
        record.staging_key = staging_key
        record.new_len = new_len
        record.new_hash = new_hash
        record.group_id = group_id
        record.weight_version = weight_version
        record.state = "committed"
        record.updated_at = timestamp
        cursor.committed_length = new_len
        cursor.prefix_hash = new_hash
        cursor.next_turn += 1
        cursor.active_turn = None
        cursor.updated_at = timestamp

    def fail_turn(
        self,
        sample_id: str,
        lease: str,
        *,
        reason: str,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        cursor, record = self._active_record(sample_id, lease)
        record.state = "failed"
        record.failure_reason = reason
        record.updated_at = timestamp
        cursor.failure_reason = reason
        cursor.active_turn = None
        cursor.updated_at = timestamp

    def fail_sample(
        self, sample_id: str, *, reason: str, now: float | None = None
    ) -> None:
        timestamp = time.time() if now is None else now
        cursor = self.samples.get(sample_id)
        if cursor is None:
            cursor = SampleCursor(
                next_turn=0,
                committed_length=0,
                prefix_hash=EMPTY_PREFIX_HASH,
                updated_at=timestamp,
            )
            self.samples[sample_id] = cursor
        cursor.failure_reason = reason
        cursor.updated_at = timestamp

    def get_finalization_manifest(self, sample_id: str) -> FinalizationManifest:
        cursor = self.samples.get(sample_id)
        if cursor is None:
            raise KeyError(f"unknown rollout sample {sample_id}")
        if cursor.active_turn is not None:
            raise CursorConflictError(
                f"sample {sample_id} still has an active reservation"
            )
        return FinalizationManifest(
            sample_id=sample_id,
            committed_length=cursor.committed_length,
            prefix_hash=cursor.prefix_hash,
            turns=tuple(cursor.turns[index] for index in sorted(cursor.turns)),
            failure_reason=cursor.failure_reason,
        )

    def clear_sample(self, sample_id: str) -> None:
        self.samples.pop(sample_id, None)

    def expire_stale(self, *, now: float | None = None) -> list[str]:
        timestamp = time.time() if now is None else now
        expired = [
            sample_id
            for sample_id, cursor in self.samples.items()
            if timestamp - cursor.updated_at > self.cursor_ttl_s
        ]
        for sample_id in expired:
            del self.samples[sample_id]
        return expired

    def _active_record(
        self, sample_id: str, lease: str
    ) -> tuple[SampleCursor, TurnRecord]:
        cursor = self.samples.get(sample_id)
        if cursor is None or cursor.active_turn is None:
            raise CursorConflictError(f"sample {sample_id} has no active reservation")
        record = cursor.turns[cursor.active_turn]
        if not hmac.compare_digest(record.lease, lease):
            raise CursorConflictError(f"stale lease for sample {sample_id}")
        return cursor, record

    @staticmethod
    def _reservation(record: TurnRecord) -> TurnReservation:
        return TurnReservation(
            turn=record.turn,
            request_nonce=record.request_nonce,
            lease=record.lease,
            prev_len=record.prev_len,
            prev_hash=record.prev_hash,
        )


def assemble_staged_batch(
    *,
    dp_client: DataPlaneClient,
    cursor,
    staging_partition: str,
    sample_ids: list[str],
    group_ids: list[str],
    weight_version: int,
    legacy_bulk: BatchedDataDict,
    legacy_carry: BatchedDataDict,
    pad_token_id: int,
    finalize_timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
    perf_metrics: dict[str, float] | None = None,
) -> FinalizedRolloutBatch:
    """Verify staging rows and assemble a canonical padded batch.

    The legacy payload supplies trusted reward/carry data in shadow mode and
    during the initial direct cutover. Token-aligned training tensors always
    come from verified staging rows.
    """
    if finalize_timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError("finalization timeout and poll interval must be positive")
    batch_size = int(legacy_bulk["input_ids"].shape[0])
    if len(sample_ids) != batch_size or len(group_ids) != batch_size:
        raise ValueError(
            "sample_ids and group_ids must match the legacy batch size; "
            f"got {len(sample_ids)}, {len(group_ids)}, and {batch_size}"
        )
    if int(legacy_carry["loss_multiplier"].shape[0]) != batch_size:
        raise ValueError(
            "legacy carry must match the legacy bulk batch size; "
            f"got {legacy_carry['loss_multiplier'].shape[0]} and {batch_size}"
        )
    deadline = time.monotonic() + finalize_timeout_s
    target_width = int(legacy_bulk["input_ids"].shape[1])
    rows: list[dict[str, torch.Tensor | int | float]] = []
    manifest_rows: list[dict[str, Any]] = []
    staging_keys: list[str] = []

    def record(name: str, elapsed_s: float) -> None:
        if perf_metrics is not None:
            perf_metrics[name] = perf_metrics.get(name, 0.0) + elapsed_s

    for index, (sample_id, group_id) in enumerate(zip(sample_ids, group_ids)):
        manifest = None
        rejection_reason = None
        sample_staging_keys: list[str] = []
        verified_turns = 0
        tokens: list[int] = []
        token_mask: list[float] = []
        generation_logprobs: list[float] = []
        first_response_length = 0
        terminal_hash = EMPTY_PREFIX_HASH
        try:
            operation_started = time.perf_counter()
            manifest = ray.get(cursor.get_finalization_manifest.remote(sample_id))
            record("manifest_s", time.perf_counter() - operation_started)
            sample_staging_keys = [
                turn.staging_key
                for turn in manifest.turns
                if turn.staging_key is not None
            ]
            if manifest.failure_reason is not None:
                raise CursorFailedError(manifest.failure_reason)
            if not manifest.turns:
                raise CursorError("missing_turns")
            expected_prev_len = 0
            expected_prev_hash = EMPTY_PREFIX_HASH
            for expected_turn, turn in enumerate(manifest.turns):
                if turn.turn != expected_turn or turn.state != "committed":
                    raise CursorError("non_contiguous_turns")
                # Gateway-identified turns commit group_id="" — sibling
                # grouping never travels through model requests; identity is
                # instead reconciled against the sealed call manifest.
                if turn.group_id not in ("", group_id):
                    raise CursorError("identity_or_weight_version_mismatch")
                if turn.weight_version != weight_version:
                    raise CursorError("identity_or_weight_version_mismatch")
                if (
                    turn.prev_len != expected_prev_len
                    or turn.prev_hash != expected_prev_hash
                    or turn.staging_key is None
                    or turn.new_len is None
                    or turn.new_hash is None
                ):
                    raise CursorError("cursor_chain_mismatch")
                while True:
                    try:
                        operation_started = time.perf_counter()
                        td = dp_client.get_samples(
                            sample_ids=[turn.staging_key],
                            partition_id=staging_partition,
                            select_fields=[
                                "token_ids_delta",
                                "token_mask_delta",
                                "generation_logprobs_delta",
                            ],
                        )
                        record(
                            "staging_read_s", time.perf_counter() - operation_started
                        )
                        break
                    except (KeyError, RuntimeError, TimeoutError, ValueError) as error:
                        record(
                            "staging_read_s", time.perf_counter() - operation_started
                        )
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise CursorError(
                                f"missing_staging_row:{turn.staging_key}"
                            ) from error
                        sleep_s = min(poll_interval_s, remaining)
                        time.sleep(sleep_s)
                        record("poll_wait_s", sleep_s)
                verification_started = time.perf_counter()
                delta_ids = td["token_ids_delta"].reshape(-1).tolist()
                delta_mask = td["token_mask_delta"].reshape(-1).tolist()
                delta_logprobs = td["generation_logprobs_delta"].reshape(-1).tolist()
                if not delta_ids or not (
                    len(delta_ids) == len(delta_mask) == len(delta_logprobs)
                ):
                    raise CursorError("invalid_delta_shape")
                if any(value not in (0, 0.0, 1, 1.0) for value in delta_mask):
                    raise CursorError("invalid_token_mask")
                if any(not math.isfinite(value) for value in delta_logprobs):
                    raise CursorError("non_finite_generation_logprob")
                tokens.extend(int(value) for value in delta_ids)
                token_mask.extend(float(value) for value in delta_mask)
                generation_logprobs.extend(float(value) for value in delta_logprobs)
                if expected_turn == 0:
                    first_response_length = int(sum(delta_mask))
                if (
                    len(tokens) != turn.new_len
                    or hash_token_ids(tokens) != turn.new_hash
                ):
                    raise CursorError("staging_hash_mismatch")
                expected_prev_len = turn.new_len
                expected_prev_hash = turn.new_hash
                verified_turns += 1
                record("verification_s", time.perf_counter() - verification_started)
            if manifest.committed_length != len(tokens):
                raise CursorError("terminal_length_mismatch")
            terminal_hash = hash_token_ids(tokens)
            if terminal_hash != manifest.prefix_hash:
                raise CursorError("terminal_hash_mismatch")
        except (
            CursorError,
            KeyError,
            RuntimeError,
            ValueError,
            ray.exceptions.RayError,
        ) as error:
            rejection_reason = str(error) or type(error).__name__
            tokens = [pad_token_id]
            token_mask = [0.0]
            generation_logprobs = [0.0]
            first_response_length = 0

        valid = 0.0 if rejection_reason is not None else 1.0
        rows.append(
            {
                "input_ids": torch.tensor(tokens, dtype=torch.int64),
                "input_length": len(tokens),
                "token_mask": torch.tensor(token_mask, dtype=torch.float32),
                "generation_logprobs": torch.tensor(
                    generation_logprobs, dtype=torch.float32
                ),
                "sample_mask": float(legacy_carry["loss_multiplier"][index]) * valid,
                "trajectory_valid_mask": valid,
                "response_token_length": first_response_length,
            }
        )
        staging_keys.extend(sample_staging_keys)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "status": "rejected" if rejection_reason else "finalized",
                "rejection_reason": rejection_reason,
                "expected_turns": len(manifest.turns) if manifest is not None else 0,
                "written_turns": verified_turns,
                "length": len(tokens) if rejection_reason is None else 0,
                "terminal_hash": terminal_hash,
                "weight_version": weight_version,
            }
        )

    assembly_started = time.perf_counter()
    if any(int(row["input_length"]) > target_width for row in rows):
        raise ValueError("direct candidate exceeds legacy padded sequence width")
    input_ids = torch.full((len(rows), target_width), pad_token_id, dtype=torch.int64)
    masks = torch.zeros((len(rows), target_width), dtype=torch.float32)
    logprobs = torch.zeros((len(rows), target_width), dtype=torch.float32)
    for index, row in enumerate(rows):
        length = int(row["input_length"])
        input_ids[index, :length] = row["input_ids"]
        masks[index, :length] = row["token_mask"]
        logprobs[index, :length] = row["generation_logprobs"]

    bulk = BatchedDataDict(dict(legacy_bulk))
    bulk["input_ids"] = input_ids
    bulk["input_lengths"] = torch.tensor(
        [int(row["input_length"]) for row in rows], dtype=torch.int64
    )
    bulk["token_mask"] = masks
    bulk["generation_logprobs"] = logprobs
    bulk["sample_mask"] = torch.tensor(
        [float(row["sample_mask"]) for row in rows], dtype=torch.float32
    )

    carry = BatchedDataDict(dict(legacy_carry))
    validity = torch.tensor(
        [float(row["trajectory_valid_mask"]) for row in rows], dtype=torch.float32
    )
    carry["trajectory_valid_mask"] = validity
    carry["loss_multiplier"] = bulk["sample_mask"].clone()
    carry["input_lengths"] = bulk["input_lengths"].clone()
    carry["response_token_lengths"] = torch.tensor(
        [int(row["response_token_length"]) for row in rows], dtype=torch.int64
    )
    carry["total_reward"] = carry["total_reward"] * validity
    for field_name in tuple(carry.keys()):
        if field_name.startswith("reward/"):
            carry[field_name] = carry[field_name] * validity
    record("assembly_s", time.perf_counter() - assembly_started)

    return FinalizedRolloutBatch(
        bulk_batch=bulk,
        driver_carry=carry,
        staging_keys=tuple(staging_keys),
        manifest_rows=tuple(manifest_rows),
    )


def compare_shadow_candidate(
    direct: FinalizedRolloutBatch,
    *,
    legacy_bulk: BatchedDataDict,
    legacy_carry: BatchedDataDict,
) -> None:
    """Raise with a precise field name when shadow tensors diverge."""
    validity = direct.driver_carry["trajectory_valid_mask"].bool()
    for field_name in (
        "input_ids",
        "input_lengths",
        "generation_logprobs",
        "token_mask",
        "sample_mask",
    ):
        if not torch.equal(
            direct.bulk_batch[field_name][validity], legacy_bulk[field_name][validity]
        ):
            raise AssertionError(f"shadow rollout mismatch in {field_name}")
    for field_name in legacy_carry:
        if field_name == "trajectory_valid_mask":
            continue
        direct_value = direct.driver_carry[field_name]
        legacy_value = legacy_carry[field_name]
        if isinstance(legacy_value, torch.Tensor) and not torch.equal(
            direct_value[validity], legacy_value[validity]
        ):
            raise AssertionError(f"shadow driver_carry mismatch in {field_name}")

    rejected = ~validity
    if rejected.any():
        if not torch.equal(
            direct.bulk_batch["sample_mask"][rejected],
            torch.zeros_like(direct.bulk_batch["sample_mask"][rejected]),
        ):
            raise AssertionError("shadow rejected rows have nonzero sample_mask")
        if torch.count_nonzero(direct.bulk_batch["token_mask"][rejected]):
            raise AssertionError("shadow rejected rows have nonzero token_mask")
        if torch.count_nonzero(direct.driver_carry["total_reward"][rejected]):
            raise AssertionError("shadow rejected rows have nonzero reward")


def persist_rollout_manifest(rows: tuple[dict[str, Any], ...], *, log_dir: str) -> None:
    """Append finalization outcomes even when external loggers are disabled."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "rollout_writer_manifest.jsonl").open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def persist_rollout_perf_metrics(row: dict[str, Any], *, log_dir: str) -> None:
    """Append one locally auditable performance record per rollout step."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "rollout_perf_metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


@ray.remote  # pragma: no cover
class RolloutCursorRegistry:
    """Ray serialization wrapper for :class:`RolloutCursorStateMachine`."""

    def __init__(self, *, lease_ttl_s: float, cursor_ttl_s: float) -> None:
        self.state = RolloutCursorStateMachine(
            lease_ttl_s=lease_ttl_s, cursor_ttl_s=cursor_ttl_s
        )

    def reserve_turn(self, sample_id: str, request_nonce: str) -> TurnReservation:
        return self.state.reserve_turn(sample_id, request_nonce)

    def commit_turn(
        self,
        sample_id: str,
        lease: str,
        *,
        staging_key: str,
        new_len: int,
        new_hash: str,
        group_id: str,
        weight_version: int,
    ) -> None:
        self.state.commit_turn(
            sample_id,
            lease,
            staging_key=staging_key,
            new_len=new_len,
            new_hash=new_hash,
            group_id=group_id,
            weight_version=weight_version,
        )

    def fail_turn(self, sample_id: str, lease: str, *, reason: str) -> None:
        self.state.fail_turn(sample_id, lease, reason=reason)

    def fail_sample(self, sample_id: str, *, reason: str) -> None:
        self.state.fail_sample(sample_id, reason=reason)

    def get_finalization_manifest(self, sample_id: str) -> FinalizationManifest:
        return self.state.get_finalization_manifest(sample_id)

    def clear_sample(self, sample_id: str) -> None:
        self.state.clear_sample(sample_id)

    def expire_stale(self, now: float | None = None) -> list[str]:
        return self.state.expire_stale(now=now)


# ---------------------------------------------------------------------------
# Forest-mode cursor (black-box harness semantics)
# ---------------------------------------------------------------------------

ROLLOUT_NODE_SCHEMA_VERSION = 1


def compute_staging_digest(
    *,
    rollout_id: str,
    call_id: str,
    prev_len: int,
    token_ids_delta: list[int],
    token_mask_delta: list[float],
    logprobs_delta: list[float],
) -> str:
    """Digest one staged row: token ids, mask values, logprob bit patterns,
    and the identifying metadata. The finalizer recomputes it over the fetched
    row, so any storage-layer corruption or substitution is detected."""
    payload = bytearray(struct.pack(">B", HASH_VERSION))
    payload.extend(_encode_text(rollout_id))
    payload.extend(_encode_text(call_id))
    payload.extend(struct.pack(">Q", prev_len))
    payload.extend(_encode_bytes(encode_token_ids(token_ids_delta)))
    # Masks and logprobs are digested as float32 BIT PATTERNS — the staging
    # columns are float32, so quantizing here makes the worker's digest (over
    # pre-tensorized python floats) and the finalizer's recomputation (over
    # fetched storage values) byte-identical, while -0.0 vs 0.0 and NaN
    # payloads still cannot alias.
    for mask_value in token_mask_delta:
        payload.extend(struct.pack(">f", mask_value))
    for logprob in logprobs_delta:
        payload.extend(struct.pack(">f", logprob))
    return hashlib.sha256(b"nemo-rl-staging-digest" + bytes(payload)).hexdigest()


NodeState = Literal["reserved", "committed", "failed"]


@dataclass
class ForestNodeRecord:
    """Registry record for one admitted model call (one staged row)."""

    call_id: str
    parent_call_id: str | None
    lease: str
    prev_len: int
    prev_hash: str
    state: NodeState
    updated_at: float
    is_new_root: bool = False
    staging_key: str | None = None
    prompt_len: int | None = None
    gen_len: int | None = None
    new_len: int | None = None
    new_hash: str | None = None
    digest: str | None = None
    weight_version: int | None = None
    schema_version: int = ROLLOUT_NODE_SCHEMA_VERSION
    failure_reason: str | None = None


@dataclass(frozen=True)
class ForestCandidate:
    """One committed node's cumulative sequence, offered as a parent."""

    call_id: str
    length: int
    sequence_hash: str


@dataclass(frozen=True)
class ForestReservation:
    """Reservation returned to a vLLM worker in forest mode."""

    call_id: str
    parent_call_id: str | None
    lease: str
    prev_len: int
    prev_hash: str
    is_new_root: bool


@dataclass(frozen=True)
class ForestManifest:
    """Immutable forest view consumed by the trusted finalizer."""

    rollout_id: str
    nodes: tuple[ForestNodeRecord, ...]
    failure_reason: str | None


@dataclass
class _RolloutForest:
    updated_at: float
    nodes: dict[str, ForestNodeRecord] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    failure_reason: str | None = None


class ForestCursorStateMachine:
    """Cursor semantics for untrusted, possibly concurrent, possibly
    context-compacting loops (black-box harnesses).

    Differences from the linear machine, per the integration design:

    - Nodes are keyed by the gate-minted ``call_id``; prompt hashes are
      integrity metadata, never identity, and two admitted calls with
      identical prompts are two nodes.
    - Reservation is "longest committed prefix of this prompt": the worker
      fetches committed candidates, hashes its rendered prompt at each
      candidate length, and reserves against the longest match. The registry
      only validates the claimed parent; it never sees token ids.
    - A prompt extending no committed prefix reserves a ``new_root``
      full-prompt row (harness compaction / history rewrite) instead of
      failing the rollout; a second child of one prefix is a branch. The
      registry retains node multiplicity for the finalizer's assembler.
    - Concurrent reservations are legal; each node carries its own lease.
    """

    def __init__(self, *, lease_ttl_s: float, cursor_ttl_s: float) -> None:
        if lease_ttl_s <= 0 or cursor_ttl_s <= 0:
            raise ValueError("cursor and lease TTLs must be positive")
        self.lease_ttl_s = lease_ttl_s
        self.cursor_ttl_s = cursor_ttl_s
        self.rollouts: dict[str, _RolloutForest] = {}

    def _forest(self, rollout_id: str, timestamp: float) -> _RolloutForest:
        forest = self.rollouts.get(rollout_id)
        if forest is None:
            forest = _RolloutForest(updated_at=timestamp)
            self.rollouts[rollout_id] = forest
        return forest

    def get_candidates(
        self, rollout_id: str, *, now: float | None = None
    ) -> list[ForestCandidate]:
        """Committed cumulative sequences the next prompt may extend,
        longest first (the parent rule picks the first hash match)."""
        timestamp = time.time() if now is None else now
        forest = self._forest(rollout_id, timestamp)
        if forest.failure_reason is not None:
            raise CursorFailedError(forest.failure_reason)
        candidates = [
            ForestCandidate(
                call_id=node.call_id,
                length=node.new_len or 0,
                sequence_hash=node.new_hash or EMPTY_PREFIX_HASH,
            )
            for node in forest.nodes.values()
            if node.state == "committed"
        ]
        candidates.sort(key=lambda c: c.length, reverse=True)
        return candidates

    def reserve_call(
        self,
        rollout_id: str,
        call_id: str,
        *,
        parent_call_id: str | None,
        prev_len: int,
        prev_hash: str,
        now: float | None = None,
    ) -> ForestReservation:
        timestamp = time.time() if now is None else now
        forest = self._forest(rollout_id, timestamp)
        if forest.failure_reason is not None:
            raise CursorFailedError(forest.failure_reason)

        existing = forest.nodes.get(call_id)
        if existing is not None:
            # Idempotent internal retry: the gateway preserves call_id.
            if existing.state == "committed":
                raise DuplicateRequestError(
                    f"committed call {call_id} retried for rollout {rollout_id}"
                )
            if existing.state == "failed":
                raise CursorFailedError(existing.failure_reason or "call_failed")
            if timestamp - existing.updated_at > self.lease_ttl_s:
                existing.lease = uuid.uuid4().hex
            existing.updated_at = timestamp
            forest.updated_at = timestamp
            return self._reservation(existing)

        if parent_call_id is None:
            if prev_len != 0 or prev_hash != EMPTY_PREFIX_HASH:
                raise CursorConflictError(
                    f"rootless reservation for {call_id} must start at length 0"
                )
            is_new_root = any(
                node.state == "committed" for node in forest.nodes.values()
            )
        else:
            parent = forest.nodes.get(parent_call_id)
            if parent is None or parent.state != "committed":
                raise CursorConflictError(
                    f"parent {parent_call_id} of call {call_id} is not committed"
                )
            if parent.new_len != prev_len or parent.new_hash != prev_hash:
                raise CursorConflictError(
                    f"parent {parent_call_id} does not match claimed prefix "
                    f"(len {prev_len})"
                )
            is_new_root = False

        record = ForestNodeRecord(
            call_id=call_id,
            parent_call_id=parent_call_id,
            lease=uuid.uuid4().hex,
            prev_len=prev_len,
            prev_hash=prev_hash,
            state="reserved",
            updated_at=timestamp,
            is_new_root=is_new_root,
        )
        forest.nodes[call_id] = record
        forest.order.append(call_id)
        forest.updated_at = timestamp
        return self._reservation(record)

    def commit_call(
        self,
        rollout_id: str,
        call_id: str,
        lease: str,
        *,
        staging_key: str,
        new_len: int,
        new_hash: str,
        prompt_len: int,
        gen_len: int,
        digest: str,
        weight_version: int,
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else now
        record = self._leased_record(rollout_id, call_id, lease)
        if new_len <= record.prev_len:
            self.fail_call(rollout_id, call_id, lease, reason="non_growing_node")
            raise CursorConflictError(
                f"call {call_id} committed length {new_len} does not extend "
                f"prefix length {record.prev_len}"
            )
        record.state = "committed"
        record.staging_key = staging_key
        record.new_len = new_len
        record.new_hash = new_hash
        record.prompt_len = prompt_len
        record.gen_len = gen_len
        record.digest = digest
        record.weight_version = weight_version
        record.updated_at = timestamp
        self.rollouts[rollout_id].updated_at = timestamp

    def fail_call(
        self,
        rollout_id: str,
        call_id: str,
        lease: str,
        *,
        reason: str,
        now: float | None = None,
    ) -> None:
        """A failed call fails its node only; the rollout survives (the
        finalizer decides validity from the manifest reconciliation)."""
        timestamp = time.time() if now is None else now
        record = self._leased_record(rollout_id, call_id, lease)
        record.state = "failed"
        record.failure_reason = reason
        record.updated_at = timestamp
        self.rollouts[rollout_id].updated_at = timestamp

    def fail_rollout(
        self, rollout_id: str, *, reason: str, now: float | None = None
    ) -> None:
        timestamp = time.time() if now is None else now
        forest = self._forest(rollout_id, timestamp)
        forest.failure_reason = reason
        forest.updated_at = timestamp

    def get_forest_manifest(self, rollout_id: str) -> ForestManifest:
        forest = self.rollouts.get(rollout_id)
        if forest is None:
            raise KeyError(f"unknown rollout {rollout_id}")
        return ForestManifest(
            rollout_id=rollout_id,
            nodes=tuple(forest.nodes[call_id] for call_id in forest.order),
            failure_reason=forest.failure_reason,
        )

    def clear_rollout(self, rollout_id: str) -> None:
        self.rollouts.pop(rollout_id, None)

    def expire_stale(self, now: float | None = None) -> list[str]:
        timestamp = time.time() if now is None else now
        stale = [
            rollout_id
            for rollout_id, forest in self.rollouts.items()
            if timestamp - forest.updated_at > self.cursor_ttl_s
        ]
        for rollout_id in stale:
            del self.rollouts[rollout_id]
        return stale

    def _leased_record(
        self, rollout_id: str, call_id: str, lease: str
    ) -> ForestNodeRecord:
        forest = self.rollouts.get(rollout_id)
        record = forest.nodes.get(call_id) if forest is not None else None
        if record is None:
            raise CursorConflictError(f"unknown call {call_id} for {rollout_id}")
        if record.state != "reserved":
            raise CursorConflictError(f"call {call_id} is {record.state}, not reserved")
        if not hmac.compare_digest(record.lease, lease):
            raise CursorConflictError(f"stale lease for call {call_id}")
        return record

    @staticmethod
    def _reservation(record: ForestNodeRecord) -> ForestReservation:
        return ForestReservation(
            call_id=record.call_id,
            parent_call_id=record.parent_call_id,
            lease=record.lease,
            prev_len=record.prev_len,
            prev_hash=record.prev_hash,
            is_new_root=record.is_new_root,
        )


@ray.remote  # pragma: no cover
class RolloutForestRegistry:
    """Ray-facing wrapper around the forest cursor state machine."""

    def __init__(self, *, lease_ttl_s: float, cursor_ttl_s: float) -> None:
        self.state = ForestCursorStateMachine(
            lease_ttl_s=lease_ttl_s, cursor_ttl_s=cursor_ttl_s
        )

    def get_candidates(self, rollout_id: str) -> list[ForestCandidate]:
        return self.state.get_candidates(rollout_id)

    def reserve_call(
        self,
        rollout_id: str,
        call_id: str,
        *,
        parent_call_id: str | None,
        prev_len: int,
        prev_hash: str,
    ) -> ForestReservation:
        return self.state.reserve_call(
            rollout_id,
            call_id,
            parent_call_id=parent_call_id,
            prev_len=prev_len,
            prev_hash=prev_hash,
        )

    def commit_call(
        self,
        rollout_id: str,
        call_id: str,
        lease: str,
        *,
        staging_key: str,
        new_len: int,
        new_hash: str,
        prompt_len: int,
        gen_len: int,
        digest: str,
        weight_version: int,
    ) -> None:
        self.state.commit_call(
            rollout_id,
            call_id,
            lease,
            staging_key=staging_key,
            new_len=new_len,
            new_hash=new_hash,
            prompt_len=prompt_len,
            gen_len=gen_len,
            digest=digest,
            weight_version=weight_version,
        )

    def fail_call(
        self, rollout_id: str, call_id: str, lease: str, *, reason: str
    ) -> None:
        self.state.fail_call(rollout_id, call_id, lease, reason=reason)

    def fail_rollout(self, rollout_id: str, *, reason: str) -> None:
        self.state.fail_rollout(rollout_id, reason=reason)

    def get_forest_manifest(self, rollout_id: str) -> ForestManifest:
        return self.state.get_forest_manifest(rollout_id)

    def clear_rollout(self, rollout_id: str) -> None:
        self.state.clear_rollout(rollout_id)

    def expire_stale(self, now: float | None = None) -> list[str]:
        return self.state.expire_stale(now=now)
