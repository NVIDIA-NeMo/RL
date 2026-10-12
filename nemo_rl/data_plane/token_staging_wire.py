# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-neutral token-staging wire contract.

Gym and inference backends exchange this contract as JSON, but neither is a
runtime dependency of the controller. The frozen digest encoding must remain
byte-compatible with Gym's token-capture staging schema.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

STAGING_SCHEMA_VERSION = 2
STAGING_DIGEST_VERSION = 2
EXTRAS_DIGEST_VERSION = 1

_CALL_DIGEST_DOMAIN = b"nemo-gym-staging-call-v2"
_EXTRAS_DIGEST_DOMAIN = b"nemo-gym-staging-extras-v1"
_TOKEN_DIGEST_DOMAIN = b"nemo-gym-staging-prefix-v2"
_CHAIN_DIGEST_DOMAIN = b"nemo-gym-staging-chain-v1"
_HEX_DIGEST_LENGTH = 64
_ADMISSION_FIELDS = {
    "schema_version",
    "rollout_id",
    "model_call_id",
    "parent_call_id",
    "prev_len",
    "mode",
    "required_prefix_token_ids",
    "staging_chain",
    "parent_chain_hash",
}

CaptureMode = Literal["text", "token_in"]


def _encode_bytes(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _encode_text(value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"expected text, got {type(value).__name__}")
    return _encode_bytes(value.encode("utf-8"))


def _encode_optional_text(value: str | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + _encode_text(value)


def _encode_uint(value: int, *, field: str) -> bytes:
    if type(value) is not int or not 0 <= value <= (2**64 - 1):
        raise ValueError(f"{field} must be an unsigned 64-bit integer, got {value!r}")
    return struct.pack(">Q", value)


def _validate_digest(value: str, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def _encode_present_digest(value: str, *, field: str) -> bytes:
    _validate_digest(value, field=field)
    return b"\x01" + bytes.fromhex(value)


def encode_token_ids(token_ids: Sequence[int]) -> bytes:
    """Encode token IDs as a length followed by unsigned big-endian values."""
    encoded = bytearray(_encode_uint(len(token_ids), field="token_ids length"))
    for token_id in token_ids:
        encoded.extend(_encode_uint(token_id, field="token_id"))
    return bytes(encoded)


def hash_token_ids(token_ids: Sequence[int]) -> str:
    """Hash one exact cumulative token prefix."""
    return hashlib.sha256(
        _TOKEN_DIGEST_DOMAIN + encode_token_ids(token_ids)
    ).hexdigest()


def compute_chain_hash(
    parent_chain_hash: str | None, token_ids_delta: Sequence[int]
) -> str:
    """Chain one staged token delta onto its parent digest."""
    payload = bytearray()
    if parent_chain_hash is None:
        payload += b"\x00"
    else:
        _validate_digest(parent_chain_hash, field="parent_chain_hash")
        payload += b"\x01" + bytes.fromhex(parent_chain_hash)
    payload += encode_token_ids(token_ids_delta)
    return hashlib.sha256(_CHAIN_DIGEST_DOMAIN + bytes(payload)).hexdigest()


def _encode_float32_values(values: Sequence[float], *, field: str) -> bytes:
    encoded = bytearray(_encode_uint(len(values), field=f"{field} length"))
    for value in values:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(
                f"{field} values must be finite Python floats, got {value!r}"
            )
        try:
            packed = struct.pack(">f", value)
        except (OverflowError, struct.error) as error:
            raise ValueError(
                f"{field} value cannot be represented as float32: {value!r}"
            ) from error
        if not math.isfinite(struct.unpack(">f", packed)[0]):
            raise ValueError(f"{field} value overflows float32: {value!r}")
        encoded.extend(packed)
    return bytes(encoded)


def _encode_extra(value: Any) -> bytes:
    if value is None:
        return b"N"
    if type(value) is bool:
        return b"B\x01" if value else b"B\x00"
    if type(value) is int:
        if not -(2**63) <= value <= (2**63 - 1):
            raise ValueError(f"extras integer is outside signed 64-bit range: {value}")
        return b"I" + struct.pack(">q", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"extras floats must be finite, got {value!r}")
        return b"F" + struct.pack(">d", value)
    if type(value) is str:
        return b"S" + _encode_text(value)
    if type(value) is list:
        return (
            b"L"
            + _encode_uint(len(value), field="extras list length")
            + b"".join(_encode_extra(item) for item in value)
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("extras mappings must have string keys")
        keys = sorted(value, key=lambda key: key.encode("utf-8"))
        return (
            b"D"
            + _encode_uint(len(keys), field="extras mapping length")
            + b"".join(_encode_text(key) + _encode_extra(value[key]) for key in keys)
        )
    raise TypeError(f"extras contain unsupported value type {type(value).__name__}")


def compute_extras_digest(extras: Mapping[str, Any] | None) -> str:
    """Digest a normalized JSON-like extras envelope."""
    normalized: dict[str, Any] | None
    if extras is None:
        normalized = None
    elif isinstance(extras, Mapping):
        normalized = dict(extras)
    else:
        raise TypeError(
            f"extras must be a mapping or None, got {type(extras).__name__}"
        )
    payload = struct.pack(">B", EXTRAS_DIGEST_VERSION) + _encode_extra(normalized)
    return hashlib.sha256(_EXTRAS_DIGEST_DOMAIN + payload).hexdigest()


def compute_staging_digest(
    *,
    schema_version: int,
    digest_version: int,
    extras_digest_version: int,
    rollout_id: str,
    model_call_id: str,
    parent_call_id: str | None,
    mode: str,
    prev_len: int,
    delta_len: int,
    cum_len: int,
    weight_version: int,
    token_ids_delta: Sequence[int],
    token_mask_delta: Sequence[float],
    generation_log_probs_delta: Sequence[float],
    extras_digest: str,
    chain_hash: str,
    cumulative_hash: str,
) -> str:
    """Compute the frozen v2 digest for one staged call delta."""
    if type(schema_version) is not int or schema_version != STAGING_SCHEMA_VERSION:
        raise ValueError(f"unsupported staging schema version {schema_version}")
    if type(digest_version) is not int or digest_version != STAGING_DIGEST_VERSION:
        raise ValueError(f"unsupported staging digest version {digest_version}")
    if (
        type(extras_digest_version) is not int
        or extras_digest_version != EXTRAS_DIGEST_VERSION
    ):
        raise ValueError(f"unsupported extras digest version {extras_digest_version}")
    if not rollout_id or not model_call_id:
        raise ValueError("rollout_id and model_call_id must be non-empty")
    if parent_call_id == "":
        raise ValueError("parent_call_id must be non-empty when present")
    if mode not in ("text", "token_in"):
        raise ValueError(f"unsupported capture mode {mode!r}")
    if parent_call_id is None and (prev_len != 0 or mode != "text"):
        raise ValueError(
            "a parentless call must be a text-mode root with prev_len == 0"
        )
    if parent_call_id is not None and (prev_len == 0 or mode != "token_in"):
        raise ValueError("a child call must use token_in mode with prev_len > 0")
    if delta_len == 0:
        raise ValueError("a staged call delta must contain at least one token")
    if delta_len != len(token_ids_delta):
        raise ValueError(
            f"delta_len {delta_len} does not match {len(token_ids_delta)} token IDs"
        )
    if not (
        len(token_ids_delta) == len(token_mask_delta) == len(generation_log_probs_delta)
    ):
        raise ValueError(
            "token IDs, masks, and log probabilities must have equal lengths"
        )
    if any(mask not in (0.0, 1.0) for mask in token_mask_delta):
        raise ValueError("token_mask_delta must contain only 0.0 or 1.0")
    if any(
        mask == 0.0 and log_prob != 0.0
        for mask, log_prob in zip(token_mask_delta, generation_log_probs_delta)
    ):
        raise ValueError("prompt-carry log probabilities must be 0.0")
    if cum_len != prev_len + delta_len:
        raise ValueError(
            f"cum_len {cum_len} does not equal prev_len + delta_len "
            f"({prev_len + delta_len})"
        )
    _validate_digest(extras_digest, field="extras_digest")

    payload = bytearray(
        struct.pack(">BBB", schema_version, digest_version, extras_digest_version)
    )
    payload.extend(_encode_text(rollout_id))
    payload.extend(_encode_text(model_call_id))
    payload.extend(_encode_optional_text(parent_call_id))
    payload.extend(_encode_text(mode))
    payload.extend(_encode_uint(prev_len, field="prev_len"))
    payload.extend(_encode_uint(delta_len, field="delta_len"))
    payload.extend(_encode_uint(cum_len, field="cum_len"))
    payload.extend(_encode_uint(weight_version, field="weight_version"))
    payload.extend(_encode_bytes(encode_token_ids(token_ids_delta)))
    payload.extend(
        _encode_bytes(
            _encode_float32_values(token_mask_delta, field="token_mask_delta")
        )
    )
    payload.extend(
        _encode_bytes(
            _encode_float32_values(
                generation_log_probs_delta,
                field="generation_log_probs_delta",
            )
        )
    )
    payload.extend(bytes.fromhex(extras_digest))
    payload.extend(_encode_present_digest(chain_hash, field="chain_hash"))
    payload.extend(_encode_present_digest(cumulative_hash, field="cumulative_hash"))
    return hashlib.sha256(_CALL_DIGEST_DOMAIN + bytes(payload)).hexdigest()


def build_staging_delta(
    *,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    generated_log_probs: list[float],
    prev_len: int,
) -> tuple[list[int], list[float], list[float]]:
    """Slice a full prompt/generation pair into the next staged delta."""
    if prev_len < 0 or prev_len > len(prompt_token_ids):
        raise ValueError(
            f"prev_len={prev_len} is outside prompt length {len(prompt_token_ids)}"
        )
    if len(generated_token_ids) != len(generated_log_probs):
        raise ValueError(
            "generated token and log-probability lengths differ: "
            f"{len(generated_token_ids)} != {len(generated_log_probs)}"
        )
    prompt_delta = prompt_token_ids[prev_len:]
    token_ids_delta = prompt_delta + generated_token_ids
    token_mask_delta = [0.0] * len(prompt_delta) + [1.0] * len(generated_token_ids)
    logprobs_delta = [0.0] * len(prompt_delta) + generated_log_probs
    if not token_ids_delta:
        raise ValueError("staging delta must contain at least one token")
    return token_ids_delta, token_mask_delta, logprobs_delta


def _strict_token_ids(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in value
    ):
        raise ValueError(f"{field} must be a list of non-negative integers")
    return list(value)


@dataclass(frozen=True, slots=True)
class TokenCaptureAdmission:
    """Validated capture admission received from Gym over JSON."""

    schema_version: int
    rollout_id: str
    model_call_id: str
    parent_call_id: str | None
    prev_len: int
    mode: CaptureMode
    required_prefix_token_ids: list[int]
    staging_chain: list[str]
    parent_chain_hash: str | None

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> TokenCaptureAdmission:
        """Validate and normalize one admission object without importing Gym."""
        unknown = set(value) - _ADMISSION_FIELDS
        if unknown:
            raise ValueError(f"unknown ng_capture fields: {sorted(unknown)}")
        schema_version = value.get("schema_version", STAGING_SCHEMA_VERSION)
        if type(schema_version) is not int or schema_version != STAGING_SCHEMA_VERSION:
            raise ValueError(f"unsupported staging schema version {schema_version!r}")
        rollout_id = value.get("rollout_id")
        model_call_id = value.get("model_call_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("rollout_id must be a non-empty string")
        if not isinstance(model_call_id, str) or not model_call_id:
            raise ValueError("model_call_id must be a non-empty string")
        parent_call_id = value.get("parent_call_id")
        if parent_call_id is not None and (
            not isinstance(parent_call_id, str) or not parent_call_id
        ):
            raise ValueError("parent_call_id must be a non-empty string or null")
        prev_len = value.get("prev_len", 0)
        if type(prev_len) is not int or prev_len < 0:
            raise ValueError("prev_len must be a non-negative integer")
        mode = value.get("mode")
        if mode not in ("text", "token_in"):
            raise ValueError(f"unsupported capture mode {mode!r}")
        required_prefix = _strict_token_ids(
            value.get("required_prefix_token_ids", []),
            field="required_prefix_token_ids",
        )
        staging_chain = value.get("staging_chain", [])
        if not isinstance(staging_chain, list) or any(
            not isinstance(key, str) or not key for key in staging_chain
        ):
            raise ValueError("staging_chain must be a list of non-empty strings")
        parent_chain_hash = value.get("parent_chain_hash")
        if parent_chain_hash is not None:
            _validate_digest(parent_chain_hash, field="parent_chain_hash")
        if mode == "token_in":
            if parent_call_id is None or prev_len == 0:
                raise ValueError(
                    "token_in admission requires a parent_call_id and prev_len > 0"
                )
            if parent_chain_hash is None:
                raise ValueError("token_in admission requires the parent's chain hash")
            if not staging_chain and len(required_prefix) != prev_len:
                raise ValueError("required_prefix_token_ids length must equal prev_len")
        elif (
            parent_call_id is not None
            or prev_len != 0
            or required_prefix
            or staging_chain
            or parent_chain_hash is not None
        ):
            raise ValueError(
                "text admission must be a parentless root with no required prefix"
            )
        return cls(
            schema_version=schema_version,
            rollout_id=rollout_id,
            model_call_id=model_call_id,
            parent_call_id=parent_call_id,
            prev_len=prev_len,
            mode=mode,
            required_prefix_token_ids=required_prefix,
            staging_chain=list(staging_chain),
            parent_chain_hash=parent_chain_hash,
        )


@dataclass(frozen=True, slots=True)
class StagedTokenRecord:
    """One complete token delta ready for durable TransferQueue staging."""

    schema_version: int
    digest_version: int
    extras_digest_version: int
    rollout_id: str
    model_call_id: str
    parent_call_id: str | None
    mode: CaptureMode
    prev_len: int
    delta_len: int
    cum_len: int
    weight_version: int
    digest: str
    token_ids_delta: list[int]
    token_mask_delta: list[float]
    generation_log_probs_delta: list[float]
    extras: dict[str, Any] | None
    extras_digest: str
    chain_hash: str
    cumulative_hash: str

    @property
    def staging_key(self) -> str:
        """Return the deterministic TransferQueue key for this call."""
        return f"{self.rollout_id}/{self.model_call_id}"


@dataclass(frozen=True, slots=True)
class StagingWriteResult:
    """Dependency-neutral result of one TransferQueue staging write."""

    ok: bool
    staging_key: str
    error: str | None = None


def build_staged_token_record(
    *,
    admission: TokenCaptureAdmission,
    prefix_token_ids: list[int],
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    generated_logprobs: list[float],
    weight_version: int,
    extras: Mapping[str, Any] | None,
) -> StagedTokenRecord:
    """Validate one completed call and build its frozen staging record."""
    prefix = _strict_token_ids(prefix_token_ids, field="prefix_token_ids")
    prompt = _strict_token_ids(prompt_token_ids, field="prompt_token_ids")
    generated = _strict_token_ids(generated_token_ids, field="generated_token_ids")
    if type(weight_version) is not int or weight_version < 0:
        raise ValueError("weight_version must be a non-negative integer")
    if admission.mode == "text":
        if prefix:
            raise ValueError("a text root admission accepts no prefix_token_ids")
    else:
        if len(prefix) != admission.prev_len:
            raise ValueError(
                f"prefix_token_ids length {len(prefix)} does not equal "
                f"prev_len {admission.prev_len}"
            )
        if admission.required_prefix_token_ids and (
            prefix != admission.required_prefix_token_ids
        ):
            raise ValueError(
                "prefix_token_ids conflict with the admission's inline prefix"
            )
        if prompt[: admission.prev_len] != prefix:
            raise ValueError(
                "generation prompt does not begin with the admitted token prefix"
            )
    normalized_logprobs = [float(value) for value in generated_logprobs]
    token_ids_delta, token_mask_delta, logprobs_delta = build_staging_delta(
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        generated_log_probs=normalized_logprobs,
        prev_len=admission.prev_len,
    )
    delta_len = len(token_ids_delta)
    cum_len = admission.prev_len + delta_len
    normalized_extras = dict(extras) if extras is not None else None
    extras_digest = compute_extras_digest(normalized_extras)
    chain_hash = compute_chain_hash(admission.parent_chain_hash, token_ids_delta)
    cumulative_hash = hash_token_ids(prompt + generated)
    digest = compute_staging_digest(
        schema_version=admission.schema_version,
        digest_version=STAGING_DIGEST_VERSION,
        extras_digest_version=EXTRAS_DIGEST_VERSION,
        rollout_id=admission.rollout_id,
        model_call_id=admission.model_call_id,
        parent_call_id=admission.parent_call_id,
        mode=admission.mode,
        prev_len=admission.prev_len,
        delta_len=delta_len,
        cum_len=cum_len,
        weight_version=weight_version,
        token_ids_delta=token_ids_delta,
        token_mask_delta=token_mask_delta,
        generation_log_probs_delta=logprobs_delta,
        extras_digest=extras_digest,
        chain_hash=chain_hash,
        cumulative_hash=cumulative_hash,
    )
    return StagedTokenRecord(
        schema_version=admission.schema_version,
        digest_version=STAGING_DIGEST_VERSION,
        extras_digest_version=EXTRAS_DIGEST_VERSION,
        rollout_id=admission.rollout_id,
        model_call_id=admission.model_call_id,
        parent_call_id=admission.parent_call_id,
        mode=admission.mode,
        prev_len=admission.prev_len,
        delta_len=delta_len,
        cum_len=cum_len,
        weight_version=weight_version,
        digest=digest,
        token_ids_delta=token_ids_delta,
        token_mask_delta=token_mask_delta,
        generation_log_probs_delta=logprobs_delta,
        extras=normalized_extras,
        extras_digest=extras_digest,
        chain_hash=chain_hash,
        cumulative_hash=cumulative_hash,
    )


def commit_coords(record: StagedTokenRecord) -> dict[str, Any]:
    """Build the staged commit acknowledgement consumed by Gym."""
    return {
        "schema_version": record.schema_version,
        "digest_version": record.digest_version,
        "extras_digest_version": record.extras_digest_version,
        "rollout_id": record.rollout_id,
        "model_call_id": record.model_call_id,
        "parent_call_id": record.parent_call_id,
        "prev_len": record.prev_len,
        "delta_len": record.delta_len,
        "cum_len": record.cum_len,
        "weight_version": record.weight_version,
        "disposition": "staged",
        "digest": record.digest,
        "extras_digest": record.extras_digest,
        "staging_key": record.staging_key,
        "chain_hash": record.chain_hash,
        "cumulative_hash": record.cumulative_hash,
    }


def failed_commit_coords(
    admission: TokenCaptureAdmission, *, weight_version: int
) -> dict[str, Any]:
    """Build a token-free failed acknowledgement for a rejected capture."""
    return {
        "schema_version": admission.schema_version,
        "digest_version": STAGING_DIGEST_VERSION,
        "extras_digest_version": EXTRAS_DIGEST_VERSION,
        "rollout_id": admission.rollout_id,
        "model_call_id": admission.model_call_id,
        "parent_call_id": admission.parent_call_id,
        "prev_len": admission.prev_len,
        "delta_len": 0,
        "cum_len": admission.prev_len,
        "weight_version": weight_version,
        "disposition": "capture_failed",
        "digest": None,
        "extras_digest": None,
        "staging_key": None,
        "chain_hash": None,
        "cumulative_hash": None,
    }
