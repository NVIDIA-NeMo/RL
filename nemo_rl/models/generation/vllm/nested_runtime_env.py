# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fail-closed runtime-environment contract for nested vLLM Ray workers."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from typing import Any


NESTED_RUNTIME_ENV_JSON_ENV = "NRL_VLLM_NESTED_RUNTIME_ENV_JSON"
NESTED_RUNTIME_ENV_SHA256_ENV = (
    "NRL_VLLM_NESTED_RUNTIME_ENV_SHA256"
)
RUNTIME_ENV_CONTRACT_SHA256_ENV = (
    "NRL_VLLM_RUNTIME_ENV_CONTRACT_SHA256"
)
NESTED_RUNTIME_ENV_SCHEMA = "nemo_rl.vllm_nested_runtime_env.v1"

_RUNTIME_ENV_REQUIRED_KEYS = frozenset({"py_executable", "env_vars"})
_RUNTIME_ENV_OPTIONAL_KEYS = frozenset({"nsight"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class NestedRuntimeEnvContractError(RuntimeError):
    """Raised when the nested-worker runtime contract is invalid."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_string_mapping(
    value: object,
    *,
    field_name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise NestedRuntimeEnvContractError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise NestedRuntimeEnvContractError(
                f"{field_name} keys must be non-empty strings"
            )
        if not isinstance(item, str):
            raise NestedRuntimeEnvContractError(
                f"{field_name}[{key!r}] must be a string"
            )
        normalized[key] = item
    return normalized


def _normalize_runtime_env(
    runtime_env: object,
    *,
    allow_contract_hash: bool,
) -> dict[str, Any]:
    if not isinstance(runtime_env, Mapping):
        raise NestedRuntimeEnvContractError(
            "nested runtime_env must be a mapping"
        )

    keys = set(runtime_env)
    if not all(isinstance(key, str) for key in keys):
        raise NestedRuntimeEnvContractError(
            "nested runtime_env keys must be strings"
        )
    allowed_keys = _RUNTIME_ENV_REQUIRED_KEYS | _RUNTIME_ENV_OPTIONAL_KEYS
    if keys - allowed_keys:
        raise NestedRuntimeEnvContractError(
            "nested runtime_env has unsupported keys: "
            f"{sorted(keys - allowed_keys)!r}"
        )
    missing_keys = _RUNTIME_ENV_REQUIRED_KEYS - keys
    if missing_keys:
        raise NestedRuntimeEnvContractError(
            "nested runtime_env is missing required keys: "
            f"{sorted(missing_keys)!r}"
        )

    python_executable = runtime_env["py_executable"]
    if not isinstance(python_executable, str) or not python_executable:
        raise NestedRuntimeEnvContractError(
            "nested runtime_env py_executable must be a non-empty string"
        )

    env_vars = _normalize_string_mapping(
        runtime_env["env_vars"],
        field_name="nested runtime_env env_vars",
    )
    contract_hash = env_vars.get(RUNTIME_ENV_CONTRACT_SHA256_ENV)
    if contract_hash is not None:
        if not allow_contract_hash:
            raise NestedRuntimeEnvContractError(
                "nested runtime_env input must not contain the reserved "
                f"{RUNTIME_ENV_CONTRACT_SHA256_ENV} variable"
            )
        if _SHA256_PATTERN.fullmatch(contract_hash) is None:
            raise NestedRuntimeEnvContractError(
                f"{RUNTIME_ENV_CONTRACT_SHA256_ENV} must be a lowercase "
                "SHA-256 digest"
            )

    normalized: dict[str, Any] = {
        "py_executable": python_executable,
        "env_vars": env_vars,
    }
    if "nsight" in runtime_env:
        normalized["nsight"] = _normalize_string_mapping(
            runtime_env["nsight"],
            field_name="nested runtime_env nsight",
        )
    return normalized


def _contract_envelope(
    runtime_env: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": NESTED_RUNTIME_ENV_SCHEMA,
        "runtime_env": runtime_env,
    }


def _with_contract_hash(
    runtime_env: Mapping[str, Any],
    digest: str,
) -> dict[str, Any]:
    desired = copy.deepcopy(dict(runtime_env))
    desired["env_vars"][RUNTIME_ENV_CONTRACT_SHA256_ENV] = digest
    return desired


def export_nested_runtime_env_contract(
    runtime_env: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Export a canonical contract and return its worker runtime environment.

    The serialized contract intentionally excludes its own digest. The
    returned runtime environment includes that digest as a reserved
    environment variable so each nested worker can prove which contract it
    received.
    """

    normalized = _normalize_runtime_env(
        runtime_env,
        allow_contract_hash=False,
    )
    serialized = _canonical_json(_contract_envelope(normalized))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    os.environ[NESTED_RUNTIME_ENV_JSON_ENV] = serialized
    os.environ[NESTED_RUNTIME_ENV_SHA256_ENV] = digest
    return _with_contract_hash(normalized, digest), digest


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NestedRuntimeEnvContractError(
                f"nested runtime contract contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def load_nested_runtime_env_contract() -> tuple[dict[str, Any], str]:
    """Load and authenticate the contract exported by the parent process."""

    serialized = os.environ.get(NESTED_RUNTIME_ENV_JSON_ENV)
    expected_digest = os.environ.get(NESTED_RUNTIME_ENV_SHA256_ENV)
    if serialized is None or expected_digest is None:
        missing = [
            name
            for name, value in (
                (NESTED_RUNTIME_ENV_JSON_ENV, serialized),
                (NESTED_RUNTIME_ENV_SHA256_ENV, expected_digest),
            )
            if value is None
        ]
        raise NestedRuntimeEnvContractError(
            "nested runtime contract is missing required environment "
            f"variables: {missing!r}"
        )
    if _SHA256_PATTERN.fullmatch(expected_digest) is None:
        raise NestedRuntimeEnvContractError(
            f"{NESTED_RUNTIME_ENV_SHA256_ENV} must be a lowercase "
            "SHA-256 digest"
        )

    try:
        envelope = json.loads(
            serialized,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except NestedRuntimeEnvContractError:
        raise
    except (TypeError, ValueError) as error:
        raise NestedRuntimeEnvContractError(
            "nested runtime contract is not valid JSON"
        ) from error

    if not isinstance(envelope, Mapping):
        raise NestedRuntimeEnvContractError(
            "nested runtime contract envelope must be a mapping"
        )
    if set(envelope) != {"schema", "runtime_env"}:
        raise NestedRuntimeEnvContractError(
            "nested runtime contract envelope must contain exactly "
            "'schema' and 'runtime_env'"
        )
    if envelope["schema"] != NESTED_RUNTIME_ENV_SCHEMA:
        raise NestedRuntimeEnvContractError(
            "nested runtime contract has an unsupported schema"
        )

    normalized = _normalize_runtime_env(
        envelope["runtime_env"],
        allow_contract_hash=False,
    )
    canonical = _canonical_json(_contract_envelope(normalized))
    if not hmac.compare_digest(serialized, canonical):
        raise NestedRuntimeEnvContractError(
            "nested runtime contract JSON is not canonical"
        )

    actual_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise NestedRuntimeEnvContractError(
            "nested runtime contract SHA-256 mismatch"
        )
    return _with_contract_hash(normalized, actual_digest), actual_digest


def _merge_string_mapping(
    existing: object,
    desired: object,
    *,
    field_name: str,
) -> dict[str, str]:
    merged = _normalize_string_mapping(existing, field_name=field_name)
    desired_mapping = _normalize_string_mapping(
        desired,
        field_name=field_name,
    )
    for key, value in desired_mapping.items():
        if key in merged and merged[key] != value:
            raise NestedRuntimeEnvContractError(
                f"{field_name}[{key!r}] conflicts with the nested "
                "runtime contract"
            )
        merged[key] = value
    return merged


def merge_nested_runtime_env(
    existing: Mapping[str, Any] | None,
    desired: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge the authenticated worker settings without overwriting conflicts."""

    normalized_desired = _normalize_runtime_env(
        desired,
        allow_contract_hash=True,
    )
    if existing is None:
        return copy.deepcopy(normalized_desired)
    if not isinstance(existing, Mapping):
        raise NestedRuntimeEnvContractError(
            "existing Ray runtime_env must be a mapping or None"
        )

    merged = copy.deepcopy(dict(existing))
    for key, desired_value in normalized_desired.items():
        if key not in merged:
            merged[key] = copy.deepcopy(desired_value)
            continue
        if key in {"env_vars", "nsight"}:
            merged[key] = _merge_string_mapping(
                merged[key],
                desired_value,
                field_name=f"Ray runtime_env {key}",
            )
            continue
        if merged[key] != desired_value:
            raise NestedRuntimeEnvContractError(
                f"Ray runtime_env {key!r} conflicts with the nested "
                "runtime contract"
            )
    return merged
