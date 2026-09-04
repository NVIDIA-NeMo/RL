# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import hashlib
import json
import os

import pytest

from nemo_rl.models.generation.vllm.nested_runtime_env import (
    NESTED_RUNTIME_ENV_JSON_ENV,
    NESTED_RUNTIME_ENV_SCHEMA,
    NESTED_RUNTIME_ENV_SHA256_ENV,
    RUNTIME_ENV_CONTRACT_SHA256_ENV,
    NestedRuntimeEnvContractError,
    export_nested_runtime_env_contract,
    load_nested_runtime_env_contract,
    merge_nested_runtime_env,
)


def _clear_contract_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NESTED_RUNTIME_ENV_JSON_ENV, raising=False)
    monkeypatch.delenv(NESTED_RUNTIME_ENV_SHA256_ENV, raising=False)


def test_export_and_load_round_trip_is_canonical_and_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_contract_env(monkeypatch)
    source = {
        "nsight": {"t": "cuda,nvtx", "o": "worker_%p"},
        "env_vars": {
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "NCCL_CUMEM_ENABLE": "1",
        },
        "py_executable": "/opt/nemo/bin/python",
    }

    exported, digest = export_nested_runtime_env_contract(source)

    expected_serialized = json.dumps(
        {
            "runtime_env": source,
            "schema": NESTED_RUNTIME_ENV_SCHEMA,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert os.environ[NESTED_RUNTIME_ENV_JSON_ENV] == expected_serialized
    assert digest == hashlib.sha256(
        expected_serialized.encode("utf-8")
    ).hexdigest()
    assert (
        os.environ[NESTED_RUNTIME_ENV_SHA256_ENV]
        == digest
    )
    assert exported["env_vars"][RUNTIME_ENV_CONTRACT_SHA256_ENV] == digest
    assert RUNTIME_ENV_CONTRACT_SHA256_ENV not in source["env_vars"]
    assert load_nested_runtime_env_contract() == (exported, digest)


@pytest.mark.parametrize(
    "runtime_env,match",
    [
        (
            {"py_executable": "/python", "env_vars": {}, "pip": []},
            "unsupported keys",
        ),
        ({"py_executable": "/python"}, "missing required keys"),
        (
            {"py_executable": "", "env_vars": {}},
            "non-empty string",
        ),
        (
            {"py_executable": "/python", "env_vars": {"A": 1}},
            "must be a string",
        ),
        (
            {"py_executable": "/python", "env_vars": {}, "nsight": []},
            "nsight must be a mapping",
        ),
        (
            {
                "py_executable": "/python",
                "env_vars": {
                    RUNTIME_ENV_CONTRACT_SHA256_ENV: "0" * 64
                },
            },
            "reserved",
        ),
    ],
)
def test_export_rejects_invalid_contracts(
    monkeypatch: pytest.MonkeyPatch,
    runtime_env: dict[str, object],
    match: str,
) -> None:
    _clear_contract_env(monkeypatch)
    with pytest.raises(NestedRuntimeEnvContractError, match=match):
        export_nested_runtime_env_contract(runtime_env)


def test_load_rejects_missing_contract_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_contract_env(monkeypatch)
    with pytest.raises(NestedRuntimeEnvContractError, match="missing"):
        load_nested_runtime_env_contract()


def test_load_rejects_modified_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_contract_env(monkeypatch)
    export_nested_runtime_env_contract(
        {"py_executable": "/python", "env_vars": {"A": "1"}}
    )
    serialized = os.environ[NESTED_RUNTIME_ENV_JSON_ENV]
    monkeypatch.setenv(
        NESTED_RUNTIME_ENV_JSON_ENV,
        serialized.replace('"A":"1"', '"A":"2"'),
    )

    with pytest.raises(NestedRuntimeEnvContractError, match="SHA-256"):
        load_nested_runtime_env_contract()


def test_load_rejects_modified_or_noncanonical_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_contract_env(monkeypatch)
    export_nested_runtime_env_contract(
        {"py_executable": "/python", "env_vars": {}}
    )
    monkeypatch.setenv(NESTED_RUNTIME_ENV_SHA256_ENV, "A" * 64)

    with pytest.raises(
        NestedRuntimeEnvContractError,
        match="lowercase SHA-256",
    ):
        load_nested_runtime_env_contract()


def test_load_rejects_noncanonical_json_even_with_matching_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_contract_env(monkeypatch)
    export_nested_runtime_env_contract(
        {"py_executable": "/python", "env_vars": {}}
    )
    envelope = json.loads(os.environ[NESTED_RUNTIME_ENV_JSON_ENV])
    noncanonical = json.dumps(envelope, indent=2)
    monkeypatch.setenv(NESTED_RUNTIME_ENV_JSON_ENV, noncanonical)
    monkeypatch.setenv(
        NESTED_RUNTIME_ENV_SHA256_ENV,
        hashlib.sha256(noncanonical.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(NestedRuntimeEnvContractError, match="canonical"):
        load_nested_runtime_env_contract()


def test_load_rejects_duplicate_json_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_contract_env(monkeypatch)
    serialized = (
        '{"runtime_env":{"env_vars":{},"py_executable":"/python"},'
        f'"schema":"{NESTED_RUNTIME_ENV_SCHEMA}",'
        f'"schema":"{NESTED_RUNTIME_ENV_SCHEMA}"}}'
    )
    monkeypatch.setenv(NESTED_RUNTIME_ENV_JSON_ENV, serialized)
    monkeypatch.setenv(
        NESTED_RUNTIME_ENV_SHA256_ENV,
        hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )

    with pytest.raises(NestedRuntimeEnvContractError, match="duplicate"):
        load_nested_runtime_env_contract()


def test_merge_preserves_unrelated_settings_and_adds_contract_values() -> None:
    desired = {
        "py_executable": "/python",
        "env_vars": {
            "A": "1",
            RUNTIME_ENV_CONTRACT_SHA256_ENV: "a" * 64,
        },
        "nsight": {"t": "cuda,nvtx"},
    }
    existing = {
        "working_dir": "/workspace",
        "env_vars": {"A": "1", "OTHER": "kept"},
        "nsight": {"t": "cuda,nvtx", "o": "kept"},
    }

    merged = merge_nested_runtime_env(existing, desired)

    assert merged == {
        "working_dir": "/workspace",
        "py_executable": "/python",
        "env_vars": {
            "A": "1",
            "OTHER": "kept",
            RUNTIME_ENV_CONTRACT_SHA256_ENV: "a" * 64,
        },
        "nsight": {"t": "cuda,nvtx", "o": "kept"},
    }
    assert existing["env_vars"] == {"A": "1", "OTHER": "kept"}


def test_merge_accepts_none_and_returns_an_independent_copy() -> None:
    desired = {
        "py_executable": "/python",
        "env_vars": {
            RUNTIME_ENV_CONTRACT_SHA256_ENV: "b" * 64,
        },
    }

    merged = merge_nested_runtime_env(None, desired)
    merged["env_vars"]["NEW"] = "value"

    assert "NEW" not in desired["env_vars"]


@pytest.mark.parametrize(
    "existing,match",
    [
        (
            {
                "py_executable": "/other",
                "env_vars": {},
            },
            "py_executable",
        ),
        (
            {
                "env_vars": {"A": "different"},
            },
            r"env_vars\['A'\]",
        ),
        (
            {
                "env_vars": {},
                "nsight": {"t": "cuda-hw"},
            },
            r"nsight\['t'\]",
        ),
    ],
)
def test_merge_rejects_conflicts(
    existing: dict[str, object],
    match: str,
) -> None:
    desired = {
        "py_executable": "/python",
        "env_vars": {
            "A": "1",
            RUNTIME_ENV_CONTRACT_SHA256_ENV: "c" * 64,
        },
        "nsight": {"t": "cuda,nvtx"},
    }

    with pytest.raises(NestedRuntimeEnvContractError, match=match):
        merge_nested_runtime_env(existing, desired)
