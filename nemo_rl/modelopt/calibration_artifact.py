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

import hashlib
import json
from collections.abc import Collection, Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from nemo_rl.modelopt.models.generation.nvfp4_refit import NVFP4Calibration

_REQUIRED_METADATA_KEYS = frozenset(
    {
        "model_id",
        "model_revision",
        "quant_cfg",
    }
)
_INPUT_AMAX_SUFFIX = ".input_quantizer._amax"
_QUANT_CFG_SHA256_KEY = "quant_cfg_sha256"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_nvfp4_calibration(
    path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    quant_cfg: str,
    expected_projection_names: Collection[str] | None = None,
) -> NVFP4Calibration:
    """Load a calibration artifact after identity and projection validation."""
    artifact_path = Path(path).expanduser()
    with safe_open(artifact_path, framework="pt", device="cpu") as artifact:
        metadata = _decode_metadata(artifact.metadata())
        _validate_identity(
            metadata,
            model_id=model_id,
            model_revision=model_revision,
            quant_cfg=quant_cfg,
        )
        raw_names = list(artifact.keys())
        normalized_names = _validate_artifact_names(raw_names)
        if expected_projection_names is not None:
            _validate_expected_projection_names(
                normalized_names,
                expected_projection_names,
            )
        input_amax = {
            normalized_name: _validated_input_amax(
                normalized_name,
                artifact.get_tensor(raw_name),
            ).clone()
            for raw_name, normalized_name in zip(
                raw_names, normalized_names, strict=True
            )
        }
    return NVFP4Calibration(input_amax=input_amax)


def _normalize_projection_name(name: str) -> str:
    if name.endswith(".weight"):
        return name
    if name.endswith(_INPUT_AMAX_SUFFIX):
        return name.removesuffix(_INPUT_AMAX_SUFFIX) + ".weight"
    raise ValueError(
        "NVFP4 calibration names must be exact HF projection '.weight' names "
        f"or ModelOpt input amax names; got {name!r}"
    )


def _validated_input_amax(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 0:
        raise ValueError(f"NVFP4 calibration requires a scalar input amax for {name!r}")
    if not bool(torch.isfinite(tensor).item()) or not bool((tensor > 0).item()):
        raise ValueError(
            f"NVFP4 calibration input amax must be finite and positive for {name!r}"
        )
    return tensor


def _decode_metadata(raw_metadata: dict[str, str] | None) -> dict[str, object]:
    metadata = raw_metadata or {}
    missing = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))
    if missing:
        raise ValueError(
            "NVFP4 calibration artifact is missing required metadata: "
            + ", ".join(missing)
        )

    decoded: dict[str, object] = {}
    metadata_keys = set(_REQUIRED_METADATA_KEYS)
    if _QUANT_CFG_SHA256_KEY in metadata:
        metadata_keys.add(_QUANT_CFG_SHA256_KEY)
    for key in metadata_keys:
        try:
            decoded[key] = json.loads(metadata[key])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"NVFP4 calibration metadata {key!r} is not valid JSON"
            ) from error
    _validate_metadata(decoded)
    return decoded


def _validate_metadata(metadata: Mapping[str, object]) -> None:
    for key in _REQUIRED_METADATA_KEYS:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"NVFP4 calibration metadata {key!r} must be a non-empty string"
            )
    quant_cfg_sha256 = metadata.get(_QUANT_CFG_SHA256_KEY)
    if quant_cfg_sha256 is not None and (
        not isinstance(quant_cfg_sha256, str)
        or len(quant_cfg_sha256) != 64
        or any(character not in "0123456789abcdef" for character in quant_cfg_sha256)
    ):
        raise ValueError(
            "NVFP4 calibration metadata 'quant_cfg_sha256' must be a lowercase "
            "SHA-256 digest"
        )


def _validate_identity(
    metadata: Mapping[str, object],
    *,
    model_id: str,
    model_revision: str,
    quant_cfg: str,
) -> None:
    expected = {
        "model_id": model_id,
        "model_revision": model_revision,
        "quant_cfg": quant_cfg,
    }
    for key, expected_value in expected.items():
        actual_value = metadata[key]
        if key == "quant_cfg" and isinstance(metadata.get(_QUANT_CFG_SHA256_KEY), str):
            quant_cfg_sha256 = metadata.get(_QUANT_CFG_SHA256_KEY)
            matches = quant_cfg_sha256 == _file_sha256(expected_value)
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                f"NVFP4 calibration {key} {actual_value!r} does not match "
                f"expected {expected_value!r}"
            )


def _file_sha256(path: object) -> str | None:
    config_path = _resolve_quant_cfg_path(path)
    if config_path is None:
        return None
    digest = hashlib.sha256()
    try:
        with config_path.open("rb") as config_file:
            for chunk in iter(lambda: config_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _resolve_quant_cfg_path(path: object) -> Path | None:
    if not isinstance(path, (str, Path)):
        return None
    config_path = Path(path).expanduser()
    if config_path.is_file():
        return config_path
    project_config_path = _PROJECT_ROOT / config_path
    if project_config_path.is_file():
        return project_config_path
    return None


def _validate_artifact_names(raw_names: list[str]) -> list[str]:
    if not raw_names:
        raise ValueError("NVFP4 calibration requires at least one input amax tensor")

    normalized_names: list[str] = []
    seen: set[str] = set()
    for raw_name in raw_names:
        normalized_name = _normalize_projection_name(raw_name)
        if normalized_name in seen:
            raise ValueError(
                "NVFP4 calibration has duplicate normalized projection name "
                f"{normalized_name!r}"
            )
        seen.add(normalized_name)
        normalized_names.append(normalized_name)

    noncanonical = [
        raw_name
        for raw_name, normalized_name in zip(raw_names, normalized_names, strict=True)
        if raw_name != normalized_name
    ]
    if noncanonical:
        raise ValueError(
            "NVFP4 calibration artifact keys must be exact HF projection "
            f"'.weight' names: {noncanonical}"
        )
    return normalized_names


def _validate_expected_projection_names(
    artifact_names: Collection[str],
    expected_projection_names: Collection[str],
) -> None:
    expected: set[str] = set()
    for name in expected_projection_names:
        normalized_name = _normalize_projection_name(name)
        if name != normalized_name:
            raise ValueError(
                "Expected NVFP4 calibration projection names must be exact HF "
                f"'.weight' names; got {name!r}"
            )
        if name in expected:
            raise ValueError(f"Duplicate expected NVFP4 projection name {name!r}")
        expected.add(name)

    actual = set(artifact_names)
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    if missing or unexpected:
        raise ValueError(
            "NVFP4 calibration projection names do not match: "
            f"missing {missing}; unexpected {unexpected}"
        )
