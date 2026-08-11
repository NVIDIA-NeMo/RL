#!/usr/bin/env python3
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

"""Bake the Super VL RL FlashInfer TRTLLM-GEN fmha dispatcher into a venv."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

DEFAULT_FMHA_ARTIFACT_PATH = "e3a8eba02eb19f4485652f84f5095524350246b5/fmha/trtllm-gen/"
DEFAULT_FMHA_MANIFEST_SHA256 = (
    "03e0f29f970de40b0fd3c6025a16a39fb6c9af2a6549a63da73cf3da8494e658"
)
DEFAULT_FMHA_MANIFEST_ENTRIES = 19227
OLD_FMHA_ARTIFACT_PATH = "158f6fa11ef139a098cfddcdddce73ca99d164ad/fmha/trtllm-gen/"


def _env(name: str, fallback_name: str | None, default: str) -> str:
    if fallback_name:
        return os.environ.get(name) or os.environ.get(fallback_name, default)
    return os.environ.get(name, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    source_dir = Path(
        os.environ.get("FLASHINFER_SOURCE_DIR", "3rdparty/flashinfer")
    ).resolve()
    artifact_path = _env(
        "FLASHINFER_FMHA_ARTIFACT_PATH",
        "FMHA_ARTIFACT_PATH",
        DEFAULT_FMHA_ARTIFACT_PATH,
    )
    manifest_sha256 = _env(
        "FLASHINFER_FMHA_MANIFEST_SHA256",
        "FMHA_MANIFEST_SHA256",
        DEFAULT_FMHA_MANIFEST_SHA256,
    )
    manifest_entries = int(
        _env(
            "FLASHINFER_FMHA_MANIFEST_ENTRIES",
            "FMHA_MANIFEST_ENTRIES",
            str(DEFAULT_FMHA_MANIFEST_ENTRIES),
        )
    )
    old_artifact_path = os.environ.get(
        "FLASHINFER_OLD_FMHA_ARTIFACT_PATH", OLD_FMHA_ARTIFACT_PATH
    )

    _require(
        source_dir.is_dir(), f"FlashInfer source directory does not exist: {source_dir}"
    )

    import flashinfer_cubin
    import flashinfer_jit_cache
    from flashinfer.artifacts import ArtifactPath, CheckSumHash
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention.modules import gen_trtllm_gen_fmha_module
    from flashinfer.jit.core import build_jit_specs

    cubin_root = Path(flashinfer_cubin.get_cubin_dir())
    manifest = cubin_root / artifact_path / "checksums.txt"
    _require(
        manifest.is_file(), f"FlashInfer cubin manifest does not exist: {manifest}"
    )
    _require(
        _sha256(manifest) == manifest_sha256,
        f"FlashInfer cubin manifest sha256 mismatch: {manifest}",
    )
    _require(
        _line_count(manifest) == manifest_entries,
        f"FlashInfer cubin manifest entry count mismatch: {manifest}",
    )
    _require(
        ArtifactPath.TRTLLM_GEN_FMHA == artifact_path,
        "flashinfer.artifacts.ArtifactPath.TRTLLM_GEN_FMHA does not match the expected artifact path",
    )
    _require(
        CheckSumHash.TRTLLM_GEN_FMHA == manifest_sha256,
        "flashinfer.artifacts.CheckSumHash.TRTLLM_GEN_FMHA does not match the expected manifest hash",
    )

    jit_env.FLASHINFER_CSRC_DIR = source_dir / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = source_dir / "include"
    jit_env.CUTLASS_INCLUDE_DIRS = [
        source_dir / "3rdparty/cutlass/include",
        source_dir / "3rdparty/cutlass/tools/util/include",
    ]
    jit_env.SPDLOG_INCLUDE_DIR = source_dir / "3rdparty/spdlog/include"
    jit_env.CCCL_INCLUDE_DIRS = [
        source_dir / "3rdparty/cccl/cub",
        source_dir / "3rdparty/cccl/libcudacxx/include",
        source_dir / "3rdparty/cccl/thrust",
    ]

    spec = gen_trtllm_gen_fmha_module()
    build_dir = jit_env.FLASHINFER_JIT_DIR / spec.name
    shutil.rmtree(build_dir, ignore_errors=True)
    build_jit_specs([spec], verbose=False, skip_prebuilt=False)

    compiled = build_dir / f"{spec.name}.so"
    _require(
        compiled.is_file(), f"FlashInfer dispatcher build did not produce {compiled}"
    )
    destination = (
        Path(flashinfer_jit_cache.get_jit_cache_dir()) / spec.name / compiled.name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(compiled, destination)

    dispatcher_bytes = destination.read_bytes()
    _require(
        artifact_path.encode() in dispatcher_bytes,
        f"{destination} does not reference the expected FlashInfer cubins",
    )
    _require(
        old_artifact_path.encode() not in dispatcher_bytes,
        f"{destination} still references the old FlashInfer cubins",
    )

    print(f"FlashInfer fmha dispatcher warmed: {destination}")
    print(f"FlashInfer fmha dispatcher sha256: {_sha256(destination)}")


if __name__ == "__main__":
    main()
