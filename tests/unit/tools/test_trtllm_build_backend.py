# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_backend():
    backend_path = (
        Path(__file__).resolve().parents[3]
        / "3rdparty"
        / "TensorRT-LLM-workspace"
        / "_backend.py"
    )
    spec = importlib.util.spec_from_file_location("trtllm_build_backend", backend_path)
    assert spec is not None
    assert spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def test_cached_wheel_is_mirrored_for_runtime(monkeypatch, tmp_path):
    backend = _load_backend()
    build_inputs = "test-build-inputs"
    source_base = tmp_path / "build-cache"
    mirror_base = tmp_path / "runtime-cache"
    wheel_directory = tmp_path / "wheel-output"
    wheel_directory.mkdir()

    monkeypatch.setattr(backend, "_build_input_tag", lambda _arch: build_inputs)
    monkeypatch.setenv("TRTLLM_WHEEL_CACHE_DIR", str(source_base))
    monkeypatch.setenv("TRTLLM_WHEEL_CACHE_MIRROR_DIR", str(mirror_base))
    monkeypatch.setenv("TRTLLM_REQUIRE_CACHED_WHEEL", "1")

    source_dir = backend._wheel_cache_dir(
        str(source_base),
        backend.TRTLLM_URL,
        backend.TRTLLM_REF,
        build_inputs,
    )
    source_dir.mkdir(parents=True)
    wheel = source_dir / f"tensorrt_llm-{backend.VERSION}-py3-none-any.whl"
    wheel.write_bytes(b"cached wheel")

    result = backend.build_wheel(str(wheel_directory))

    mirror_dir = backend._wheel_cache_dir(
        str(mirror_base),
        backend.TRTLLM_URL,
        backend.TRTLLM_REF,
        build_inputs,
    )
    assert result == wheel.name
    assert (wheel_directory / wheel.name).read_bytes() == b"cached wheel"
    assert (mirror_dir / wheel.name).read_bytes() == b"cached wheel"

    runtime_wheel_directory = tmp_path / "runtime-wheel-output"
    runtime_wheel_directory.mkdir()
    monkeypatch.setenv("TRTLLM_WHEEL_CACHE_DIR", str(mirror_base))
    monkeypatch.delenv("TRTLLM_WHEEL_CACHE_MIRROR_DIR")

    runtime_result = backend.build_wheel(str(runtime_wheel_directory))
    assert runtime_result == wheel.name
    assert (runtime_wheel_directory / wheel.name).read_bytes() == b"cached wheel"


def test_expanded_trtllm_url_substitutes_placeholders(monkeypatch):
    backend = _load_backend()
    monkeypatch.setattr(
        backend, "TRTLLM_URL", "https://${TRTLLM_TOKEN}@github.com/org/repo.git"
    )

    assert (
        backend._expanded_trtllm_url({"TRTLLM_TOKEN": "s3cret"})
        == "https://s3cret@github.com/org/repo.git"
    )


@pytest.mark.parametrize("value", [None, ""])
def test_expanded_trtllm_url_rejects_unset_or_empty_placeholder(monkeypatch, value):
    backend = _load_backend()
    monkeypatch.setattr(
        backend, "TRTLLM_URL", "https://${TRTLLM_TOKEN}@github.com/org/repo.git"
    )
    env = {} if value is None else {"TRTLLM_TOKEN": value}

    with pytest.raises(RuntimeError, match="TRTLLM_TOKEN"):
        backend._expanded_trtllm_url(env)


def test_build_failure_message_omits_the_expanded_url(monkeypatch, tmp_path):
    backend = _load_backend()
    token = "s3cret-clone-token"
    monkeypatch.setattr(backend, "_build_input_tag", lambda _arch: "test-build-inputs")
    monkeypatch.setattr(
        backend, "TRTLLM_URL", "https://${TRTLLM_TOKEN}@github.com/org/repo.git"
    )
    monkeypatch.setenv("TRTLLM_WHEEL_CACHE_DIR", str(tmp_path / "build-cache"))
    monkeypatch.delenv("TRTLLM_WHEEL_CACHE_MIRROR_DIR", raising=False)
    monkeypatch.delenv("TRTLLM_REQUIRE_CACHED_WHEEL", raising=False)
    monkeypatch.setenv("TRTLLM_TOKEN", token)

    recorded = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = argv
        return subprocess.CompletedProcess(argv, 2)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)

    wheel_directory = tmp_path / "wheel-output"
    wheel_directory.mkdir()
    with pytest.raises(RuntimeError) as excinfo:
        backend.build_wheel(str(wheel_directory))

    # The script really is handed the expanded url ...
    assert f"https://{token}@github.com/org/repo.git" in recorded["argv"]
    # ... but it must never reach the exception text that lands in build logs.
    assert token not in str(excinfo.value)
    assert "exit code 2" in str(excinfo.value)
