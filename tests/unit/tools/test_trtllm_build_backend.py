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
from pathlib import Path


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
    wheel = source_dir / "tensorrt_llm-1.3.0rc21-py3-none-any.whl"
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
