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

"""Check the actor build adapter without a CUDA toolkit or network access."""

import importlib.util
import subprocess
from email.parser import Parser
from pathlib import Path

import pytest


def _load_backend(monkeypatch, actor=True):
    path = (
        Path(__file__).resolve().parents[3]
        / "3rdparty/DeepEP-workspace/actor/_backend.py"
    )
    monkeypatch.chdir(path.parent if actor else path.parent.parent)
    spec = importlib.util.spec_from_file_location("deepep_build_backend", path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


@pytest.mark.parametrize("actor", [True, False])
def test_metadata_does_not_fetch_or_compile(monkeypatch, tmp_path, actor):
    backend = _load_backend(monkeypatch, actor)

    def unexpected(*args, **kwargs):
        raise AssertionError("Metadata discovery must not invoke external commands")

    monkeypatch.setattr(backend.subprocess, "run", unexpected)
    name = backend.prepare_metadata_for_build_wheel(tmp_path)
    metadata = Parser().parsestr((tmp_path / name / "METADATA").read_text())
    assert metadata["Version"] == backend._META["project"]["version"]
    assert metadata.get_all("Requires-Dist") == (
        ["nvidia-ml-py>=12.0.0"] if actor else ["pynvml"]
    )


@pytest.mark.parametrize("actor", [True, False])
def test_build_uses_pinned_commit_and_multinode_settings(monkeypatch, tmp_path, actor):
    backend = _load_backend(monkeypatch, actor)
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()

    git("init")
    git("config", "user.name", "Build test")
    git("config", "user.email", "build@example.com")
    version = backend._META["project"]["version"]
    wheel_name = f"deep_ep-{version}-cp313-cp313-linux_x86_64.whl"
    # Stand-in for the CUDA build: observe the actual subprocess environment.
    (source / "setup.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "assert os.environ['HYBRID_EP_MULTINODE'] == '1'\n"
        "assert os.environ['USE_NIXL'] == '0'\n"
        "assert os.environ['TORCH_CUDA_ARCH_LIST'] == '9.0'\n"
        f"(Path(sys.argv[-1]) / {wheel_name!r}).write_bytes(b'pinned build')\n"
    )
    git("add", "setup.py")
    git("commit", "-m", "pinned source")
    revision = git("rev-parse", "HEAD")
    (source / "setup.py").write_text("raise AssertionError('Built unpinned HEAD')\n")
    git("commit", "-am", "newer source must not be built")
    monkeypatch.setitem(
        backend._META, "tool", {"deepep": {"url": str(source), "ref": revision}}
    )
    monkeypatch.setenv("HYBRID_EP_MULTINODE", "0")
    monkeypatch.setenv("USE_NIXL", "1")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    output = tmp_path / "wheels"
    assert backend.build_wheel(output) == wheel_name
    assert (output / wheel_name).read_bytes() == b"pinned build"
