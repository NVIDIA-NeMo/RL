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

import ast
import json
from contextlib import contextmanager

import pytest

from nemo_rl.models.generation.trtllm import patches

pytestmark = pytest.mark.trtllm

MARKER = "# --- nemo-rl IFB metric patch ---"

# Minimal stand-in for TRT-LLM's executor/base_worker.py. The patch is a pure
# append that binds a method onto whatever `BaseWorker` is in module scope, so
# a synthetic module with the same shape exercises it faithfully.
_FAKE_BASE_WORKER = """
import json


class BaseWorker:
    def __init__(self, stats):
        self._stats = stats

    def fetch_stats(self):
        return self._stats

    def _stats_serializer(self, stat):
        return json.dumps(stat)
"""


class _StubLogger:
    """Collects logger calls so tests can assert on warnings without caplog."""

    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def _fmt(self, msg, args):
        return msg % args if args else msg

    def info(self, msg, *args):
        self.infos.append(self._fmt(msg, args))

    def warning(self, msg, *args):
        self.warnings.append(self._fmt(msg, args))


@pytest.fixture
def fake_base_worker(tmp_path, monkeypatch):
    """Point `_get_trtllm_file` at a synthetic base_worker.py under tmp_path."""
    path = tmp_path / "base_worker.py"
    path.write_text(_FAKE_BASE_WORKER)
    monkeypatch.setattr(patches, "_get_trtllm_file", lambda _relative: str(path))
    return path


def test_fetch_stats_patch_appends_once_and_installs_working_method(
    fake_base_worker,
):
    logger = _StubLogger()

    # Co-located replica actors share the venv, so the patch runs more than once
    # against the same file; the second call must be a no-op.
    patches._patch_trtllm_fetch_stats_serialized(logger)
    patches._patch_trtllm_fetch_stats_serialized(logger)

    source = fake_base_worker.read_text()
    assert source.count(MARKER) == 1
    assert logger.warnings == []

    # The appended block must leave the module parseable and importable.
    ast.parse(source)
    namespace: dict = {}
    exec(compile(source, str(fake_base_worker), "exec"), namespace)

    stats = [{"iter": 0, "numQueuedRequests": 2}]
    worker = namespace["BaseWorker"](stats)
    serialized = worker.fetch_stats_serialized()

    # Only list[str] survives the Ray RPC pickle round-trip — the C++ stats
    # bindings do not, which is the entire reason this patch exists.
    assert isinstance(serialized, list)
    assert all(isinstance(item, str) for item in serialized)
    assert [json.loads(item) for item in serialized] == stats


def test_fetch_stats_patch_warns_when_write_does_not_persist(
    fake_base_worker, monkeypatch
):
    @contextmanager
    def _no_op_write(file_path):
        with open(file_path) as handle:
            content = handle.read()

        def write_back(_new_content: str) -> None:
            """Simulate a write that silently fails to land (e.g. read-only venv)."""

        yield content, write_back

    monkeypatch.setattr(patches, "_locked_file_patch", _no_op_write)

    logger = _StubLogger()
    patches._patch_trtllm_fetch_stats_serialized(logger)

    assert MARKER not in fake_base_worker.read_text()
    assert len(logger.warnings) == 1
    assert str(fake_base_worker) in logger.warnings[0]
    assert "did not persist" in logger.warnings[0]
