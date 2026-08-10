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

import os
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from nemo_rl.models.generation.profiling import load_rollout_profiler


class _FakeRolloutProfiler:
    def __init__(self, *, rank: int) -> None:
        self.rank = rank

    def begin_engine_initialization(self) -> object:
        return object()

    def end_engine_initialization(self, token: object) -> None:
        pass

    def begin_rollout(self, *, step_id: int | str) -> None:
        pass

    def finish_rollout(self) -> None:
        pass

    def abort_rollout(self, *, reason: str) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def fake_profiler_module():
    module_name = "nemo_rl_test_rollout_profiler"
    module = ModuleType(module_name)
    module.FakeRolloutProfiler = _FakeRolloutProfiler
    module.not_a_class = object()
    sys.modules[module_name] = module
    try:
        yield module_name
    finally:
        sys.modules.pop(module_name, None)


def test_rollout_profiler_is_disabled_without_class_path():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "nemo_rl.models.generation.profiling.importlib.import_module"
        ) as importer,
    ):
        assert load_rollout_profiler(rank=3) is None

    importer.assert_not_called()


def test_rollout_profiler_loads_class_with_generation_rank(fake_profiler_module):
    class_path = f"{fake_profiler_module}.FakeRolloutProfiler"
    with patch.dict(os.environ, {"NRL_ROLLOUT_PROFILER_CLASS": class_path}):
        profiler = load_rollout_profiler(rank=5)

    assert isinstance(profiler, _FakeRolloutProfiler)
    assert profiler.rank == 5


@pytest.mark.parametrize("class_path", ["Profiler", "module.", ".Profiler"])
def test_rollout_profiler_rejects_malformed_class_path(class_path):
    with (
        patch.dict(os.environ, {"NRL_ROLLOUT_PROFILER_CLASS": class_path}),
        pytest.raises(ValueError, match="fully qualified class path"),
    ):
        load_rollout_profiler(rank=0)


def test_rollout_profiler_fails_loudly_when_module_is_missing():
    class_path = "nemo_rl_missing_rollout_profiler.Profiler"
    with (
        patch.dict(os.environ, {"NRL_ROLLOUT_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="vLLM generation-worker environment"),
    ):
        load_rollout_profiler(rank=0)


@pytest.mark.parametrize("attribute", ["Missing", "not_a_class"])
def test_rollout_profiler_rejects_attribute_that_is_not_a_class(
    fake_profiler_module, attribute
):
    class_path = f"{fake_profiler_module}.{attribute}"
    with (
        patch.dict(os.environ, {"NRL_ROLLOUT_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="does not resolve to a class"),
    ):
        load_rollout_profiler(rank=0)


def test_rollout_profiler_rejects_class_with_missing_methods(
    fake_profiler_module,
):
    class IncompleteProfiler:
        def __init__(self, *, rank: int) -> None:
            pass

    module = sys.modules[fake_profiler_module]
    module.IncompleteProfiler = IncompleteProfiler
    class_path = f"{fake_profiler_module}.IncompleteProfiler"
    with (
        patch.dict(os.environ, {"NRL_ROLLOUT_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="missing required method"),
    ):
        load_rollout_profiler(rank=0)


def test_rollout_profiler_reports_constructor_failure(fake_profiler_module):
    class BrokenProfiler(_FakeRolloutProfiler):
        def __init__(self, *, rank: int) -> None:
            raise ValueError("bad configuration")

    module = sys.modules[fake_profiler_module]
    module.BrokenProfiler = BrokenProfiler
    class_path = f"{fake_profiler_module}.BrokenProfiler"
    with (
        patch.dict(os.environ, {"NRL_ROLLOUT_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="Failed to initialize"),
    ):
        load_rollout_profiler(rank=0)
