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

from nemo_rl.models.policy.profiling import load_policy_profiler


class _FakePolicyProfiler:
    def __init__(self, *, rank: int) -> None:
        self.rank = rank

    def begin_train_step(self) -> None:
        pass

    def finish_train_step(self) -> None:
        pass

    def abort_train_step(self, *, reason: str) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def fake_profiler_module():
    module_name = "nemo_rl_test_policy_profiler"
    module = ModuleType(module_name)
    module.FakePolicyProfiler = _FakePolicyProfiler
    module.not_a_class = object()
    sys.modules[module_name] = module
    try:
        yield module_name
    finally:
        sys.modules.pop(module_name, None)


def test_profiler_is_disabled_without_class_path():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("nemo_rl.models.policy.profiling.importlib.import_module") as importer,
    ):
        assert load_policy_profiler(rank=3) is None

    importer.assert_not_called()


def test_profiler_loads_class_with_distributed_rank(fake_profiler_module):
    class_path = f"{fake_profiler_module}.FakePolicyProfiler"
    with patch.dict(os.environ, {"NRL_POLICY_PROFILER_CLASS": class_path}):
        profiler = load_policy_profiler(rank=5)

    assert isinstance(profiler, _FakePolicyProfiler)
    assert profiler.rank == 5


@pytest.mark.parametrize(
    "class_path",
    ["Profiler", "module.", ".Profiler"],
)
def test_profiler_rejects_malformed_class_path(class_path):
    with (
        patch.dict(os.environ, {"NRL_POLICY_PROFILER_CLASS": class_path}),
        pytest.raises(ValueError, match="fully qualified class path"),
    ):
        load_policy_profiler(rank=0)


def test_profiler_fails_loudly_when_module_is_missing():
    class_path = "nemo_rl_missing_policy_profiler.Profiler"
    with (
        patch.dict(os.environ, {"NRL_POLICY_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="policy-worker environment"),
    ):
        load_policy_profiler(rank=0)


@pytest.mark.parametrize("attribute", ["Missing", "not_a_class"])
def test_profiler_rejects_attribute_that_is_not_a_class(
    fake_profiler_module, attribute
):
    class_path = f"{fake_profiler_module}.{attribute}"
    with (
        patch.dict(os.environ, {"NRL_POLICY_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="does not resolve to a class"),
    ):
        load_policy_profiler(rank=0)


def test_profiler_rejects_class_with_missing_methods(fake_profiler_module):
    class IncompleteProfiler:
        def __init__(self, *, rank: int) -> None:
            pass

    module = sys.modules[fake_profiler_module]
    module.IncompleteProfiler = IncompleteProfiler
    class_path = f"{fake_profiler_module}.IncompleteProfiler"
    with (
        patch.dict(os.environ, {"NRL_POLICY_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="missing required method"),
    ):
        load_policy_profiler(rank=0)


def test_profiler_reports_constructor_failure(fake_profiler_module):
    class BrokenProfiler(_FakePolicyProfiler):
        def __init__(self, *, rank: int) -> None:
            raise ValueError("bad configuration")

    module = sys.modules[fake_profiler_module]
    module.BrokenProfiler = BrokenProfiler
    class_path = f"{fake_profiler_module}.BrokenProfiler"
    with (
        patch.dict(os.environ, {"NRL_POLICY_PROFILER_CLASS": class_path}),
        pytest.raises(RuntimeError, match="Failed to initialize"),
    ):
        load_policy_profiler(rank=0)
