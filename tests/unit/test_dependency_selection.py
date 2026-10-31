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
"""Selection contracts for the shared vLLM + Gym shard."""

import sys
from types import SimpleNamespace

import pytest

from tests.unit.conftest import pytest_collection_modifyitems, pytest_sessionfinish


def config_for(*flags):
    return SimpleNamespace(getoption=lambda name, default=False: name in flags)


@pytest.fixture
def dependency_items(request, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nemo_gym", SimpleNamespace(config_types=object()))
    items = []
    for name, markers in [
        ("base", []),
        ("vllm", ["vllm"]),
        ("gym", ["nemo_gym"]),
        ("both", ["vllm", "nemo_gym"]),
        ("gated", ["vllm", "nemo_gym", "hf_gated"]),
        ("third", ["vllm", "nemo_gym", "mcore"]),
    ]:
        item = pytest.Function.from_parent(
            request.node.parent, name=name, callobj=lambda: None
        )
        for marker in markers:
            item.add_marker(marker)
        items.append(item)
    return items


@pytest.mark.parametrize(
    "flags,expected",
    [
        ((), ["base"]),
        (("--vllm-only",), ["vllm"]),
        (("--nemo-gym-only",), ["gym"]),
        (("--vllm-only", "--nemo-gym-only"), ["both"]),
        (("--vllm-only", "--nemo-gym-only", "--hf-gated"), ["both", "gated"]),
    ],
)
def test_dependency_lane_selection(dependency_items, flags, expected):
    pytest_collection_modifyitems(config_for(*flags), dependency_items)
    assert [item.name for item in dependency_items] == expected


@pytest.mark.parametrize(
    "flags",
    [
        ("--vllm-only", "--mcore-only"),
        ("--nemo-gym-only", "--automodel-only"),
        ("--vllm-only", "--nemo-gym-only", "--sglang-only"),
    ],
)
def test_other_dependency_combinations_stay_invalid(flags):
    with pytest.raises(ValueError, match="mutually exclusive"):
        pytest_collection_modifyitems(config_for(*flags), [])


@pytest.mark.parametrize("dependency", ["vllm", "nemo_gym"])
def test_common_lane_requires_both_dependencies(
    dependency_items, monkeypatch, dependency
):
    monkeypatch.setitem(sys.modules, dependency, None)
    with pytest.raises(ImportError, match=f"Cannot run {dependency} tests"):
        pytest_collection_modifyitems(
            config_for("--vllm-only", "--nemo-gym-only"), dependency_items
        )


@pytest.mark.parametrize("combined", [False, True])
def test_empty_common_lane_cannot_be_silently_accepted(combined):
    config = config_for("--vllm-only", "--nemo-gym-only") if combined else config_for()
    session = SimpleNamespace(
        config=config, exitstatus=pytest.ExitCode.NO_TESTS_COLLECTED
    )
    pytest_sessionfinish(session, session.exitstatus)
    assert session.exitstatus == (
        pytest.ExitCode.USAGE_ERROR if combined else pytest.ExitCode.NO_TESTS_COLLECTED
    )
