# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the ``load_response_dataset`` dispatch in
``nemo_rl.data.datasets.response_datasets`` — covers both built-in
registry lookup and dotted-path import of user-defined dataset classes
(see GitHub issue #1020).
"""

from __future__ import annotations

import sys
import types

import pytest

from nemo_rl.data.datasets.response_datasets import (
    DATASET_REGISTRY,
    _resolve_external_dataset_class,
    load_response_dataset,
)


class _StubDataset:
    """Minimal dataset stub satisfying the contract used by
    ``load_response_dataset`` (constructor + ``set_task_spec`` +
    ``set_processor``)."""

    last_init_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs
        self.task_spec_config = None
        self.processor_set = False

    def set_task_spec(self, data_config):
        self.task_spec_config = data_config

    def set_processor(self):
        self.processor_set = True


@pytest.fixture
def stub_module():
    """Install a throwaway module exposing ``StubDataset`` for the duration
    of one test, then remove it from ``sys.modules``."""
    module_name = "nemo_rl_test_stub_external_module_for_1020"
    module = types.ModuleType(module_name)
    module.StubDataset = _StubDataset
    module.not_a_class = 42
    sys.modules[module_name] = module
    try:
        yield module_name
    finally:
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# _resolve_external_dataset_class
# ---------------------------------------------------------------------------


def test_resolve_external_returns_class(stub_module):
    cls = _resolve_external_dataset_class(f"{stub_module}.StubDataset")
    assert cls is _StubDataset


def test_resolve_external_rejects_bare_name():
    with pytest.raises(ValueError, match="Unsupported dataset_name"):
        _resolve_external_dataset_class("totally_unknown_dataset")


def test_resolve_external_rejects_unimportable_module():
    with pytest.raises(ValueError, match="Could not import module"):
        _resolve_external_dataset_class("nemo_rl_does_not_exist_xyz.SomeDataset")


def test_resolve_external_rejects_missing_attribute(stub_module):
    with pytest.raises(ValueError, match="has no attribute 'Missing'"):
        _resolve_external_dataset_class(f"{stub_module}.Missing")


def test_resolve_external_rejects_non_class(stub_module):
    with pytest.raises(ValueError, match="which is not a class"):
        _resolve_external_dataset_class(f"{stub_module}.not_a_class")


# ---------------------------------------------------------------------------
# load_response_dataset
# ---------------------------------------------------------------------------


def test_load_response_dataset_builtin_unchanged():
    """Built-in registry names must still resolve via DATASET_REGISTRY."""
    # The registry should always contain at least the loadable formats.
    assert "ResponseDataset" in DATASET_REGISTRY


def test_load_response_dataset_uses_external_class(stub_module):
    config = {
        "dataset_name": f"{stub_module}.StubDataset",
        "data_path": "/tmp/does-not-need-to-exist.jsonl",
    }
    dataset = load_response_dataset(config)

    assert isinstance(dataset, _StubDataset)
    assert _StubDataset.last_init_kwargs == config
    # Lifecycle methods are exercised by the dispatcher.
    assert dataset.task_spec_config == config
    assert dataset.processor_set is True


def test_load_response_dataset_unknown_bare_name_errors():
    config = {"dataset_name": "definitely_not_in_registry"}
    with pytest.raises(ValueError, match="Unsupported dataset_name"):
        load_response_dataset(config)


def test_load_response_dataset_bad_dotted_path_errors():
    config = {"dataset_name": "nemo_rl_missing_module.MyDataset"}
    with pytest.raises(ValueError, match="Could not import module"):
        load_response_dataset(config)
