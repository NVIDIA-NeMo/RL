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

"""Unit tests for the ``load_response_dataset`` and
``load_preference_dataset`` dispatchers plus the shared
``resolve_external_dataset_class`` helper in
``nemo_rl.data.datasets.utils`` — covers both built-in registry lookup
and dotted-path import of user-defined dataset classes (see GitHub
issue #1020).
"""

from __future__ import annotations

import sys
import types

import pytest

from nemo_rl.data.datasets.preference_datasets import (
    DATASET_REGISTRY as PREFERENCE_REGISTRY,
)
from nemo_rl.data.datasets.preference_datasets import (
    load_preference_dataset,
)
from nemo_rl.data.datasets.response_datasets import (
    DATASET_REGISTRY as RESPONSE_REGISTRY,
)
from nemo_rl.data.datasets.response_datasets import (
    load_response_dataset,
)
from nemo_rl.data.datasets.utils import resolve_external_dataset_class


class _StubResponseDataset:
    """Minimal stub satisfying the contract used by
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


class _StubPreferenceDataset:
    """Minimal stub satisfying the contract used by
    ``load_preference_dataset`` (constructor + ``set_task_spec``)."""

    last_init_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs
        self.task_spec_config = None

    def set_task_spec(self, data_config):
        self.task_spec_config = data_config


@pytest.fixture
def stub_module():
    """Install a throwaway module exposing the stub datasets for the
    duration of one test, then remove it from ``sys.modules``."""
    module_name = "nemo_rl_test_stub_external_module_for_1020"
    module = types.ModuleType(module_name)
    module.StubResponseDataset = _StubResponseDataset
    module.StubPreferenceDataset = _StubPreferenceDataset
    module.not_a_class = 42
    sys.modules[module_name] = module
    try:
        yield module_name
    finally:
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# resolve_external_dataset_class (shared helper in nemo_rl.data.datasets.utils)
# ---------------------------------------------------------------------------


def test_resolve_external_returns_class(stub_module):
    cls = resolve_external_dataset_class(f"{stub_module}.StubResponseDataset")
    assert cls is _StubResponseDataset


def test_resolve_external_rejects_unimportable_module():
    with pytest.raises(ValueError, match="Could not import module"):
        resolve_external_dataset_class("nemo_rl_does_not_exist_xyz.SomeDataset")


def test_resolve_external_rejects_missing_attribute(stub_module):
    with pytest.raises(ValueError, match="has no attribute 'Missing'"):
        resolve_external_dataset_class(f"{stub_module}.Missing")


def test_resolve_external_rejects_non_class(stub_module):
    with pytest.raises(ValueError, match="which is not a class"):
        resolve_external_dataset_class(f"{stub_module}.not_a_class")


# ---------------------------------------------------------------------------
# load_response_dataset
# ---------------------------------------------------------------------------


def test_load_response_dataset_builtin_registry_present():
    """Built-in registry must still expose the loadable formats."""
    assert "ResponseDataset" in RESPONSE_REGISTRY


def test_load_response_dataset_uses_external_class(stub_module):
    config = {
        "dataset_name": f"{stub_module}.StubResponseDataset",
        "data_path": "/tmp/does-not-need-to-exist.jsonl",
    }
    dataset = load_response_dataset(config)

    assert isinstance(dataset, _StubResponseDataset)
    assert _StubResponseDataset.last_init_kwargs == config
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


# ---------------------------------------------------------------------------
# load_preference_dataset
# ---------------------------------------------------------------------------


def test_load_preference_dataset_builtin_registry_present():
    """Built-in registry must still expose the loadable formats."""
    assert "PreferenceDataset" in PREFERENCE_REGISTRY
    assert "BinaryPreferenceDataset" in PREFERENCE_REGISTRY


def test_load_preference_dataset_uses_external_class(stub_module):
    config = {
        "dataset_name": f"{stub_module}.StubPreferenceDataset",
        "data_path": "/tmp/does-not-need-to-exist.jsonl",
    }
    dataset = load_preference_dataset(config)

    assert isinstance(dataset, _StubPreferenceDataset)
    assert _StubPreferenceDataset.last_init_kwargs == config
    assert dataset.task_spec_config == config


def test_load_preference_dataset_unknown_bare_name_errors():
    config = {"dataset_name": "definitely_not_in_registry"}
    with pytest.raises(ValueError, match="Unsupported dataset_name"):
        load_preference_dataset(config)


def test_load_preference_dataset_bad_dotted_path_errors():
    config = {"dataset_name": "nemo_rl_missing_module.MyDataset"}
    with pytest.raises(ValueError, match="Could not import module"):
        load_preference_dataset(config)
