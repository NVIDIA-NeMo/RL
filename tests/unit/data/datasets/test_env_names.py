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
"""Unit tests for ``extract_necessary_env_names``.

The multi-dataloader path lets each ``data.train`` entry route to its own
``env_name``. This helper must collect env names from every entry so that
``setup_response_data`` creates each environment; a regression here surfaces
as ``KeyError`` when binding a task to an env that was never created.
"""

from nemo_rl.data.datasets.utils import extract_necessary_env_names


def test_list_form_collects_per_entry_env_names():
    """List-form train entries with distinct env_names are all collected."""
    data_config = {
        "train": [
            {"dataset_name": "gsm8k", "env_name": "math"},
            {"dataset_name": "DAPOMath17K", "env_name": "math_multi_reward"},
        ],
        "default": {"env_name": "math"},
    }
    assert set(extract_necessary_env_names(data_config)) == {
        "math",
        "math_multi_reward",
    }


def test_dict_form_single_dataset():
    """Legacy single-dataset (dict) form still resolves its env_name."""
    data_config = {
        "train": {"dataset_name": "gsm8k", "env_name": "math"},
        "default": {"env_name": "math"},
    }
    assert set(extract_necessary_env_names(data_config)) == {"math"}


def test_env_name_falls_back_to_default():
    """Entries without an explicit env_name are covered by ``default``."""
    data_config = {
        "train": [{"dataset_name": "gsm8k"}],
        "default": {"env_name": "math"},
    }
    assert set(extract_necessary_env_names(data_config)) == {"math"}


def test_validation_entries_contribute_env_names():
    """Validation-only envs are collected alongside train envs."""
    data_config = {
        "train": [{"dataset_name": "gsm8k", "env_name": "math"}],
        "validation": [{"dataset_name": "AIME2024", "env_name": "math_multi_reward"}],
        "default": {"env_name": "math"},
    }
    assert set(extract_necessary_env_names(data_config)) == {
        "math",
        "math_multi_reward",
    }


def test_none_values_are_ignored():
    """A ``None`` validation entry does not raise."""
    data_config = {
        "train": [{"dataset_name": "gsm8k", "env_name": "math"}],
        "validation": None,
        "default": {"env_name": "math"},
    }
    assert set(extract_necessary_env_names(data_config)) == {"math"}
