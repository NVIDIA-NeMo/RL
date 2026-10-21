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
from typing import cast
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.lm_value import Value


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():
    monitor = MagicMock()
    monitor._collect_metrics.return_value = {}
    yield monitor


@pytest.fixture(scope="session", autouse=True)
def session_data(_unit_test_data):
    yield _unit_test_data


def test_value_rejects_policy_only_deterministic_option():
    config = cast(
        ValueConfig,
        {
            "megatron_cfg": {"enabled": False},
            "automodel_cfg": {"enabled": True, "deterministic": True},
        },
    )

    with pytest.raises(
        ValueError,
        match=r"value\.automodel_cfg\.deterministic is not supported",
    ):
        Value(cluster=MagicMock(), config=config, tokenizer=MagicMock())
