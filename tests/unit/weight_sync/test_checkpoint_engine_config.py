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

"""Tests for checkpoint-engine configuration selection."""

import pytest

from nemo_rl.weight_sync.checkpoint_engine_config import (
    enabled_checkpoint_engine_config,
)


@pytest.mark.parametrize(
    "generation_config",
    [
        {},
        {"checkpoint_engine": None},
        {"checkpoint_engine": {"enabled": False}},
    ],
)
def test_enabled_checkpoint_engine_config_returns_none_when_disabled(
    generation_config,
):
    assert enabled_checkpoint_engine_config(generation_config) is None


def test_enabled_checkpoint_engine_config_returns_enabled_config():
    checkpoint_engine = {"enabled": True, "backend": "nixl"}

    assert (
        enabled_checkpoint_engine_config({"checkpoint_engine": checkpoint_engine})
        is checkpoint_engine
    )
