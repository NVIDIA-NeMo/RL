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

from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.grpo import (
    _raise_if_async_grpo_discard_weights_on_sleep_enabled,
)
from nemo_rl.algorithms.grpo_sync import _raise_if_discard_weights_on_sleep_enabled


@pytest.mark.parametrize(
    "validation",
    [
        _raise_if_discard_weights_on_sleep_enabled,
        _raise_if_async_grpo_discard_weights_on_sleep_enabled,
    ],
)
def test_unsupported_trainers_reject_discard_weights_on_sleep(validation) -> None:
    master_config = MagicMock()
    master_config.policy = {
        "generation": {
            "backend": "vllm",
            "vllm_cfg": {"discard_weights_on_sleep": True},
        }
    }

    with pytest.raises(NotImplementedError, match="#4659"):
        validation(master_config)
