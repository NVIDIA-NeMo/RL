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

import pytest
from pydantic import ValidationError

from nemo_rl.algorithms.single_controller_utils.config import RolloutCheckpointConfig
from nemo_rl.experience.rollout_recovery import RetryScope


def test_retry_scope_override_is_typed() -> None:
    config = RolloutCheckpointConfig(
        default_retry_scope="sibling",
        retry_scope_overrides={"genrm_agent": "prompt_group"},
    )

    assert config.default_retry_scope == RetryScope.SIBLING
    assert config.retry_scope_overrides == {
        "genrm_agent": RetryScope.PROMPT_GROUP,
    }


def test_unknown_retry_scope_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RolloutCheckpointConfig(
            retry_scope_overrides={"genrm_agent": "unknown"},
        )
