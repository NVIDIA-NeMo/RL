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
"""Standard GRPO setup rejects ``policy.sequence_packing.share_prefixes``."""

from typing import Any
from unittest.mock import patch

import pytest

from nemo_rl.algorithms.grpo import MasterConfig, setup


def test_setup_rejects_share_prefixes() -> None:
    master_config = MasterConfig.model_construct(
        policy={"sequence_packing": {"enabled": True, "share_prefixes": True}}
    )

    with pytest.raises(
        ValueError, match="share_prefixes needs the single-controller GRPO path"
    ):
        setup(master_config, tokenizer=None, dataset=None, val_dataset=None)


@pytest.mark.parametrize(
    "policy_config",
    [
        {},
        {"sequence_packing": {"enabled": True}},
        {"sequence_packing": {"enabled": True, "share_prefixes": False}},
    ],
    ids=["no sequence_packing block", "key absent", "key false"],
)
def test_setup_continues_when_share_prefixes_is_off(
    policy_config: dict[str, Any],
) -> None:
    class SetupContinued(Exception):
        pass

    master_config = MasterConfig.model_construct(policy=policy_config)

    with (
        patch("nemo_rl.algorithms.grpo.time.perf_counter", side_effect=SetupContinued),
        pytest.raises(SetupContinued),
    ):
        setup(master_config, tokenizer=None, dataset=None, val_dataset=None)
