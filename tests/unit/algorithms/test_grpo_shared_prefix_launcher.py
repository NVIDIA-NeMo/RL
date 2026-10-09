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
"""Standard GRPO must reject sharing before allocating policy resources."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nemo_rl.algorithms.grpo import setup


@pytest.mark.parametrize("mode", ["logprobs", "train"])
def test_standard_grpo_rejects_missing_metadata_producer(mode):
    # The deliberately minimal config ensures the guard runs before other setup.
    config = SimpleNamespace(policy={"shared_prefix_training": {"mode": mode}})
    with pytest.raises(
        ValueError, match="SingleController.*token_capture.enabled=true"
    ):
        setup(config, tokenizer=None, dataset=None, val_dataset=None)


@pytest.mark.parametrize("mode", [None, "disabled", "dense"])
def test_standard_grpo_keeps_nonshared_setup_path(mode):
    policy = {} if mode is None else {"shared_prefix_training": {"mode": mode}}
    config = SimpleNamespace(policy=policy)

    class SetupReached(Exception):
        pass

    with (
        patch("nemo_rl.algorithms.grpo.time.perf_counter", side_effect=SetupReached),
        pytest.raises(SetupReached),
    ):
        setup(config, tokenizer=None, dataset=None, val_dataset=None)
