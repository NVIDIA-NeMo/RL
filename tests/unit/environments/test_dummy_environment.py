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

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.dummy_environment import DummyEnvironment


def test_dummy_environment_returns_zero_rewards_and_terminates():
    env_cls = DummyEnvironment.__ray_metadata__.modified_class
    env = env_cls()
    metadata = [{"sample": 0}, {"sample": 1}]

    result = env.step(
        message_log_batch=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        metadata=metadata,
    )

    assert result.observations == [
        {"role": "assistant", "content": "dummy content"},
        {"role": "assistant", "content": "dummy content"},
    ]
    torch.testing.assert_close(result.rewards, torch.zeros(2))
    torch.testing.assert_close(result.terminateds, torch.ones(2))
    assert result.metadata == metadata
    assert result.next_stop_strings == [None, None]
    assert result.answers == [None, None]


def test_dummy_environment_global_post_process_is_noop():
    env_cls = DummyEnvironment.__ray_metadata__.modified_class
    env = env_cls()
    batch = BatchedDataDict({"values": torch.tensor([1, 2, 3])})

    processed_batch, metrics = env.global_post_process_and_metrics(batch)

    assert processed_batch is batch
    assert metrics == {}
