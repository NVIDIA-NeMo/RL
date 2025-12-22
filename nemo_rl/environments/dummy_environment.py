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

from typing import Any

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class DummyEnvironment(EnvironmentInterface):
    def _init__(self):
        pass

    def shutdown(self):
        pass

    def step(
        self, message_log_batch: list[LLMMessageLogType], metadata: list[Any], *args
    ) -> EnvironmentReturn:
        """Dummy environment step function. Always return 0 for reward."""
        observations = [
            {"role": "assistant", "content": "dummy content"} for _ in message_log_batch
        ]
        rewards = torch.zeros(len(message_log_batch))
        done = torch.ones_like(rewards)
        answers = [None] * len(message_log_batch)
        next_stop_strings = [None] * len(message_log_batch)
        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Dummy environment global post processing and metrics function. Always return empty dict for metrics."""
        metrics = {}
        return batch, metrics
