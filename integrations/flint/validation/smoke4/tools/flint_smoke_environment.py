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
"""Synthetic response-dependent reward for integration testing, not model quality."""

import hashlib
import json

import ray
import torch

from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


@ray.remote  # pragma: no cover
class FlintSmokeEnvironment(EnvironmentInterface):
    def __init__(self, cfg):
        self.cfg = cfg

    def shutdown(self):
        pass

    def step(self, message_log_batch, metadata, return_extracted_answer=False):
        responses = [
            "".join(str(m["content"]) for m in conversation if m["role"] == "assistant")
            for conversation in message_log_batch
        ]
        # Stable reward in [0,1), determined solely by each actual generated response.
        scores = [
            int.from_bytes(hashlib.sha256(s.encode()).digest()[:6], "big") / 2**48
            for s in responses
        ]
        print(
            "FLINT_SYNTHETIC_REWARDS "
            + json.dumps({"responses": responses, "rewards": scores}),
            flush=True,
        )
        rewards = torch.tensor(scores, dtype=torch.float32)
        return EnvironmentReturn(
            observations=[
                {"role": "environment", "content": "Synthetic integration reward"}
                for _ in responses
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(responses),
            rewards=rewards,
            terminateds=torch.ones_like(rewards),
            answers=responses if return_extracted_answer else None,
        )

    def global_post_process_and_metrics(self, batch):
        rewards = batch["rewards"].float()
        return batch, {
            "synthetic_reward_mean": rewards.mean().item(),
            "synthetic_reward_std": rewards.std(unbiased=False).item(),
        }
