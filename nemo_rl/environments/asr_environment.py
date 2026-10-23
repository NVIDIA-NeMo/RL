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

"""Single-turn CPU transcript reward environment."""

from typing import Any

import ray
import torch
from pydantic import BaseModel, Field

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.asr_reward import asr_reward
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers


class ASREnvConfig(BaseModel):
    num_workers: int = Field(default=1, ge=1)


@ray.remote
class ASRRewardWorker:  # pragma: no cover
    def score(
        self, hypotheses: list[str], metadata: list[dict[str, Any]]
    ) -> list[float]:
        if len(hypotheses) != len(metadata):
            raise ValueError("ASR hypotheses and metadata must have matching lengths")
        return [
            asr_reward(info["ground_truth"], hyp, info)
            for hyp, info in zip(hypotheses, metadata)
        ]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ASREnvironment(EnvironmentInterface):  # pragma: no cover
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = ASREnvConfig(**cfg)
        self.workers = [
            ASRRewardWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.cfg.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[list[dict[str, Any]]],
        metadata: list[dict[str, Any]],
    ) -> EnvironmentReturn:
        hypotheses = [
            "".join(str(m["content"]) for m in messages if m["role"] == "assistant")
            for messages in message_log_batch
        ]
        chunks = chunk_list_to_workers(hypotheses, self.cfg.num_workers)
        info_chunks = chunk_list_to_workers(metadata, self.cfg.num_workers)
        results = ray.get(
            [
                worker.score.remote(chunk, info)
                for worker, chunk, info in zip(self.workers, chunks, info_chunks)
            ]
        )
        scores = [value for chunk in results for value in chunk]
        rewards = torch.tensor(scores, dtype=torch.float32)
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""} for _ in scores],
            metadata=metadata,
            next_stop_strings=[None] * len(scores),
            rewards=rewards,
            terminateds=torch.ones_like(rewards),
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        batch["rewards"] = batch["rewards"] * batch["is_end"]
        rewards = batch["rewards"]
        return batch, {
            "asr_reward_mean": rewards.mean().item(),
            "asr_reward_min": rewards.min().item(),
            "asr_reward_max": rewards.max().item(),
            "num_problems_in_batch": rewards.numel(),
        }
