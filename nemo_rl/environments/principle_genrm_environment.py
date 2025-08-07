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

import logging
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class PrincipleGenrmEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env


class PrincipleGenrmEnvironmentMetadata(TypedDict):
    ground_truth: str  # "Yes" or "No" - the correct answer for principle evaluation


@ray.remote  # pragma: no cover
class PrincipleGenrmVerifyWorker:
    def __init__(self) -> None:
        pass

    def calculate_rewards(
        self, 
        pred_responses: list[str], 
        metadata_list: list[PrincipleGenrmEnvironmentMetadata],
        yes_logprobs: Optional[list[float]] = None,
        no_logprobs: Optional[list[float]] = None,
    ) -> list[float]:
        """Calculate rewards for principle_genrm based on logprobs and ground truth.

        Reward calculation:
        - If ground truth is "Yes": reward = logprob(" Yes") - logprob(" No") # speicial note on the whitespace
        - If ground truth is "No": reward = logprob(" No") - logprob(" Yes") # speicial note on the whitespace

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            metadata_list: list[PrincipleGenrmEnvironmentMetadata]. Contains ground truth.
            yes_logprobs: Optional list of Yes token logprobs from generation.
            no_logprobs: Optional list of No token logprobs from generation.

        Returns:
            list[float]. The rewards for each predicted response.
        """
        results = []
        
        for i, (response, metadata) in enumerate(zip(pred_responses, metadata_list)):
            ground_truth = metadata["ground_truth"]

            yes_logprob = yes_logprobs[i]
            no_logprob = no_logprobs[i]

            if yes_logprob == -float('inf') and no_logprob == -float('inf'):
                reward = -50.0
                results.append(reward)
                print(f"ground_truth: {ground_truth}, yes_logprob: {yes_logprob}, no_logprob: {no_logprob}, reward: {reward} (INVALID RESPONSE PENALTY)")
                continue
            
            # Calculate reward: logprob(correct) - logprob(incorrect)
            if ground_truth == "Yes":
                no_logprob = -50.0 if no_logprob == -float('inf') else no_logprob
                if yes_logprob == -float('inf'):
                    reward = -50.0
                else:
                    reward = yes_logprob - no_logprob
            elif ground_truth == "No":
                yes_logprob = -50.0 if yes_logprob == -float('inf') else yes_logprob
                if no_logprob == -float('inf'):
                    reward = -50.0
                else:
                    reward = no_logprob - yes_logprob
            else:
                raise ValueError(f"Invalid ground truth: {ground_truth}")

            print(f"ground_truth: {ground_truth}, yes_logprob: {yes_logprob}, no_logprob: {no_logprob}, reward: {reward}")
            results.append(reward)
            
        return results


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class PrincipleGenrmEnvironment(EnvironmentInterface):
    """Environment for training principle-based reward models.
    
    This environment evaluates whether LLM responses follow a given principle,
    using log probabilities of "Yes"/"No" tokens for precise reward calculation.
    """
    
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM
    
    def __init__(self, cfg: PrincipleGenrmEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            PrincipleGenrmVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": self.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[PrincipleGenrmEnvironmentMetadata],
        generation_data: Optional[BatchedDataDict] = None,
    ) -> EnvironmentReturn:
        """Runs a step in the principle_genrm environment.

        Args:
            message_log_batch: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs.
            metadata: list[PrincipleGenrmEnvironmentMetadata]. Contains ground truth "Yes"/"No".
            generation_data: Optional[BatchedDataDict]. Contains logprobs and other outputs from generation.
                           Expected to contain 'principle_genrm_yes_logprobs' and 'principle_genrm_no_logprobs'
                           when enable_principle_genrm_logprobs=True was used during generation.

        Returns:
            EnvironmentReturn: A tuple containing:
                - list[dict[str, str]]: Observations/responses batch
                - list[dict]: Updated metadata
                - list[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # EXTRACT ASSISTANT RESPONSES:
        # Get the last assistant message from each conversation for evaluation
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))
        
        print(f"example assistant response: {assistant_response_batch[0]}")
        # EXTRACT LOGPROBS FROM GENERATION OUTPUTS:
        # These come from the additional forward pass in vLLM when enable_principle_genrm_logprobs=True
        yes_logprobs = None
        no_logprobs = None
        if generation_data is not None:
            if "principle_genrm_yes_logprobs" in generation_data:
                yes_logprobs = generation_data["principle_genrm_yes_logprobs"].tolist()
            if "principle_genrm_no_logprobs" in generation_data:
                no_logprobs = generation_data["principle_genrm_no_logprobs"].tolist()
        assert generation_data is not None, "generation_data is required for principle_genrm environment"
        assert yes_logprobs is not None, "yes_logprobs is required for principle_genrm environment"
        assert no_logprobs is not None, "no_logprobs is required for principle_genrm environment"

        # PARALLEL PROCESSING SETUP:
        # Distribute work across multiple worker processes for efficiency
        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_metadata = chunk_list_to_workers(metadata, self.num_workers)

        # Chunk logprobs to match worker distribution
        chunked_yes_logprobs = None
        chunked_no_logprobs = None
        if yes_logprobs is not None:
            chunked_yes_logprobs = chunk_list_to_workers(yes_logprobs, self.num_workers)
        if no_logprobs is not None:
            chunked_no_logprobs = chunk_list_to_workers(no_logprobs, self.num_workers)

        # PARALLEL REWARD CALCULATION:
        # Each worker calculates rewards for its assigned chunk
        futures = []
        for i in range(self.num_workers):
            chunk_yes = chunked_yes_logprobs[i] if chunked_yes_logprobs else None
            chunk_no = chunked_no_logprobs[i] if chunked_no_logprobs else None
            
            future = self.workers[i].calculate_rewards.remote(
                chunked_assistant_response_batch[i],
                chunked_metadata[i],
                chunk_yes,
                chunk_no,
            )
            futures.append(future)

        results = ray.get(futures)

        # COMBINE RESULTS:
        # Flatten results from all workers back into a single list
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "Environment: principle evaluation complete, reward: " + str(results[i])
            }
            for i in range(len(results))
        ]

        # CREATE TENSORS FOR RL FRAMEWORK:
        # Convert rewards to tensors and mark all episodes as terminated (single-turn evaluation)
        rewards = torch.tensor(results, dtype=torch.float32).cpu()
        done = torch.ones_like(rewards, dtype=torch.bool).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Computes metrics for this environment given a global rollout batch."""

        # Calculate prediction accuracy from rewards
        # Since reward = logprob(correct) - logprob(incorrect), positive reward = correct prediction
        rewards = batch["total_reward"]
        
        total_predictions = len(rewards)
        correct_predictions = (rewards > 0).sum().item()
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            "principle_genrm_accuracy": accuracy
        }

        return batch, metrics 