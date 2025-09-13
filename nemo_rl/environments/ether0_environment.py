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

from typing import Dict, List, Optional, Tuple, TypedDict
from collections import defaultdict

import ray
import torch

from ether0.rewards import EVAL_FUNCTIONS
from ether0.model_prompts import extract_answer_loose

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class Ether0EnvConfig(TypedDict):
    num_workers: int


class Ether0EnvironmentMetadata(TypedDict):
    reward_function_info: Dict[str, str]  # Contains fxn_name, answer_info, problem_type
    problem_type: str
    ideal_answer: Optional[str]
    original_id: Optional[str]


@ray.remote
class Ether0EvaluatorWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.ETHER0

    def __init__(self):
        pass

    def evaluate_responses(
        self, 
        responses: List[str], 
        metadata_list: List[Ether0EnvironmentMetadata]
    ) -> List[Tuple[float, str]]:
        """Use ether0 reward functions.
        
        Returns:
            List of (reward, reason) tuples
        """
        results = []
        
        for response, metadata in zip(responses, metadata_list):
            try:
                extracted_answer = extract_answer_loose(response)
                
                if extracted_answer is None:
                    reward = 0.0
                    reason = "Failed to extract answer"
                else:
                    try:
                        reward_info = metadata["reward_function_info"]
                        fxn_name = reward_info["fxn_name"]
                        answer_info = reward_info["answer_info"]
                        
                        reward_fn = EVAL_FUNCTIONS[fxn_name]
                        eval_metadata = {}
                        
                        reward = reward_fn(
                            yhat=extracted_answer,
                            y=answer_info,
                            test=True,
                            metadata=eval_metadata
                        )
                        reason = eval_metadata.get("reward_reason", "success")
                        
                    except Exception as e:
                        reward = 0.0
                        reason = f"Error in reward function: {str(e)}"
                        
            except Exception as e:
                reward = 0.0
                reason = f"Error during evaluation: {str(e)}"
                
            results.append((reward, reason))
            
        return results


@ray.remote
class Ether0Environment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.ETHER0

    def __init__(self, cfg: Ether0EnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            Ether0EvaluatorWorker.options(
                runtime_env={
                    "py_executable": Ether0EvaluatorWorker.DEFAULT_PY_EXECUTABLE,
                    "excludes": ["*.jsonl", "*.json", "results/", "logs/", "__pycache__/", ".git/"]
                }
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Ether0EnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the ether0 environment.

        Args:
            message_log_batch: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs.
            metadata: List[Ether0EnvironmentMetadata]. Contains reward function info for evaluation.

        Returns:
            EnvironmentReturn: A tuple containing observations, metadata, stop strings, rewards, and terminateds.
        """
        # get assistant msgs
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        chunked_responses = chunk_list_to_workers(assistant_response_batch, self.num_workers)
        chunked_metadata = chunk_list_to_workers(metadata, self.num_workers)

        futures = [
            self.workers[i].evaluate_responses.remote(response_chunk, metadata_chunk)
            for i, (response_chunk, metadata_chunk) in enumerate(
                zip(chunked_responses, chunked_metadata)
            )
        ]

        results = ray.get(futures)

        flat_results = [item for sublist in results for item in sublist]
        rewards = [result[0] for result in flat_results]
        reasons = [result[1] for result in flat_results]

        observations = [
            {
                "role": "environment",
                "content": f"Environment: {'correct' if reward > 0 else 'incorrect'} ({reason})",
            }
            for reward, reason in zip(rewards, reasons)
        ]

        rewards_tensor = torch.tensor(rewards).cpu()
        done_tensor = torch.ones_like(rewards_tensor).cpu()  

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=done_tensor,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        
        rewards = batch["total_reward"]
        problem_types = batch.get("problem_type", [])
        
        accuracy = rewards.mean().item()
        success_rate = (rewards > 0).float().mean().item()
        
        metrics = {
            "accuracy": accuracy,
            "success_rate": success_rate,
            "total_samples": len(rewards),
        }
        
        if hasattr(batch, 'idx'):
            pass_rates = calculate_pass_rate_per_prompt(
                batch["idx"], rewards, num_generations=1 
            )
            metrics.update(pass_rates)
        
        if problem_types:
            rewards_by_type = defaultdict(list)
            for reward, ptype in zip(rewards, problem_types):
                main_task = ptype.split('/')[0] if isinstance(ptype, str) else str(ptype)
                rewards_by_type[main_task].append(reward.item())
            
            for task, task_rewards in rewards_by_type.items():
                task_accuracy = sum(task_rewards) / len(task_rewards)
                task_success_rate = sum(1 for r in task_rewards if r > 0) / len(task_rewards)
                metrics[f"{task}_accuracy"] = task_accuracy
                metrics[f"{task}_success_rate"] = task_success_rate
                metrics[f"{task}_count"] = len(task_rewards)
        
        return batch, metrics