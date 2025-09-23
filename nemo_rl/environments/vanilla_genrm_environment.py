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

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class VanillaGenRMConfig(TypedDict):
    """Configuration for Vanilla GenRM training environment."""

    num_workers: int  # Number of worker processes for parallel evaluation
    score_weight: float  # Weight for individual scores (default: 1.0)
    ranking_weight: float  # Weight for ranking scores (default: 1.0)
    reasoning_split_word: Optional[
        str
    ]  # Split word to separate reasoning from final output (default: "</think>")


class VanillaGenRMMetadata(TypedDict):
    """Metadata structure for vanilla GenRM environment."""

    question_id: Optional[str]
    num_responses: Optional[int]
    score_1: Optional[int]  # Individual helpfulness score for response 1
    score_2: Optional[int]  # Individual helpfulness score for response 2
    ranking: Optional[int]  # Ranking score (1-6 scale)
    question: Optional[str]


@ray.remote
class VanillaGenRMWorker:
    """Worker that evaluates GenRM responses and extracts scoring information."""

    def __init__(self):
        """Initialize the GenRM evaluation worker."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def extract_scores_from_response(self, response: str) -> Dict[str, Any]:
        """Extract scores from the model's JSON response.

        Args:
            response: The model's response containing JSON with scores (already split if needed)

        Returns:
            Dictionary containing extracted scores, or None values if parsing fails
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)

                score_1 = float(parsed.get("score_1"))
                score_2 = float(parsed.get("score_2"))
                ranking = float(parsed.get("ranking"))

                # Check if any required scores are None - if so, mark as parsing failure
                parsing_success = (
                    score_1 is not None and score_2 is not None and ranking is not None
                )

                return {
                    "score_1": score_1,
                    "score_2": score_2,
                    "ranking": ranking,
                    "response_1_analysis": parsed.get("response_1_analysis", ""),
                    "response_2_analysis": parsed.get("response_2_analysis", ""),
                    "parsing_success": parsing_success,
                }
            else:
                self.logger.warning(f"No JSON found in response: {response[:200]}...")
                return {
                    "score_1": None,
                    "score_2": None,
                    "ranking": None,
                    "response_1_analysis": "",
                    "response_2_analysis": "",
                    "parsing_success": False,
                }

        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {e}")
            return {
                "score_1": None,
                "score_2": None,
                "ranking": None,
                "response_1_analysis": "",
                "response_2_analysis": "",
                "parsing_success": False,
            }

    def evaluate_responses(
        self,
        assistant_responses: List[str],
        metadata_list: List[VanillaGenRMMetadata],
        config: VanillaGenRMConfig,
    ) -> List[float]:
        """Evaluate assistant responses and compute rewards.

        Args:
            assistant_responses: List of assistant responses to evaluate
            metadata_list: List of metadata containing ground truth scores
            config: Configuration for the environment

        Returns:
            List of rewards for each response
        """
        rewards = []

        for response, metadata in zip(assistant_responses, metadata_list):
            # Extract scores from the model's response
            extracted = self.extract_scores_from_response(response)

            # Get ground truth values
            gt_score_1 = metadata.get("score_1")
            gt_score_2 = metadata.get("score_2")
            gt_ranking = metadata.get("ranking")

            # Calculate reward based on configuration
            reward = self._calculate_reward(
                extracted, gt_score_1, gt_score_2, gt_ranking, config
            )
            rewards.append(reward)

        return rewards

    def _calculate_reward(
        self,
        extracted: Dict[str, Any],
        gt_score_1: Optional[float],
        gt_score_2: Optional[float],
        gt_ranking: Optional[int],
        config: VanillaGenRMConfig,
    ) -> float:
        """Calculate reward based on extracted scores vs ground truth using negative L1 distance.

        Always evaluates both individual scores and ranking (combined approach).

        Args:
            extracted: Extracted scores from model response
            gt_score_1: Ground truth score for response 1
            gt_score_2: Ground truth score for response 2
            gt_ranking: Ground truth ranking
            config: Environment configuration

        Returns:
            Calculated reward value (negative L1 distance)
        """
        if not extracted["parsing_success"]:
            return -100.0  # Large negative penalty if parsing failed

        total_l1_distance = 0.0
        num_components = 0

        # Individual score accuracy using L1 distance
        if gt_score_1 is not None and extracted["score_1"] is not None:
            distance_1 = abs(float(extracted["score_1"]) - float(gt_score_1))
            total_l1_distance += distance_1 * config["score_weight"]
            num_components += 1

        if gt_score_2 is not None and extracted["score_2"] is not None:
            distance_2 = abs(float(extracted["score_2"]) - float(gt_score_2))
            total_l1_distance += distance_2 * config["score_weight"]
            num_components += 1

        # Ranking accuracy using L1 distance
        if gt_ranking is not None and extracted["ranking"] is not None:
            distance_ranking = abs(float(extracted["ranking"]) - float(gt_ranking))
            total_l1_distance += distance_ranking * config["ranking_weight"]
            num_components += 1

        # Return negative L1 distance (higher rewards for smaller distances)
        if num_components > 0:
            reward = -total_l1_distance
        else:
            reward = -100.0  # Large negative penalty if no valid components

        return reward


@ray.remote
class VanillaGenRMEnvironment(EnvironmentInterface):
    """Environment for training vanilla GenRM models on response evaluation tasks."""

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: VanillaGenRMConfig):
        """Initialize the vanilla GenRM environment.

        Args:
            cfg: Configuration for the environment
        """
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        # Set default values for optional config parameters
        self.cfg.setdefault("score_weight", 1.0)
        self.cfg.setdefault("ranking_weight", 1.0)
        self.cfg.setdefault("reasoning_split_word", "</think>")

        # Create worker pool
        self.workers = [
            VanillaGenRMWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

        self.logger = logging.getLogger(__name__)

    def shutdown(self):
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[VanillaGenRMMetadata],
    ) -> EnvironmentReturn:
        """Run a step in the vanilla GenRM environment.

        Args:
            message_log_batch: Batch of message logs from the LLM interactions
            metadata: Batch of metadata containing ground truth scores

        Returns:
            EnvironmentReturn with observations, metadata, stop strings, rewards, and done flags
        """
        # Extract assistant responses from message logs
        assistant_responses = []
        for conversation in message_log_batch:
            # Find the last assistant response
            assistant_response = ""
            for message in reversed(conversation):
                if message["role"] == "assistant":
                    assistant_response = message["content"]
                    break
            assistant_responses.append(assistant_response)

        # Split reasoning from responses if reasoning_split_word is configured
        processed_responses = []
        reasoning_split_word = self.cfg.get("reasoning_split_word")
        for response in assistant_responses:
            if reasoning_split_word:
                # Check if reasoning split word is in the response
                if reasoning_split_word in response:
                    # Split on reasoning word and take the last part (after reasoning)
                    processed_response = response.split(reasoning_split_word)[
                        -1
                    ].lstrip()
                else:
                    # If split word not found, send empty string to worker
                    processed_response = ""
            else:
                processed_response = response
            processed_responses.append(processed_response)

        # Distribute work across workers
        batch_size = len(assistant_responses)
        chunk_size = max(1, batch_size // self.num_workers)

        futures = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            worker_idx = (i // chunk_size) % self.num_workers

            future = self.workers[worker_idx].evaluate_responses.remote(
                processed_responses[i:end_idx], metadata[i:end_idx], self.cfg
            )
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        rewards_list = [reward for chunk in results for reward in chunk]

        # Create observations (environment feedback)
        observations = []
        for reward in rewards_list:
            content = f"Environment: Score = {reward:.2f}"

            observations.append({"role": "environment", "content": content})

        # Convert to tensors
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).cpu()
        done_tensor = torch.ones_like(rewards_tensor, dtype=torch.bool).cpu()

        # All episodes terminate after one step for this task
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=done_tensor,
            answers=None,  # GenRM doesn't extract specific answers like Math environment
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Compute global metrics for the vanilla GenRM environment.

        Args:
            batch: Global batch of rollout data

        Returns:
            Tuple of (processed_batch, metrics_dict)
        """
        metrics = {}
        return batch, metrics
