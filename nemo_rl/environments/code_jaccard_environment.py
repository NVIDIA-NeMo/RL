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

from typing import Any, Optional, TypedDict, Union

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class CodeJaccardEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env


class CodeJaccardEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote  # pragma: no cover
class CodeJaccardVerifyWorker:
    """Worker for evaluating code responses using Jaccard-based similarity."""

    def __init__(self) -> None:
        pass

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify code responses against ground-truth solutions using Jaccard-based similarity.

        We use a simple text similarity approach (Jaccard over tokenized words)
        to evaluate how well the model's response aligns with the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground-truth solutions.
            return_extracted_answer: bool. Whether to return extracted answers (here, the full response).

        Returns:
            Union[list[float], tuple[list[float], list[str | None]]].
            If return_extracted_answer is False, returns only the scores.
            If return_extracted_answer is True, returns (scores, extracted_answers).
        """
        results = []
        extracted_answers: list[str | None] = []

        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                # Simple reward based on text similarity/alignment
                # This is a basic implementation - could be enhanced with more sophisticated metrics
                score = self._calculate_preference_score(response, ground_truth)
                results.append(float(score))

                if return_extracted_answer:
                    # For CodeJaccard, the "extracted answer" is the full response
                    extracted_answers.append(response.strip())

            except Exception:
                results.append(0.0)
                if return_extracted_answer:
                    extracted_answers.append(None)

        if return_extracted_answer:
            return results, extracted_answers
        else:
            return results

    def _calculate_preference_score(self, response: str, ground_truth: str) -> float:
        """Calculate a Jaccard-based alignment score between response and ground truth.

        This is a simplified scoring function. In practice, you might want to use:
        - Semantic similarity models
        - BLEU/ROUGE scores
        - Tokenize both texts into sets A and B (here we use whitespace tokenization).
        - Compute intersection size |A ∩ B| and union size |A ∪ B|.
        - J(A, B) = |A ∩ B| / |A ∪ B|, with guards for union=0 -> 0.0.
        - Optionally combine with a length-ratio penalty to discourage degenerate very short/long matches.

        Complexity:
        - Tokenization: O(n + m)
        - Set ops: O(n + m) average (hash sets)

        Args:
            response: The model's response

        Returns:
            float: Score between 0.0 and 1.0
        """
        # Normalize both texts
        response_clean = response.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()

        # Simple exact match (could be enhanced)
        if response_clean == ground_truth_clean:
            return 1.0

        # Basic similarity based on common words
        response_words = set(response_clean.split())
        ground_truth_words = set(ground_truth_clean.split())

        if not ground_truth_words:
            return 0.0

        # Jaccard similarity
        intersection = len(response_words & ground_truth_words)
        union = len(response_words | ground_truth_words)

        if union == 0:
            return 0.0

        jaccard_score = intersection / union

        # Length penalty for responses that are too short or too long
        len_ratio = min(len(response_clean), len(ground_truth_clean)) / max(
            len(response_clean), len(ground_truth_clean), 1
        )

        # Combine scores
        final_score = jaccard_score * len_ratio

        return min(1.0, max(0.0, final_score))


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class CodeJaccardEnvironment(EnvironmentInterface[CodeJaccardEnvironmentMetadata]):
    """Environment for evaluating code responses using Jaccard similarity."""

    def __init__(self, cfg: CodeJaccardEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        # Create worker pool
        self.workers = [
            CodeJaccardVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CodeJaccardEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[CodeJaccardEnvironmentMetadata]:
        """Runs a step in the CodeJaccard environment.

        Args:
            message_log_batch: Batch of OpenAI-API-like message logs.
            metadata: Batch of CodeJaccardEnvironmentMetadata with ground truth.
            return_extracted_answer: Whether to return extracted answers.

        Returns:
            EnvironmentReturn: Tuple containing observations, metadata, stop strings, rewards, and done flags.
        """
        # Extract the assistant's responses from the message history
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        # Chunk work across workers
        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(
                chunk, ground_truth_chunk, return_extracted_answer
            )
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        worker_results = ray.get(futures)

        # Flatten the results and extract both scores and answers
        results = []
        extracted_answers: list[str | None] | None = (
            [] if return_extracted_answer else None
        )

        for worker_result in worker_results:
            if return_extracted_answer:
                worker_scores, worker_answers = worker_result
                results.extend(worker_scores)
                extracted_answers.extend(worker_answers)
            else:
                results.extend(worker_result)

        # Create observations based on Jaccard alignment
        observations = [
            {
                "role": "environment",
                "content": f"Environment: jaccard aligned (score: {result:.2f})"
                if result > 0.5
                else f"Environment: jaccard misaligned (score: {result:.2f})",
            }
            for result in results
        ]

        # Create reward and done tensors
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=extracted_answers,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Post-process batch and compute metrics for CodeJaccard."""
        # Calculate preference alignment metrics
        rewards = batch["rewards"]

        metrics = {
            "preference_alignment_rate": float(torch.mean((rewards > 0.5).float())),
            "average_preference_score": float(torch.mean(rewards)),
            "high_alignment_rate": float(torch.mean((rewards > 0.8).float())),
        }

        return batch, metrics
