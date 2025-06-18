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

from typing import Any, Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers

try:
    from reasoning_gym import get_score_answer_fn  # type: ignore
except ImportError:  # pragma: no cover
    get_score_answer_fn = None  # type: ignore


class ReasoningGymEnvConfig(TypedDict):
    """Configuration for the Reasoning Gym environment."""

    num_workers: int


@ray.remote
class _ReasoningGymWorker:
    """Stateless worker that scores model responses with Reasoning Gym."""

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self):
        if get_score_answer_fn is None:
            raise ImportError(
                "Failed to import `reasoning_gym`. Please make sure the package is installed."
            )

        self._get_score_answer_fn = get_score_answer_fn

    def verify(
        self,
        pred_responses: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> List[float]:
        """Return a float reward for every prediction."""
        results: List[float] = []

        for response, meta in zip(pred_responses, metadatas):
            dataset_name: Optional[str] = meta.get("source_dataset")
            if dataset_name is None:
                results.append(0.0)
                continue

            score_fn = self._get_score_answer_fn(dataset_name)
            assert "ground_truth" in meta, f"ground_truth not found in meta: {meta}"
            entry = {}
            entry["answer"] = meta["ground_truth"]
            entry["metadata"] = meta
            score = score_fn(response, entry)
            results.append(float(score))

        return results


@ray.remote
class ReasoningGymEnvironment(EnvironmentInterface):
    """Environment used for tasks from the Reasoning Gym benchmark."""

    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: ReasoningGymEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        self.workers = [
            _ReasoningGymWorker.options(
                runtime_env={"py_executable": _ReasoningGymWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        # Terminate all Ray actors cleanly
        for worker in getattr(self, "workers", []):
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[Dict[str, Any]],
    ) -> EnvironmentReturn:
        """Compute rewards for a batch of conversations.

        Args:
            message_log_batch: A batch of chat transcripts. Each element is the
                full message log for a single prompt, containing (user,
                assistant, environment) messages.
            metadata: List of metadata objects one per prompt used by reasoning gym grader.
        """
        assistant_response_batch: List[str] = []
        for conversation in message_log_batch:
            # There may be multiple assistant messages (multi-turn). We join
            # them so the scorer sees the entire answer.
            assistant_responses = [
                m["content"] for m in conversation if m["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_metadata = chunk_list_to_workers(metadata, self.num_workers)

        futures = [
            self.workers[i].verify.remote(resp_chunk, meta_chunk)
            for i, (resp_chunk, meta_chunk) in enumerate(
                zip(chunked_responses, chunked_metadata)
            )
        ]

        results_nested: List[List[float]] = ray.get(futures)
        results: List[float] = [item for sublist in results_nested for item in sublist]

        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if score >= 1.0
                else "Environment: incorrect",
            }
            for score in results
        ]

        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()  # One-shot tasks – always done

        next_stop_strings: List[Optional[str]] = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Compute environment-specific metrics over an entire rollout batch."""
        # Ensure rewards are zero for sequences that did not properly terminate
        batch["rewards"] = batch["rewards"] * batch["is_end"]

        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0.0

        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
