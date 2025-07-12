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
import contextlib
import io
import logging
from typing import Any, Optional, TypedDict
import re

import ray
import torch

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


class VLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env

@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield

def format_and_exact_answer(ground_truth: str, response: str) -> str:
    ''' 
    Reward the agent for the following:
    1. response follows the format: (.*) <think> (.*) </think> <answer> (.*) </answer>
    2. the answer within the <answer> tags is the same as the ground truth (case-insensitive)
    '''
    rew = 0.0
    # Check for <think> tags
    if re.search(r"<think>[\s\S]*</think>", response):
        rew += 0.25  # 0.25 points for having think tags
    
    # Check for <answer> tags
    if re.search(r"<answer>[\s\S]*</answer>", response):
        rew += 0.75  # 0.75 points for having answer tags
    
    is_correct_answer = False
    match = re.search(r"<answer>([\s\S]*)</answer>", response)
    if match:
        answer = match.group(1)
        # Remove all non-alphanumeric characters (including whitespace, punctuation, etc.)
        answer_clean = ''.join(c for c in answer if c.isalnum()).lower()
        ground_truth_clean = ''.join(c for c in ground_truth if c.isalnum()).lower()
        if answer_clean == ground_truth_clean:
            rew += 4.0  # four points for the answer
            is_correct_answer = True
    return rew, is_correct_answer


@ray.remote
class VLMVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("vlm_worker").setLevel(logging.CRITICAL)

        # this is a simple reward function that rewards the agent for correct answer and correct format
        self.verify_func = format_and_exact_answer

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[float]. The rewards for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                with _mute_output():
                    try:
                        ret_score, _ = self.verify_func(
                            ground_truth, response
                        )
                    except Exception as e:
                        ret_score = 0.0
                        print(f"Error in verify_func: {e}")
                results.append(float(ret_score))
            except Exception as e:
                print(f"Error in verify: {e}")
                results.append(0.0)
        return results


class VLMEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)
class VLMEnvironment(EnvironmentInterface):
    def __init__(self, cfg: VLMEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            VLMVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
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
        metadata: list[VLMEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the vlm environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the VLM.
            metadata: list[VLMEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - list[dict[str, str]]: Observations/responses batch
                - list[dict]: Updated metadata
                - list[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk)
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

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
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
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
            correct_solution_generation_lengths = 0

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
