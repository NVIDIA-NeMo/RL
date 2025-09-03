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
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.code.livecodebench import compute_score, prepare_tests
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers, extract_code


class CodeEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env
    timeout: int = 10  # Timeout for the code execution
    reasoning_split_word: Optional[
        str
    ]  # Configurable split word for response processing (default: "</think>")


class CodeEnvironmentMetadata(TypedDict):
    unittests: Optional[List[Dict[str, str]]]
    fn_name: Optional[str]


@ray.remote
class CodeVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(
        self,
        reasoning_split_word: Optional[str] = None,  # Configurable split word for response processing
    ):
        logging.getLogger("code_verify").setLevel(logging.INFO)
        self.verify_func = compute_score
        self.reasoning_split_word = reasoning_split_word 

    def verify(
        self,
        pred_responses: List[str],
        metadata: List[CodeEnvironmentMetadata],
        timeout: int,
    ) -> List[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            metadata: List[CodeEnvironmentMetadata]. The metadata containing unit tests.
            timeout: int. Timeout for code execution.

        Returns:
            List[float]. The rewards for each predicted response.
        """
        results = []
        for response, metadata_item in zip(pred_responses, metadata):
            try:
                # Only split if reasoning_split_word is provided
                if self.reasoning_split_word:
                    final_response = response.split(self.reasoning_split_word)[
                        -1
                    ].lstrip()
                code_str = extract_code(final_response)

                ret_score, execution_metadata = self.verify_func(
                    code_str, metadata_item, timeout
                )

            except Exception as e:
                ret_score = 0.0

            results.append(float(ret_score))
        return results


@ray.remote
class CodeEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: CodeEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            CodeVerifyWorker.options(
                runtime_env={
                    "py_executable": CodeVerifyWorker.DEFAULT_PY_EXECUTABLE,
                    "env_vars": {"OMP_NUM_THREADS": "1"},
                }
            ).remote(reasoning_split_word=cfg.get("reasoning_split_word"))
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[CodeEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the code environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[CodeEnvironmentMetadata]. The grader will use the 'unittests' and 'fn_name' keys to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
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

        unittests = [prepare_tests(m) for m in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_unittests = chunk_list_to_workers(unittests, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, unittests_chunk, self.cfg["timeout"])
            for i, (chunk, unittests_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_unittests)
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
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
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
            # "table": table, TODO @sahilj WIP
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
