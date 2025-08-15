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
from typing import Dict, List, Optional, Tuple, TypedDict

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


class IFEvalEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


class IFEvalEnvironmentMetadata(TypedDict):
    instruction_id_list: list
    instruction_kwargs: list


@ray.remote
class IFEvalVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.IFEVAL

    def __init__(self):
        from nemo_rl.environments.instruction_following.instructions_registry import (
            INSTRUCTION_DICT,
        )

        self.INSTRUCTION_DICT = INSTRUCTION_DICT
        logging.getLogger("ifeval_verify").setLevel(logging.CRITICAL)

    def instruction_following_rewards(self, prompt, response, args):
        """Tests response to see if instrutions are followed."""
        try:
            task_args = args
            instruction_list = task_args["instruction_id_list"]
            is_following_list = []

            for index, instruction_id in enumerate(instruction_list):
                try:
                    instruction_cls = self.INSTRUCTION_DICT[instruction_id]
                    instruction = instruction_cls(instruction_id)

                    kwargs = (
                        task_args["instruction_kwargs"][index]
                        if task_args["instruction_kwargs"][index] is not None
                        else {}
                    )
                    instruction.build_description(**kwargs)
                    instruction_args = instruction.get_instruction_args()
                    if instruction_args and "prompt" in instruction_args:
                        instruction.build_description(prompt=prompt)

                    if response.strip() and instruction.check_following(response):
                        is_following_list.append(True)
                    else:
                        is_following_list.append(False)
                except Exception as e:
                    print(f"Error in instruction_following_rewards: {e}, task: {args}")

            low, high = 0, 1
            correctness = sum(is_following_list) / len(is_following_list)
            score = low + (high - low) * correctness
            return score, True
        except Exception as e:
            print(f"Error in instruction_following_rewards: {e}")
            return 0, False

    def verify(
        self,
        pred_responses: List[str],
        prompts: List[str],
        metadata: List[IFEvalEnvironmentMetadata],
    ) -> List[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            prompts: List[str]. The prompts that were used to generate the responses.
            metadata: List[IFEvalEnvironmentMetadata]. The metadata for the prompts.

        Returns:
            List[float]. The rewards for each predicted response.
        """
        results = []
        for response, prompt, metadata in zip(pred_responses, prompts, metadata):
            try:
                ret_score, _ = self.instruction_following_rewards(
                    prompt, response, metadata
                )

                results.append(float(ret_score))
            except Exception:
                results.append(0.0)
        return results


@ray.remote
class IFEvalEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.IFEVAL

    def __init__(self, cfg: IFEvalEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            IFEvalVerifyWorker.options(
                runtime_env={"py_executable": IFEvalVerifyWorker.DEFAULT_PY_EXECUTABLE}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[IFEvalEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[IFEvalEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

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
        prompts = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append(assistant_responses[-1])
            prompts.append(conversation[0]["content"])

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_prompts = chunk_list_to_workers(prompts, self.num_workers)
        chunked_metadata = chunk_list_to_workers(metadata, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(response, prompt, metadata)
            for i, (response, prompt, metadata) in enumerate(
                zip(chunked_assistant_response_batch, chunked_prompts, chunked_metadata)
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
