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
import re
from typing import Any, NotRequired, TypedDict, Union

import ray
import torch
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.dapo_math_verifier import compute_score as dapo_math_verify
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.evals import answer_parsing


class MathEnvConfig(TypedDict):
    num_workers: int
    stop_strings: NotRequired[list[str] | None]  # Default stop strings for this env
    # The verifier type. None defaults to "math".
    verifier_type: NotRequired[str | None]
    math_verify_impl: NotRequired[str | None]


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


@ray.remote  # pragma: no cover
class HFVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
        **kwargs,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            Union[list[float], tuple[list[float], list[str | None]]].
            If return_extracted_answer is False, returns only the scores.
            If return_extracted_answer is True, returns (scores, extracted_answers).
        """
        results = []
        extracted_answers: list[str | None] = []

        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                with _mute_output():
                    math_verify_impl = kwargs.get("math_verify_impl", "hf_math_verify")
                    if kwargs.get("math_verify_impl") == "dapo_math_verify":
                        # This compute_score is from the DAPO Math Verifier from Verl
                        reward_dict = dapo_math_verify(response, ground_truth)
                        ret_score = reward_dict["score"]
                        extracted_answer = reward_dict["pred"]
                    elif kwargs.get("math_verify_impl") == "hf_math_verify":
                        ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                        ret_score, extracted_answer = self.verify_func(
                            [ground_truth_parsable], [response]
                        )
                    else:
                        raise ValueError(
                            f"Unknown math_verify_impl: {math_verify_impl}. Expected 'hf_math_verify' or 'dapo_math_verify'."
                        )

                results.append(float(ret_score))

                if return_extracted_answer:
                    # Make sure the extracted answer is not None and is a list of two elements
                    assert extracted_answer is not None
                    assert len(extracted_answer) == 2
                    extracted_gold, extracted_prediction = extracted_answer
                    # Get the extracted answer with the same logic as in the HFVerifyWorker
                    for pred in extracted_prediction:
                        if any(grader.verify(gold, pred) for gold in extracted_gold):
                            extracted_answers.append(pred)
                            break
                    else:
                        # If no match is found, means all answers are incorrect, just use the first prediction
                        extracted_answers.append(extracted_prediction[0][0])

            # It's possible to emit a TimeoutException and that wouldn't be caught since
            # it actually subclasses from BaseException and math-verify itself does not
            # to catch it.
            except (Exception, TimeoutException):
                results.append(0.0)
                extracted_answers.append(None)

        if return_extracted_answer:
            return results, extracted_answers
        else:
            return results


@ray.remote  # pragma: no cover
class MultilingualMultichoiceVerifyWorker:
    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
        **kwargs,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            Union[list[float], tuple[list[float], list[str | None]]].
            If return_extracted_answer is False, returns only the scores.
            If return_extracted_answer is True, returns (scores, extracted_answers).
        """
        results = []
        extracted_answers: list[str | None] = []

        for response, ground_truth in zip(pred_responses, ground_truths):
            response = answer_parsing.normalize_response(response)
            extracted_answer = None
            for answer_regex in answer_parsing.MULTILINGUAL_ANSWER_REGEXES:
                regex = answer_parsing.MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(
                    answer_regex
                )
                match = re.search(regex, response)
                if match:
                    extracted_answer = answer_parsing.normalize_extracted_answer(
                        match.group(1)
                    )
                    break
            score = 1.0 if extracted_answer == ground_truth else 0.0
            results.append(score)
            extracted_answers.append(extracted_answer)

        if return_extracted_answer:
            return results, extracted_answers
        else:
            return results


@ray.remote  # pragma: no cover
class EnglishMultichoiceVerifyWorker:
    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
        **kwargs,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            Union[list[float], tuple[list[float], list[str | None]]].
            If return_extracted_answer is False, returns only the scores.
            If return_extracted_answer is True, returns (scores, extracted_answers).
        """
        results = []
        extracted_answers: list[str | None] = []

        for response, ground_truth in zip(pred_responses, ground_truths):
            ground_truth = answer_parsing.normalize_response(ground_truth)
            response = answer_parsing.normalize_response(response)
            extracted_answer = None
            match = re.search(r"(?i)Answer\s*:[ \t]*([A-Z])", response)
            if match:
                extracted_answer = answer_parsing.normalize_extracted_answer(
                    match.group(1)
                )
            score = 1.0 if extracted_answer == ground_truth else 0.0
            results.append(score)
            if return_extracted_answer:
                extracted_answers.append(extracted_answer)

        if return_extracted_answer:
            return results, extracted_answers
        else:
            return results


@ray.remote  # pragma: no cover
class HFMultiRewardVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("math_multi_reward_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        return_extracted_answer: bool = False,
        **kwargs,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            Union[list[float], tuple[list[float], list[str | None]]].
            If return_extracted_answer is False, returns only the scores.
            If return_extracted_answer is True, returns (scores, extracted_answers).
        """
        def extract_xml_answer(text: str) -> str:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()

        def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
            extracted_responses = [extract_xml_answer(r) for r in completions]
            return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

        def int_reward_func(completions, **kwargs) -> list[float]:
            extracted_responses = [extract_xml_answer(r) for r in completions]
            return [1.0 if r.isdigit() else 0.0 for r in extracted_responses]

        def format_reward_func(completions, **kwargs) -> list[float]:
            """Reward function that checks if the completion has a specific format."""
            rewards = []
            for response in completions:
                
                pattern = r"^<think>.*?</think>\n<answer>.*?</answer>$"

                if re.search(pattern, response, re.DOTALL) and response.count("<answer>") == 1 and response.count("</answer>") == 1:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)

            return rewards

        number_of_rewards = 3
        results = [[] for i in range(number_of_rewards)]
        extracted_answers: list[str | None] = []

        for response, ground_truth in zip(pred_responses, ground_truths):

            try:
                # with _mute_output():
                math_verify_impl = kwargs.get("math_verify_impl", "hf_math_verify")
                if kwargs.get("math_verify_impl") == "hf_math_verify":
                    # # print(f"response {response}")
                    # print(f"len(response) {len(response)}")
                    # print(f"response[0] {response[0]}")
                    # # print(f"ground_truth {ground_truth}")
                    # print(f"len(ground_truth) {len(ground_truth)}")
                    # print(f"ground_truth[0] {ground_truth[0]}")
                    # print(f"ground_truth {ground_truth}")
                    cor_reward = correctness_reward_func([response],[ground_truth])
                    int_reward = int_reward_func([response])
                    format_reward = format_reward_func([response])
                    extracted_answer = extract_xml_answer(response)
                    # print(f"cor_reward {cor_reward}")
                    # print(f"int_reward {int_reward}")
                    # print(f"format_reward {format_reward}")
                    # ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                    # ret_score, extracted_answer = self.verify_func(
                    #     [ground_truth_parsable], [response]
                    # )
                else:
                    raise ValueError(
                        f"Unknown math_verify_impl: {math_verify_impl}. Expected 'hf_math_verify'"
                    )

                results[0].extend(float(cor_reward))
                results[1].extend(float(int_reward))
                results[2].extend(float(format_reward))

                if return_extracted_answer:
                    # Make sure the extracted answer is not None and is a list of two elements
                    assert extracted_answer is not None
                    assert len(extracted_answer) == 2
                    extracted_gold, extracted_prediction = extracted_answer
                    # Get the extracted answer with the same logic as in the HFVerifyWorker
                    for pred in extracted_prediction:
                        if any(grader.verify(gold, pred) for gold in extracted_gold):
                            extracted_answers.append(pred)
                            break
                    else:
                        # If no match is found, means all answers are incorrect, just use the first prediction
                        extracted_answers.append(extracted_prediction[0][0])

            # It's possible to emit a TimeoutException and that wouldn't be caught since
            # it actually subclasses from BaseException and math-verify itself does not
            # to catch it.
            except (Exception, TimeoutException):
                results[0].append(0.0)
                results[1].append(0.0)
                results[2].append(0.0)
                extracted_answers.append(None)

        if return_extracted_answer:
            # print(f"return_extracted_answer results {len(results)}")
            return results, extracted_answers
        else:
            # print(f"results {len(results)}")
            return results
            # return results --> [[0,1,0], [0,2,0], .........]



class MathEnvironmentMetadata(TypedDict):
    ground_truth: str
    extracted_answer: str | None


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class MathEnvironment(EnvironmentInterface[MathEnvironmentMetadata]):
    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        # TODO: split out this environment since it's doing more than just math
        verifier_type = cfg.get("verifier_type", "math")
        assert isinstance(verifier_type, str), (
            f"{verifier_type=} must be a string but was {type(verifier_type)}"
        )

        worker_cls = {
            "math": HFVerifyWorker,
            "english_multichoice": EnglishMultichoiceVerifyWorker,
            "multilingual_multichoice": MultilingualMultichoiceVerifyWorker,
        }[verifier_type]
        self.workers = [
            worker_cls.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MathEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[MathEnvironmentMetadata]:
        """Runs a step in the math environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: list[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness. The extracted answer will be stored to caculate cons@k.

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
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(
                chunk,
                ground_truth_chunk,
                return_extracted_answer,
                math_verify_impl=self.cfg.get("math_verify_impl", "hf_math_verify"),
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
            answers=extracted_answers,
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




@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class MathMultiRewardEnvironment(EnvironmentInterface[MathEnvironmentMetadata]):
    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        # TODO: split out this environment since it's doing more than just math
        verifier_type = cfg.get("verifier_type", "math")
        assert isinstance(verifier_type, str), (
            f"{verifier_type=} must be a string but was {type(verifier_type)}"
        )

        worker_cls = {
            "math": HFMultiRewardVerifyWorker,
        }[verifier_type]
        self.workers = [
            worker_cls.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]
        # print(f"self.workers {self.workers}")
        # print('Init MathMultiRewardEnvironment')


    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MathEnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[MathEnvironmentMetadata]:
        """Runs a step in the math environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: list[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness. The extracted answer will be stored to caculate cons@k.

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
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        ground_truths = [g["ground_truth"] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(
                chunk,
                ground_truth_chunk,
                return_extracted_answer,
                math_verify_impl=self.cfg.get("math_verify_impl", "hf_math_verify"),
            )
            for i, (chunk, ground_truth_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_ground_truths)
            )
        ]

        worker_results = ray.get(futures)
        print(f"worker_results {len(worker_results)}")
        print(f"worker_results[0] {len(worker_results[0])}")
        
        # Flatten the results and extract both scores and answers
        number_of_rewards = 3
        results = [[]for i in range(number_of_rewards)]
        extracted_answers: list[str | None] | None = (
            [] if return_extracted_answer else None
        )

        # for r1, r2 in zip(returns, returns_2):
        #     results.append(r1 + r2)

        for worker_result in worker_results:
            if return_extracted_answer:
                raise NotImplementedError("Skip return_extracted_answer handling")
                # worker_scores, worker_answers = worker_result
                # results.extend(worker_scores)
                # extracted_answers.extend(worker_answers)
            else:
                for i in range(number_of_rewards):
                    results[i].extend(worker_result[i])

        print(f"results len len {len(results)}")
        print(f"results[0] len len {len(results[0])}")
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if result
                else "Environment: incorrect",
            }
            for result in results[0] ## index 0 always store corretness reward
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).T.cpu() ## Shape Batch_size, Number_rewards
        print(f"rewards ??? {rewards.shape}")
        done = torch.ones_like(rewards[0]).cpu()
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
