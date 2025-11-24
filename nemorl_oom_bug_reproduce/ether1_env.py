import contextlib
import io
import logging
import re
from typing import Any, NotRequired, TypedDict, Union

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
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

from ether0.rewards import EVAL_FUNCTIONS, accuracy_reward, format_reward, extract_thought_answer_strict
from ether0.models import RewardFunctionInfo

class Ether1EnvConfig(TypedDict):
    num_workers: int
    stop_strings: NotRequired[list[str] | None]  # Default stop strings for this env


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


LOOSE_XML_ANSWER_LOOSE_PATTERN = r"<answer>\s*(\S[\s\S]*?)\s*<\/answer>"
def extract_answer(
    text: str,
) -> tuple[str | None, str | None]:
    # find all text inside <answer>...</answer> tags, there may be more than one match
    matches = list(re.finditer(LOOSE_XML_ANSWER_LOOSE_PATTERN, text))
    if not matches:
        return None
    # return the last match
    # the thought is everything before that last match
    answer = matches[-1].group(1)
    return answer


@ray.remote  # pragma: no cover
class HFVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("ether1_verify").setLevel(logging.CRITICAL)

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            Union[list[float], tuple[list[float], list[str | None]]].
        """

        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                with _mute_output():
                    extracted_answer = extract_answer(response) or response
                    # _, extracted_answer = extract_thought_answer_strict(response, reasoning=True)
                    if extracted_answer is None:
                        format_rew = 0.0
                        ret_score = 0.0
                    else:
                        format_rew = 1.0
                        reward_info = RewardFunctionInfo.model_validate(ground_truth)
                        fxn_name, ideal, problem_type = tuple(reward_info.model_dump().values())
                        reward_function = "rxn_eval" if problem_type == "reaction-name" else fxn_name
                        ret_score = EVAL_FUNCTIONS[reward_function](extracted_answer, ideal, soft=False, metadata=None)
                    results.append(format_rew * ret_score)
            except Exception:
                results.append(0.0)

        return results


class Ether1EnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class Ether1Environment(EnvironmentInterface[Ether1EnvironmentMetadata]):
    def __init__(self, cfg: Ether1EnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        worker_cls = HFVerifyWorker
        self.workers = [
            worker_cls.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[Ether1EnvironmentMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[Ether1EnvironmentMetadata]:
        """Runs a step in the ether1 environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: list[Ether1EnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness. The extracted answer will be stored to caculate cons@k.

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