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
from typing import Any, Optional, TypedDict

import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

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
from nemo_rl.evals import answer_parsing


class DAPOEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    verifier_type: Optional[str]


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    #print(f'{(pred == gt)=}, {pred=}, {gt=}, {solution_str=}')

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


@ray.remote  # pragma: no cover
class DAPOVerifyWorker:
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
                correct, pred = is_correct_minerva(response, ground_truth)

                # ret_score = 1.0 if correct else -1.0
                # TODO(jk): we can't separate reward and accuracy in this code, so will keep as 1 and 0
                ret_score = 1.0 if correct else 0.0
                results.append(float(ret_score))
            except Exception:
                # results.append(-1.0)
                # TODO(jk): we can't separate reward and accuracy in this code, so will keep as 1 and 0
                results.append(0.0)
        
        #print(f'{results=}')
        #print(f'num_correct={sum(results)}')
        return results


class DAPOEnvironmentMetadata(TypedDict):
    ground_truth: str


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class DAPOEnvironment(EnvironmentInterface):
    def __init__(self, cfg: DAPOEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        worker_cls = {
            "math": DAPOVerifyWorker,
        }[cfg.get("verifier_type", "math")]
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

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[DAPOEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: list[DAPOEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

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
            answers=None,
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
            # "accuracy_pos": batch["rewards"].clamp(min=0).float().mean().item(),
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
