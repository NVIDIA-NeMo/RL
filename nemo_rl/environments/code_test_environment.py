# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Environment for evaluating code generation with test cases.

Executes model-generated Python code against input/output test cases and
returns binary pass/fail rewards. Designed for benchmarks like LiveCodeBench
where solutions read from stdin and write to stdout.
"""

import json
import re
import subprocess
import sys
import tempfile
from typing import Any, NotRequired, TypedDict, Union

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


class CodeTestEnvConfig(TypedDict):
    """Configuration for CodeTestCaseEnvironment."""

    num_workers: int
    timeout_per_test: int
    stop_strings: NotRequired[list[str] | None]


class CodeTestMetadata(TypedDict):
    """Metadata for each code evaluation sample."""

    test_cases: list[dict[str, str]]
    ground_truth: str


def extract_code(response: str) -> str:
    """Extract Python code from a model response.

    Supports markdown code blocks (```python ... ``` or ``` ... ```)
    and falls back to using the full response as code.

    Args:
        response: The model's text response potentially containing code blocks.

    Returns:
        Extracted Python code string.
    """
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response.strip()


def run_single_test(code: str, test_input: str, expected_output: str, timeout: int) -> bool:
    """Run code in an isolated subprocess and check stdout against expected output.

    The subprocess is hardened with Python isolation flags (-I, -S),
    a temporary working directory, and a minimal environment.

    Args:
        code: Python source code to execute.
        test_input: String to feed via stdin.
        expected_output: Expected stdout content.
        timeout: Maximum execution time in seconds.

    Returns:
        True if code exits successfully and stdout matches expected output.
    """
    try:
        with tempfile.TemporaryDirectory() as sandbox_dir:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", code],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=sandbox_dir,
                env={"PYTHONNOUSERSITE": "1"},
            )
        return (
            result.returncode == 0
            and result.stdout.strip() == expected_output.strip()
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False


@ray.remote  # pragma: no cover
class CodeTestCaseVerifyWorker:
    """Worker that executes generated code against test cases.

    Distributes across Ray workers for parallel evaluation of multiple
    code samples. Each worker handles a chunk of the batch.
    """

    def verify(
        self,
        pred_responses: list[str],
        test_cases_batch: list[list[dict[str, str]]],
        return_extracted_answer: bool = False,
        timeout: int = 5,
    ) -> Union[list[float], tuple[list[float], list[str | None]]]:
        """Verify code responses against test cases.

        Args:
            pred_responses: Model-generated responses containing code.
            test_cases_batch: Test cases for each response.
            return_extracted_answer: If True, also return extracted code.
            timeout: Timeout per test case in seconds.

        Returns:
            Scores (and optionally extracted answers) for each response.
        """
        results: list[float] = []
        extracted_answers: list[str | None] = []

        for response, test_cases in zip(pred_responses, test_cases_batch, strict=True):
            code = extract_code(response)

            if not code or not test_cases:
                results.append(0.0)
                extracted_answers.append(None)
                continue

            all_passed = True
            for tc in test_cases:
                test_input = tc.get("input", "")
                expected_output = tc.get("expected_output", tc.get("output", ""))
                if not run_single_test(code, test_input, expected_output, timeout):
                    all_passed = False
                    break

            results.append(1.0 if all_passed else 0.0)
            if return_extracted_answer:
                extracted_answers.append(code)

        if return_extracted_answer:
            return results, extracted_answers
        return results


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class CodeTestCaseEnvironment(EnvironmentInterface[CodeTestMetadata]):
    """Environment for evaluating code generation using test cases.

    Extracts code from model responses, runs it against input/output test cases
    in isolated subprocesses, and returns binary pass/fail rewards (1.0 if all
    tests pass, 0.0 otherwise). Follows the MathEnvironment pattern for metrics.
    """

    def __init__(self, cfg: CodeTestEnvConfig) -> None:
        """Initialize the environment with worker pool.

        Args:
            cfg: Environment configuration with num_workers and timeout_per_test.
        """
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.timeout = cfg["timeout_per_test"]
        self.workers = [
            CodeTestCaseVerifyWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        """Shutdown all Ray workers."""
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[CodeTestMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[CodeTestMetadata]:
        """Execute generated code against test cases and return rewards.

        Args:
            message_log_batch: Batch of conversation histories with assistant responses.
            metadata: Test cases and ground truth for each sample.
            return_extracted_answer: If True, return extracted code in answers.

        Returns:
            EnvironmentReturn with rewards (1.0=pass, 0.0=fail) and observations.
        """
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(msg["content"])
                for msg in conversation
                if msg["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        test_cases_batch: list[list[dict[str, str]]] = []
        for m in metadata:
            raw = m.get("test_cases", [])
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    raw = []
            if not isinstance(raw, list):
                raw = []
            test_cases_batch.append(raw)

        chunked_responses = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_test_cases = chunk_list_to_workers(
            test_cases_batch, self.num_workers
        )

        futures = [
            self.workers[i].verify.remote(
                resp_chunk,
                tc_chunk,
                return_extracted_answer,
                timeout=self.timeout,
            )
            for i, (resp_chunk, tc_chunk) in enumerate(
                zip(chunked_responses, chunked_test_cases, strict=True)
            )
        ]

        worker_results = ray.get(futures)

        results: list[float] = []
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
                "content": "Environment: all tests passed"
                if score == 1.0
                else "Environment: tests failed",
            }
            for score in results
        ]

        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()
        next_stop_strings: list[None] = [None] * len(message_log_batch)

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
        """Compute metrics for the batch after all rollouts complete.

        Args:
            batch: Global rollout batch with rewards, generation lengths, etc.

        Returns:
            Tuple of (updated batch, metrics dict with accuracy, pass@k, etc.)
        """
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
