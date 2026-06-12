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

"""Environment that scores Python solutions against LiveCodeBench unit tests.

Used by the SDPO LCBv6 recipe: each rollout produces Python code; this env
runs each test as a subprocess with a timeout, computes a binary all-pass
reward, and returns a textual feedback string (failed input / expected /
actual, or runtime error) suitable for SDPO's env-feedback teacher.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, NotRequired, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers


class LiveCodeBenchEnvConfig(TypedDict):
    num_workers: int
    timeout_per_test_seconds: NotRequired[float]
    max_feedback_chars: NotRequired[int]
    max_failed_tests_in_feedback: NotRequired[int]
    # Per-test input-dump caps (applied inside _truncate_input_for_feedback).
    # Prevent a single huge input (e.g. atcoder JSON-array) from consuming the
    # entire aggregated feedback budget and pushing Output/Expected out of view.
    max_input_chars_per_test: NotRequired[int]
    max_input_lines_per_test: NotRequired[int]


class LiveCodeBenchEnvMetadata(TypedDict):
    tests: list[dict[str, str]]
    starter_code: NotRequired[str]
    function_name: NotRequired[str]
    platform: NotRequired[str]
    question_id: NotRequired[str]
    environment_output: NotRequired[str]


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+\-]*[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(response: str) -> str:
    """Pull a Python code block out of a model response.

    Falls back to the whole response if no fenced block is found.
    """
    matches = _FENCE_RE.findall(response)
    if matches:
        return matches[-1].strip()
    return response.strip()


def _build_functional_harness(
    user_code: str, function_name: str, test_input: str, test_output: str
) -> str:
    """Wrap user code with a harness that calls a function and compares output.

    The harness reads JSON args from stdin, calls either Solution().<fn>(*args)
    or a top-level <fn>(*args), and prints "PASS" / "FAIL: ...".
    """
    harness = textwrap.dedent(
        f"""
        import json, sys, traceback
        try:
            # LCB LeetCode tests pass each function argument as a JSON value on
            # its own line, e.g. "[6,8]\\n5" -> args = [[6,8], 5].
            # Fall back to spreading a single top-level list (some tests use a
            # bare JSON list "[2, 3]" to mean args=(2, 3)).
            raw_in = sys.stdin.read()
            parsed = [json.loads(line) for line in raw_in.splitlines() if line.strip()]
            if len(parsed) == 1 and isinstance(parsed[0], list):
                args = parsed[0]
            else:
                args = parsed
            try:
                target = Solution().{function_name}
            except NameError:
                target = {function_name}
            actual = target(*args)
            expected = json.loads({json.dumps(test_output)})
            if actual == expected:
                print("__LCB_PASS__")
            else:
                print("__LCB_FAIL__")
                print("EXPECTED:", json.dumps(expected))
                print("ACTUAL:  ", json.dumps(actual))
        except Exception:
            print("__LCB_ERROR__")
            traceback.print_exc()
        """
    )
    return user_code + "\n\n" + harness


_FRAME_RE = re.compile(r'File\s+"[^"]*",\s+line\s+(\d+),\s+in\s+(\S+)')


def _truncate_input_for_feedback(
    text: str, max_lines: int = 40, max_chars: int = 800
) -> str:
    """Truncate input by lines AND chars, per paper F.3 plus a single-line guard.

    The line cap handles vertical inputs (paper Listing 5: `... (N more lines)`).
    The char cap handles huge single-line JSON-array inputs common in atcoder
    and LeetCode-style problems, where the line cap is a no-op and an
    unbounded single-line input would consume the entire aggregated feedback
    budget — pushing the Output / Expected sections out of view.
    """
    text = (text or "").rstrip()
    lines = text.splitlines()
    if len(lines) > max_lines:
        head = "\n".join(lines[:max_lines])
        remaining = len(lines) - max_lines
        text = f"{head}\n... ({remaining} more lines)"
    if len(text) > max_chars:
        remaining = len(text) - max_chars
        text = f"{text[:max_chars]}\n... ({remaining} more chars)"
    return text


def _parse_traceback(stderr: str) -> tuple[str, list[str]]:
    """Parse a Python traceback into (exception_line, paper-style frame lines).

    Returns:
        exception_line: e.g. "ZeroDivisionError: division by zero"
        frames: e.g. ["Line 91 in <module> (Solution.py)", "Line 25 in solve (Solution.py)"]
    """
    text = (stderr or "").strip()
    lines = text.splitlines()
    frames: list[str] = []
    for line in lines:
        m = _FRAME_RE.search(line)
        if m:
            frames.append(f"Line {m.group(1)} in {m.group(2)} (Solution.py)")
    exc_line = ""
    for line in reversed(lines):
        if line and not line.startswith(" ") and not line.lstrip().startswith(
            ("File ", "Traceback ", '"')
        ):
            exc_line = line.strip()
            break
    if not exc_line:
        exc_line = lines[-1].strip() if lines else "UnknownError:"
    if exc_line and ":" not in exc_line:
        exc_line = exc_line + ":"
    return exc_line, frames


def _format_wrong_answer(
    test_index: int,
    test_input: str,
    actual: str,
    expected: str,
    max_lines: int = 40,
    max_chars: int = 800,
) -> str:
    """Paper F.3 Listing 4 format."""
    return (
        f"Test Case {test_index}: Wrong Answer\n"
        f"Input\n{_truncate_input_for_feedback(test_input, max_lines, max_chars)}\n"
        f"Output\n{(actual or '').rstrip() or '<empty>'}\n"
        f"Expected\n{(expected or '').rstrip()}"
    )


def _format_runtime_error(
    stderr: str,
    test_input: str,
    max_lines: int = 40,
    max_chars: int = 800,
) -> str:
    """Paper F.3 Listing 5/6 format."""
    exc_line, frames = _parse_traceback(stderr)
    parts = ["Runtime Error", exc_line]
    parts.extend(frames)
    parts.append("Last Executed Input")
    parts.append(_truncate_input_for_feedback(test_input, max_lines, max_chars))
    return "\n".join(parts)


def _format_timeout(
    test_index: int,
    test_input: str,
    timeout: float,
    max_lines: int = 40,
    max_chars: int = 800,
) -> str:
    """Paper-style Time Limit Exceeded block."""
    return (
        f"Test Case {test_index}: Time Limit Exceeded\n"
        f"Input\n{_truncate_input_for_feedback(test_input, max_lines, max_chars)}\n"
        f"Timeout: {timeout:.1f}s"
    )


def _outputs_match(expected: str, actual: str) -> bool:
    """Compare program outputs with per-line whitespace normalization.

    rstrip() of each line handles trailing spaces that don't matter for
    most LCB grading; blank trailing lines are dropped.
    """
    def norm(s: str) -> list[str]:
        return [ln.rstrip() for ln in (s or "").splitlines()]

    e, a = norm(expected), norm(actual)
    while e and not e[-1]:
        e.pop()
    while a and not a[-1]:
        a.pop()
    return e == a


def _run_one_test(
    code: str,
    test: dict[str, str],
    function_name: str | None,
    timeout: float,
    test_index: int = 1,
    max_input_lines: int = 40,
    max_input_chars: int = 800,
) -> tuple[bool, str]:
    r"""Run a single test against the candidate code.

    Returns (passed, feedback_block). When the test passes, feedback_block is "".
    On failure, feedback_block is a paper F.3-style formatted block:
      - wrong answer  -> "Test Case N: Wrong Answer\nInput\n...\nOutput\n...\nExpected\n..."
      - runtime error -> "Runtime Error\n<ExcType>: <msg>\nLine N in <fn> (Solution.py)\nLast Executed Input\n..."
      - timeout       -> "Test Case N: Time Limit Exceeded\n..."
    """
    testtype = test.get("testtype") or test.get("type") or (
        "functional" if function_name else "stdin"
    )
    test_input = test.get("input", "")
    test_output = test.get("output", "")

    if testtype == "functional" and function_name:
        program = _build_functional_harness(
            code, function_name, test_input, test_output
        )
        stdin_data = test_input
    else:
        program = code
        stdin_data = test_input

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w") as f:
            f.write(program)

        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return False, _format_timeout(
                test_index, test_input, timeout, max_input_lines, max_input_chars
            )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    if testtype == "functional" and function_name:
        if "__LCB_PASS__" in stdout:
            return True, ""
        if "__LCB_ERROR__" in stdout:
            tb = stdout.split("__LCB_ERROR__", 1)[-1].strip()
            return False, _format_runtime_error(
                tb or stderr, test_input, max_input_lines, max_input_chars
            )
        if "__LCB_FAIL__" in stdout:
            details = stdout.split("__LCB_FAIL__", 1)[-1].strip()
            # Functional harness already printed "EXPECTED:" / "ACTUAL:" lines; extract them.
            actual_match = re.search(r"ACTUAL:\s*(.+)", details)
            expected_match = re.search(r"EXPECTED:\s*(.+)", details)
            actual_v = actual_match.group(1).strip() if actual_match else "<unknown>"
            expected_v = expected_match.group(1).strip() if expected_match else "<unknown>"
            return False, _format_wrong_answer(
                test_index, test_input, actual_v, expected_v,
                max_input_lines, max_input_chars,
            )
        return False, _format_runtime_error(
            stderr or stdout, test_input, max_input_lines, max_input_chars
        )

    if proc.returncode != 0:
        return False, _format_runtime_error(
            stderr, test_input, max_input_lines, max_input_chars
        )

    if _outputs_match(test_output, stdout):
        return True, ""
    return False, _format_wrong_answer(
        test_index, test_input, stdout, test_output,
        max_input_lines, max_input_chars,
    )


@ray.remote  # pragma: no cover
class LiveCodeBenchWorker:
    """Worker that scores rollouts against LCB unit tests in subprocesses."""

    def grade(
        self,
        responses: list[str],
        metadata: list[LiveCodeBenchEnvMetadata],
        timeout: float,
        max_feedback_chars: int,
        max_failed_in_feedback: int,
        max_input_lines: int = 40,
        max_input_chars: int = 800,
    ) -> list[tuple[float, str]]:
        results: list[tuple[float, str]] = []
        for response, meta in zip(responses, metadata):
            code = extract_python_code(response)
            tests = meta.get("tests", [])
            if not tests:
                results.append((0.0, "no tests available"))
                continue

            function_name = meta.get("function_name") or None
            failure_blocks: list[str] = []
            num_failed = 0
            for i, test in enumerate(tests):
                passed, fb = _run_one_test(
                    code, test, function_name, timeout, test_index=i + 1,
                    max_input_lines=max_input_lines,
                    max_input_chars=max_input_chars,
                )
                if not passed:
                    num_failed += 1
                    if len(failure_blocks) < max_failed_in_feedback:
                        failure_blocks.append(fb)

            if num_failed == 0:
                # All tests passed: emit a short human-readable note. Combined-mode
                # SDPO teacher doesn't read env_feedback for successful rollouts.
                feedback = f"All {len(tests)} tests passed."
                reward = 1.0
            else:
                # Paper F.3 format: concatenate failure blocks with blank-line
                # separator, no per-batch summary header.
                feedback = "\n\n".join(failure_blocks)
                reward = 0.0

            if len(feedback) > max_feedback_chars:
                feedback = feedback[:max_feedback_chars] + "\n...[truncated]"

            results.append((reward, feedback))
        return results


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class LiveCodeBenchEnvironment(EnvironmentInterface[LiveCodeBenchEnvMetadata]):
    """Run candidate code against LiveCodeBench unit tests."""

    def __init__(self, cfg: LiveCodeBenchEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.timeout = cfg.get("timeout_per_test_seconds", 6.0)
        self.max_feedback_chars = cfg.get("max_feedback_chars", 2000)
        self.max_failed_in_feedback = cfg.get("max_failed_tests_in_feedback", 3)
        self.max_input_lines = cfg.get("max_input_lines_per_test", 40)
        self.max_input_chars = cfg.get("max_input_chars_per_test", 800)
        self.workers = [
            LiveCodeBenchWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[LiveCodeBenchEnvMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[LiveCodeBenchEnvMetadata]:
        responses: list[str] = []
        for conversation in message_log_batch:
            assistant_chunks = [
                str(turn["content"])
                for turn in conversation
                if turn["role"] == "assistant"
            ]
            responses.append("".join(assistant_chunks))

        chunked_responses = chunk_list_to_workers(responses, self.num_workers)
        chunked_metadata = chunk_list_to_workers(list(metadata), self.num_workers)

        futures = [
            self.workers[i].grade.remote(
                resp_chunk,
                meta_chunk,
                self.timeout,
                self.max_feedback_chars,
                self.max_failed_in_feedback,
                self.max_input_lines,
                self.max_input_chars,
            )
            for i, (resp_chunk, meta_chunk) in enumerate(
                zip(chunked_responses, chunked_metadata)
            )
        ]
        worker_results: list[list[tuple[float, str]]] = ray.get(futures)

        rewards_list: list[float] = []
        feedbacks: list[str] = []
        for chunk in worker_results:
            for reward, feedback in chunk:
                rewards_list.append(reward)
                feedbacks.append(feedback)

        rewards = torch.tensor(rewards_list, dtype=torch.float32)
        terminateds = torch.ones_like(rewards, dtype=torch.bool)

        observations = [
            {"role": "environment", "content": fb} for fb in feedbacks
        ]

        new_metadata: list[LiveCodeBenchEnvMetadata] = []
        for meta, fb in zip(metadata, feedbacks):
            updated = dict(meta)
            updated["environment_output"] = fb
            new_metadata.append(updated)  # type: ignore[arg-type]

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        rewards = (
            batch["rewards"] if batch["rewards"].ndim == 1 else batch["rewards"][:, 0]
        )
        rewards = rewards * batch["is_end"]
        metrics = {
            "accuracy": rewards.mean().item(),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics
