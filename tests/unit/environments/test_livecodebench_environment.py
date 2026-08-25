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

"""Unit tests for the LiveCodeBench environment helpers.

These tests cover the pure code-extraction and per-test grading paths in
isolation (no Ray, no models). The full @ray.remote actor flow is exercised
in the nightly suite where Ray initialization is acceptable.
"""

from __future__ import annotations

import pytest

from nemo_rl.environments.livecodebench_environment import (
    _run_one_test,
    _truncate_input_for_feedback,
    extract_python_code,
)


def test_extract_python_code_picks_fenced_block():
    response = (
        "Sure, here's my solution:\n\n"
        "```python\nimport sys\nprint(sys.stdin.read())\n```\n"
        "Hope that helps!"
    )
    assert extract_python_code(response) == "import sys\nprint(sys.stdin.read())"


def test_extract_python_code_returns_none_without_fence():
    """Reference behavior: no fenced block -> None (graded INCORRECT_FORMAT),
    never execute prose as Python."""
    response = "no fence here\nimport math\nprint(math.pi)\n"
    assert extract_python_code(response) is None


def test_extract_python_code_picks_longest_fence():
    """Reference behavior: max(blocks, key=len) — a trailing usage-example
    snippet must not displace the main solution block."""
    response = (
        "Solution:\n```python\ndef solve(x):\n    return x * 2\nprint(solve(int(input())))\n```\n"
        "Example usage:\n```python\nsolve(3)\n```"
    )
    assert "def solve" in extract_python_code(response)


def test_run_one_test_stdin_passes():
    code = "n = int(input())\nprint(n * 2)"
    test = {"input": "21\n", "output": "42\n", "testtype": "stdin"}
    passed, _kind, feedback = _run_one_test(code, test, function_name=None, timeout=4.0)
    assert passed is True
    assert feedback == ""


def test_run_one_test_stdin_wrong_answer_paper_format():
    """Wrong-answer feedback follows paper F.3 Listing 4 (`Test Case N: Wrong Answer / Input / Output / Expected`)."""
    code = "n = int(input())\nprint(n + 1)"  # off by one
    test = {"input": "21\n", "output": "42\n", "testtype": "stdin"}
    passed, _kind, feedback = _run_one_test(code, test, function_name=None, timeout=4.0, test_index=3)
    assert passed is False
    assert "Test Case 3: Wrong Answer" in feedback
    assert "\nInput\n" in feedback
    assert "\nOutput\n" in feedback
    assert "\nExpected\n" in feedback
    assert "22" in feedback  # actual model output
    assert "42" in feedback  # expected


def test_run_one_test_stdin_runtime_error_paper_format():
    """Runtime-error feedback follows paper F.3 Listing 5/6."""
    code = "raise ZeroDivisionError('boom')"
    test = {"input": "abc", "output": "", "testtype": "stdin"}
    passed, _kind, feedback = _run_one_test(code, test, function_name=None, timeout=4.0)
    assert passed is False
    assert feedback.startswith("Runtime Error\n")
    assert "ZeroDivisionError" in feedback
    assert "Solution.py)" in feedback
    assert "Last Executed Input" in feedback
    assert "abc" in feedback


def test_run_one_test_functional_passes():
    code = (
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
    )
    test = {"input": "[2, 3]", "output": "5", "testtype": "functional"}
    passed, _kind, feedback = _run_one_test(code, test, function_name="add", timeout=4.0)
    assert passed is True
    assert feedback == ""


def test_run_one_test_functional_wrong_answer_paper_format():
    code = (
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return a - b\n"  # bug
    )
    test = {"input": "[2, 3]", "output": "5", "testtype": "functional"}
    passed, _kind, feedback = _run_one_test(code, test, function_name="add", timeout=4.0, test_index=2)
    assert passed is False
    assert "Test Case 2: Wrong Answer" in feedback
    assert "Expected" in feedback


def test_run_one_test_timeout_paper_format():
    code = "while True:\n    pass\n"
    test = {"input": "", "output": "", "testtype": "stdin"}
    passed, _kind, feedback = _run_one_test(code, test, function_name=None, timeout=1.0, test_index=5)
    assert passed is False
    assert "Test Case 5: Time Limit Exceeded" in feedback
    assert "Timeout: 1.0s" in feedback


@pytest.mark.parametrize("function_name", [None, "add"])
def test_extract_function_name_optional(function_name):
    """Smoke: stdin tests still work even when function_name is provided."""
    code = "x = int(input())\nprint(x)"
    test = {"input": "7\n", "output": "7\n", "testtype": "stdin"}
    passed, _kind, _ = _run_one_test(code, test, function_name=function_name, timeout=4.0)
    assert passed is True


# ── paper F.3 format regression tests ────────────────────────────────────────


def test_extract_python_code_picks_capitalized_fence():
    """Fence regex is case-insensitive (`\`\`\`Python\\n` matches)."""
    response = "thinking...\n```Python\nprint(42)\n```\n"
    assert extract_python_code(response) == "print(42)"


def test_extract_python_code_picks_fence_without_language_tag():
    response = "```\nprint('hello')\n```"
    assert extract_python_code(response) == "print('hello')"


def test_extract_python_code_picks_fence_with_py3_tag():
    response = "```py3\nprint('hello')\n```"
    assert extract_python_code(response) == "print('hello')"


def test_run_one_test_normalizes_per_line_whitespace():
    """Trailing whitespace on a non-last line shouldn't cause a false-fail."""
    code = "print('a ')\nprint('b')"
    test = {"input": "", "output": "a\nb\n", "testtype": "stdin"}
    passed, _kind, _ = _run_one_test(code, test, function_name=None, timeout=4.0)
    assert passed is True


def test_run_one_test_functional_multiline_json_input():
    """LCB LeetCode tests pass multiple args as JSON-per-line, e.g. "[6,8]\\n5".

    Before this fix the harness crashed with JSONDecodeError on multi-line inputs,
    silently false-negative-ing every LeetCode-style problem in LCBv6.
    """
    code = (
        "class Solution:\n"
        "    def maxScore(self, points, m):\n"
        "        return sum(points) + m\n"
    )
    test = {"input": "[6,8]\n5", "output": "19", "testtype": "functional"}
    passed, _kind, feedback = _run_one_test(code, test, function_name="maxScore", timeout=4.0)
    assert passed is True, feedback


def test_run_one_test_functional_three_args_json_per_line():
    """Three function arguments, each its own JSON line."""
    code = (
        "class Solution:\n"
        "    def f(self, a, b, c):\n"
        "        return a + b + c\n"
    )
    test = {"input": "1\n2\n3", "output": "6", "testtype": "functional"}
    passed, _kind, feedback = _run_one_test(code, test, function_name="f", timeout=4.0)
    assert passed is True, feedback


# ── input-cap regression tests ───────────────────────────────────────────────


def test_truncate_input_for_feedback_caps_huge_single_line():
    """Big single-line JSON arrays (atcoder/LeetCode) must be capped by chars,
    not just lines — the line cap alone is a no-op for them, and an unbounded
    single-line input would consume the entire aggregated feedback budget."""
    huge = "[" + ",".join(str(i) for i in range(2000)) + "]"
    assert len(huge) > 2000
    out = _truncate_input_for_feedback(huge, max_lines=40, max_chars=800)
    assert len(out) <= 800 + 50  # 800 head + "(N more chars)" suffix
    assert "more chars)" in out


def test_truncate_input_for_feedback_caps_by_lines_first():
    """Vertical inputs are line-capped (paper F.3 Listing 5 style)."""
    vertical = "\n".join(str(i) for i in range(100))
    out = _truncate_input_for_feedback(vertical, max_lines=8, max_chars=10000)
    assert "more lines)" in out
    assert out.count("\n") <= 8 + 1  # 8 retained lines + the suffix line


def test_run_one_test_huge_input_keeps_output_and_expected_visible():
    """A failed test with a 3000-char single-line input must still surface
    the Output and Expected sections in the feedback block — they shouldn't
    be lost to a giant Input dump consuming the entire char budget."""
    code = "data = input().split()\nprint(0)"  # always prints 0
    big_input = "[" + ",".join(str(i) for i in range(1000)) + "]\n"
    test = {"input": big_input, "output": "42", "testtype": "stdin"}
    passed, _kind, fb = _run_one_test(
        code, test, function_name=None, timeout=4.0, test_index=2,
        max_input_lines=40, max_input_chars=800,
    )
    assert passed is False
    assert "Test Case 2: Wrong Answer" in fb
    assert "\nOutput\n" in fb, fb
    assert "\nExpected\n" in fb, fb
    assert "42" in fb  # expected
    # Block should be reasonably sized; not blown out by the huge input
    assert len(fb) < 1500, len(fb)


# ── reference-harness grading behavior (grade_batch) ─────────────────────────

from nemo_rl.environments.livecodebench_environment import (  # noqa: E402
    INCORRECT_FORMAT_FEEDBACK,
    grade_batch,
)

_ADD_TESTS = [
    {"input": "1\n", "output": "2\n", "testtype": "stdin"},
    {"input": "5\n", "output": "10\n", "testtype": "stdin"},
    {"input": "0\n", "output": "0\n", "testtype": "stdin"},
    {"input": "7\n", "output": "14\n", "testtype": "stdin"},
]


def _grade_one(response, tests, split="train", dense=False, max_failed=2):
    meta = {"tests": tests, "split": split}
    return grade_batch(
        [response], [meta], timeout=4.0,
        max_feedback_chars=2000, max_failed_in_feedback=max_failed,
        max_input_lines=8, max_input_chars=250,
        dense_train_rewards=dense,
    )[0]


def test_grade_no_fence_is_incorrect_format():
    reward, feedback = _grade_one("I think the answer is doubling.", _ADD_TESTS)
    assert reward == 0.0
    assert feedback == INCORRECT_FORMAT_FEEDBACK


def test_grade_dense_train_reward_is_fraction_passed():
    # Doubler that breaks on input 0 -> passes 3 of 4 tests.
    code = "```python\nn = int(input())\nprint(n * 2 if n else 1)\n```"
    reward, _ = _grade_one(code, _ADD_TESTS, split="train", dense=True)
    assert reward == pytest.approx(3 / 4)


def test_grade_validation_stays_sparse_even_with_dense_flag():
    code = "```python\nn = int(input())\nprint(n * 2 if n else 1)\n```"
    reward, _ = _grade_one(code, _ADD_TESTS, split="validation", dense=True)
    assert reward == 0.0
    # and a fully correct solution still scores 1.0
    good = "```python\nprint(int(input()) * 2)\n```"
    reward, feedback = _grade_one(good, _ADD_TESTS, split="validation", dense=True)
    assert reward == 1.0 and "passed" in feedback


def test_grade_error_block_has_priority_and_is_single():
    # Crashes on input 0 (ZeroDivision), wrong answer elsewhere -> feedback
    # must be exactly one Runtime Error block, no Wrong Answer blocks.
    code = "```python\nn = int(input())\nprint(n * 2 + 1 // n)\n```"
    _, feedback = _grade_one(code, _ADD_TESTS)
    assert feedback.count("Runtime Error") == 1
    assert "Wrong Answer" not in feedback


def test_grade_wrong_answers_shortest_first_capped():
    tests = [
        {"input": "100000\n", "output": str(100000 * 2) + "\n", "testtype": "stdin"},
        {"input": "1\n", "output": "2\n", "testtype": "stdin"},
        {"input": "22\n", "output": "44\n", "testtype": "stdin"},
    ]
    code = "```python\nprint(int(input()) + 100)\n```"  # wrong everywhere
    _, feedback = _grade_one(code, tests, max_failed=2)
    # exactly two blocks, and the shortest (input "1") comes first
    assert feedback.count("Wrong Answer") == 2
    first_block = feedback.split("\n\n")[0]
    assert "Input\n1\n" in first_block
