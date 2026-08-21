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


def test_extract_python_code_falls_back_to_full_response():
    response = "no fence here\nimport math\nprint(math.pi)\n"
    assert "import math" in extract_python_code(response)


def test_extract_python_code_picks_last_fence():
    response = (
        "First attempt:\n```python\nprint('first')\n```\n"
        "Actually, the right one is:\n```python\nprint('second')\n```"
    )
    assert extract_python_code(response) == "print('second')"


def test_run_one_test_stdin_passes():
    code = "n = int(input())\nprint(n * 2)"
    test = {"input": "21\n", "output": "42\n", "testtype": "stdin"}
    passed, feedback = _run_one_test(code, test, function_name=None, timeout=4.0)
    assert passed is True
    assert feedback == ""


def test_run_one_test_stdin_wrong_answer_paper_format():
    """Wrong-answer feedback follows paper F.3 Listing 4 (`Test Case N: Wrong Answer / Input / Output / Expected`)."""
    code = "n = int(input())\nprint(n + 1)"  # off by one
    test = {"input": "21\n", "output": "42\n", "testtype": "stdin"}
    passed, feedback = _run_one_test(code, test, function_name=None, timeout=4.0, test_index=3)
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
    passed, feedback = _run_one_test(code, test, function_name=None, timeout=4.0)
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
    passed, feedback = _run_one_test(code, test, function_name="add", timeout=4.0)
    assert passed is True
    assert feedback == ""


def test_run_one_test_functional_wrong_answer_paper_format():
    code = (
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return a - b\n"  # bug
    )
    test = {"input": "[2, 3]", "output": "5", "testtype": "functional"}
    passed, feedback = _run_one_test(code, test, function_name="add", timeout=4.0, test_index=2)
    assert passed is False
    assert "Test Case 2: Wrong Answer" in feedback
    assert "Expected" in feedback


def test_run_one_test_timeout_paper_format():
    code = "while True:\n    pass\n"
    test = {"input": "", "output": "", "testtype": "stdin"}
    passed, feedback = _run_one_test(code, test, function_name=None, timeout=1.0, test_index=5)
    assert passed is False
    assert "Test Case 5: Time Limit Exceeded" in feedback
    assert "Timeout: 1.0s" in feedback


@pytest.mark.parametrize("function_name", [None, "add"])
def test_extract_function_name_optional(function_name):
    """Smoke: stdin tests still work even when function_name is provided."""
    code = "x = int(input())\nprint(x)"
    test = {"input": "7\n", "output": "7\n", "testtype": "stdin"}
    passed, _ = _run_one_test(code, test, function_name=function_name, timeout=4.0)
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
    passed, _ = _run_one_test(code, test, function_name=None, timeout=4.0)
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
    passed, feedback = _run_one_test(code, test, function_name="maxScore", timeout=4.0)
    assert passed is True, feedback


def test_run_one_test_functional_three_args_json_per_line():
    """Three function arguments, each its own JSON line."""
    code = (
        "class Solution:\n"
        "    def f(self, a, b, c):\n"
        "        return a + b + c\n"
    )
    test = {"input": "1\n2\n3", "output": "6", "testtype": "functional"}
    passed, feedback = _run_one_test(code, test, function_name="f", timeout=4.0)
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
    passed, fb = _run_one_test(
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
