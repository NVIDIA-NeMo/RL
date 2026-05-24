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

"""Build a JSONL dataset of simple Python coding tasks for GRPO training.

Each task asks the model to write a short Python function. The environment
verifies correctness by executing the function against test cases.

Usage:
    uv run python examples/data/build_nemorl_code_tasks.py
"""

import json
import os

TASKS = [
    # --- String matching ---
    {
        "problem": (
            "Write a Python function called `exact_match(a, b)` that returns True "
            "if strings a and b are exactly equal, False otherwise.\n\n"
            "Example:\n  exact_match('hello', 'hello') -> True\n  exact_match('hello', 'world') -> False"
        ),
        "test_cases": [
            {"input": "exact_match('hello', 'hello')", "expected": "True"},
            {"input": "exact_match('hello', 'world')", "expected": "False"},
            {"input": "exact_match('', '')", "expected": "True"},
            {"input": "exact_match('a', 'A')", "expected": "False"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `case_insensitive_match(a, b)` that returns True "
            "if strings a and b are equal ignoring case, False otherwise.\n\n"
            "Example:\n  case_insensitive_match('Hello', 'hello') -> True"
        ),
        "test_cases": [
            {"input": "case_insensitive_match('Hello', 'hello')", "expected": "True"},
            {"input": "case_insensitive_match('ABC', 'abc')", "expected": "True"},
            {"input": "case_insensitive_match('abc', 'xyz')", "expected": "False"},
            {"input": "case_insensitive_match('', '')", "expected": "True"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `starts_with(s, prefix)` that returns True "
            "if string s starts with prefix, False otherwise.\n\n"
            "Example:\n  starts_with('hello world', 'hello') -> True"
        ),
        "test_cases": [
            {"input": "starts_with('hello world', 'hello')", "expected": "True"},
            {"input": "starts_with('hello', 'world')", "expected": "False"},
            {"input": "starts_with('abc', '')", "expected": "True"},
            {"input": "starts_with('', 'a')", "expected": "False"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `ends_with(s, suffix)` that returns True "
            "if string s ends with suffix, False otherwise.\n\n"
            "Example:\n  ends_with('hello world', 'world') -> True"
        ),
        "test_cases": [
            {"input": "ends_with('hello world', 'world')", "expected": "True"},
            {"input": "ends_with('hello', 'world')", "expected": "False"},
            {"input": "ends_with('abc', '')", "expected": "True"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `count_char(s, c)` that returns the number "
            "of times character c appears in string s.\n\n"
            "Example:\n  count_char('banana', 'a') -> 3"
        ),
        "test_cases": [
            {"input": "count_char('banana', 'a')", "expected": "3"},
            {"input": "count_char('hello', 'l')", "expected": "2"},
            {"input": "count_char('hello', 'z')", "expected": "0"},
            {"input": "count_char('', 'a')", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `reverse_string(s)` that returns the string s reversed.\n\n"
            "Example:\n  reverse_string('hello') -> 'olleh'"
        ),
        "test_cases": [
            {"input": "reverse_string('hello')", "expected": "olleh"},
            {"input": "reverse_string('abc')", "expected": "cba"},
            {"input": "reverse_string('')", "expected": ""},
            {"input": "reverse_string('a')", "expected": "a"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `is_palindrome(s)` that returns True if "
            "string s reads the same forwards and backwards, False otherwise.\n\n"
            "Example:\n  is_palindrome('racecar') -> True"
        ),
        "test_cases": [
            {"input": "is_palindrome('racecar')", "expected": "True"},
            {"input": "is_palindrome('hello')", "expected": "False"},
            {"input": "is_palindrome('aba')", "expected": "True"},
            {"input": "is_palindrome('')", "expected": "True"},
        ],
    },
    # --- Number operations ---
    {
        "problem": (
            "Write a Python function called `add(a, b)` that returns the sum of "
            "two numbers a and b.\n\n"
            "Example:\n  add(2, 3) -> 5"
        ),
        "test_cases": [
            {"input": "add(2, 3)", "expected": "5"},
            {"input": "add(0, 0)", "expected": "0"},
            {"input": "add(-1, 1)", "expected": "0"},
            {"input": "add(10, 20)", "expected": "30"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `multiply(a, b)` that returns the product of "
            "two numbers a and b.\n\n"
            "Example:\n  multiply(3, 4) -> 12"
        ),
        "test_cases": [
            {"input": "multiply(3, 4)", "expected": "12"},
            {"input": "multiply(0, 5)", "expected": "0"},
            {"input": "multiply(-2, 3)", "expected": "-6"},
            {"input": "multiply(7, 7)", "expected": "49"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `absolute(n)` that returns the absolute value of n.\n\n"
            "Example:\n  absolute(-5) -> 5"
        ),
        "test_cases": [
            {"input": "absolute(-5)", "expected": "5"},
            {"input": "absolute(5)", "expected": "5"},
            {"input": "absolute(0)", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `maximum(a, b)` that returns the larger of "
            "two numbers a and b.\n\n"
            "Example:\n  maximum(3, 7) -> 7"
        ),
        "test_cases": [
            {"input": "maximum(3, 7)", "expected": "7"},
            {"input": "maximum(10, 2)", "expected": "10"},
            {"input": "maximum(5, 5)", "expected": "5"},
            {"input": "maximum(-1, -5)", "expected": "-1"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `minimum(a, b)` that returns the smaller of "
            "two numbers a and b.\n\n"
            "Example:\n  minimum(3, 7) -> 3"
        ),
        "test_cases": [
            {"input": "minimum(3, 7)", "expected": "3"},
            {"input": "minimum(10, 2)", "expected": "2"},
            {"input": "minimum(5, 5)", "expected": "5"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `clamp(x, lo, hi)` that returns x clamped "
            "to the range [lo, hi]. If x < lo return lo, if x > hi return hi, else return x.\n\n"
            "Example:\n  clamp(5, 0, 10) -> 5\n  clamp(-3, 0, 10) -> 0"
        ),
        "test_cases": [
            {"input": "clamp(5, 0, 10)", "expected": "5"},
            {"input": "clamp(-3, 0, 10)", "expected": "0"},
            {"input": "clamp(15, 0, 10)", "expected": "10"},
            {"input": "clamp(0, 0, 10)", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `is_even(n)` that returns True if n is even, "
            "False otherwise.\n\n"
            "Example:\n  is_even(4) -> True\n  is_even(3) -> False"
        ),
        "test_cases": [
            {"input": "is_even(4)", "expected": "True"},
            {"input": "is_even(3)", "expected": "False"},
            {"input": "is_even(0)", "expected": "True"},
            {"input": "is_even(-2)", "expected": "True"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `is_positive(n)` that returns True if n > 0, "
            "False otherwise.\n\n"
            "Example:\n  is_positive(5) -> True\n  is_positive(-3) -> False"
        ),
        "test_cases": [
            {"input": "is_positive(5)", "expected": "True"},
            {"input": "is_positive(-3)", "expected": "False"},
            {"input": "is_positive(0)", "expected": "False"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `sign(n)` that returns 1 if n > 0, "
            "-1 if n < 0, and 0 if n == 0.\n\n"
            "Example:\n  sign(5) -> 1\n  sign(-3) -> -1\n  sign(0) -> 0"
        ),
        "test_cases": [
            {"input": "sign(5)", "expected": "1"},
            {"input": "sign(-3)", "expected": "-1"},
            {"input": "sign(0)", "expected": "0"},
        ],
    },
    # --- List operations ---
    {
        "problem": (
            "Write a Python function called `sum_list(lst)` that returns the sum of "
            "all numbers in the list lst.\n\n"
            "Example:\n  sum_list([1, 2, 3]) -> 6"
        ),
        "test_cases": [
            {"input": "sum_list([1, 2, 3])", "expected": "6"},
            {"input": "sum_list([])", "expected": "0"},
            {"input": "sum_list([10])", "expected": "10"},
            {"input": "sum_list([-1, 1])", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `mean_list(lst)` that returns the arithmetic mean "
            "of a non-empty list of numbers.\n\n"
            "Example:\n  mean_list([1, 2, 3]) -> 2.0"
        ),
        "test_cases": [
            {"input": "mean_list([1, 2, 3])", "expected": "2.0"},
            {"input": "mean_list([10])", "expected": "10.0"},
            {"input": "mean_list([0, 0, 0])", "expected": "0.0"},
            {"input": "mean_list([2, 4])", "expected": "3.0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `max_list(lst)` that returns the maximum value "
            "in a non-empty list of numbers.\n\n"
            "Example:\n  max_list([3, 1, 4, 1, 5]) -> 5"
        ),
        "test_cases": [
            {"input": "max_list([3, 1, 4, 1, 5])", "expected": "5"},
            {"input": "max_list([1])", "expected": "1"},
            {"input": "max_list([-1, -5, -2])", "expected": "-1"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `min_list(lst)` that returns the minimum value "
            "in a non-empty list of numbers.\n\n"
            "Example:\n  min_list([3, 1, 4, 1, 5]) -> 1"
        ),
        "test_cases": [
            {"input": "min_list([3, 1, 4, 1, 5])", "expected": "1"},
            {"input": "min_list([1])", "expected": "1"},
            {"input": "min_list([-1, -5, -2])", "expected": "-5"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `length(lst)` that returns the number of "
            "elements in the list lst.\n\n"
            "Example:\n  length([1, 2, 3]) -> 3"
        ),
        "test_cases": [
            {"input": "length([1, 2, 3])", "expected": "3"},
            {"input": "length([])", "expected": "0"},
            {"input": "length(['a'])", "expected": "1"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `first_element(lst)` that returns the first "
            "element of a non-empty list.\n\n"
            "Example:\n  first_element([10, 20, 30]) -> 10"
        ),
        "test_cases": [
            {"input": "first_element([10, 20, 30])", "expected": "10"},
            {"input": "first_element(['a', 'b'])", "expected": "a"},
            {"input": "first_element([42])", "expected": "42"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `last_element(lst)` that returns the last "
            "element of a non-empty list.\n\n"
            "Example:\n  last_element([10, 20, 30]) -> 30"
        ),
        "test_cases": [
            {"input": "last_element([10, 20, 30])", "expected": "30"},
            {"input": "last_element(['a'])", "expected": "a"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `flatten(lst)` that takes a list of lists "
            "and returns a single flat list with all elements.\n\n"
            "Example:\n  flatten([[1, 2], [3, 4]]) -> [1, 2, 3, 4]"
        ),
        "test_cases": [
            {"input": "flatten([[1, 2], [3, 4]])", "expected": "[1, 2, 3, 4]"},
            {"input": "flatten([[], [1]])", "expected": "[1]"},
            {"input": "flatten([])", "expected": "[]"},
            {"input": "flatten([[1], [2], [3]])", "expected": "[1, 2, 3]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `chunk_list(lst, n)` that splits list lst "
            "into sublists of size n. The last chunk may be smaller.\n\n"
            "Example:\n  chunk_list([1, 2, 3, 4, 5], 2) -> [[1, 2], [3, 4], [5]]"
        ),
        "test_cases": [
            {
                "input": "chunk_list([1, 2, 3, 4, 5], 2)",
                "expected": "[[1, 2], [3, 4], [5]]",
            },
            {"input": "chunk_list([1, 2, 3, 4], 2)", "expected": "[[1, 2], [3, 4]]"},
            {"input": "chunk_list([], 3)", "expected": "[]"},
            {"input": "chunk_list([1], 5)", "expected": "[[1]]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `unique(lst)` that returns a list of "
            "unique elements from lst, preserving order of first occurrence.\n\n"
            "Example:\n  unique([1, 2, 2, 3, 1]) -> [1, 2, 3]"
        ),
        "test_cases": [
            {"input": "unique([1, 2, 2, 3, 1])", "expected": "[1, 2, 3]"},
            {"input": "unique([])", "expected": "[]"},
            {"input": "unique([1, 1, 1])", "expected": "[1]"},
            {"input": "unique([3, 2, 1])", "expected": "[3, 2, 1]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `contains(lst, item)` that returns True "
            "if item is in lst, False otherwise.\n\n"
            "Example:\n  contains([1, 2, 3], 2) -> True"
        ),
        "test_cases": [
            {"input": "contains([1, 2, 3], 2)", "expected": "True"},
            {"input": "contains([1, 2, 3], 4)", "expected": "False"},
            {"input": "contains([], 1)", "expected": "False"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `count_item(lst, item)` that returns how many "
            "times item appears in lst.\n\n"
            "Example:\n  count_item([1, 2, 2, 3], 2) -> 2"
        ),
        "test_cases": [
            {"input": "count_item([1, 2, 2, 3], 2)", "expected": "2"},
            {"input": "count_item([1, 1, 1], 1)", "expected": "3"},
            {"input": "count_item([], 1)", "expected": "0"},
            {"input": "count_item([1, 2, 3], 4)", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `zip_lists(a, b)` that returns a list of tuples "
            "pairing elements from lists a and b. Stop at the shorter list.\n\n"
            "Example:\n  zip_lists([1, 2], ['a', 'b']) -> [(1, 'a'), (2, 'b')]"
        ),
        "test_cases": [
            {
                "input": "zip_lists([1, 2], ['a', 'b'])",
                "expected": "[(1, 'a'), (2, 'b')]",
            },
            {"input": "zip_lists([], [])", "expected": "[]"},
            {"input": "zip_lists([1, 2, 3], ['a'])", "expected": "[(1, 'a')]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `filter_positive(lst)` that returns a new list "
            "containing only the positive numbers from lst.\n\n"
            "Example:\n  filter_positive([-1, 2, -3, 4]) -> [2, 4]"
        ),
        "test_cases": [
            {"input": "filter_positive([-1, 2, -3, 4])", "expected": "[2, 4]"},
            {"input": "filter_positive([])", "expected": "[]"},
            {"input": "filter_positive([-1, -2])", "expected": "[]"},
            {"input": "filter_positive([1, 2, 3])", "expected": "[1, 2, 3]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `double_list(lst)` that returns a new list "
            "with each element doubled.\n\n"
            "Example:\n  double_list([1, 2, 3]) -> [2, 4, 6]"
        ),
        "test_cases": [
            {"input": "double_list([1, 2, 3])", "expected": "[2, 4, 6]"},
            {"input": "double_list([])", "expected": "[]"},
            {"input": "double_list([0, -1])", "expected": "[0, -2]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `square_list(lst)` that returns a new list "
            "with each element squared.\n\n"
            "Example:\n  square_list([1, 2, 3]) -> [1, 4, 9]"
        ),
        "test_cases": [
            {"input": "square_list([1, 2, 3])", "expected": "[1, 4, 9]"},
            {"input": "square_list([])", "expected": "[]"},
            {"input": "square_list([-2, 0, 2])", "expected": "[4, 0, 4]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `reverse_list(lst)` that returns a new list "
            "with elements in reverse order.\n\n"
            "Example:\n  reverse_list([1, 2, 3]) -> [3, 2, 1]"
        ),
        "test_cases": [
            {"input": "reverse_list([1, 2, 3])", "expected": "[3, 2, 1]"},
            {"input": "reverse_list([])", "expected": "[]"},
            {"input": "reverse_list([42])", "expected": "[42]"},
        ],
    },
    # --- Dict / config operations ---
    {
        "problem": (
            "Write a Python function called `has_key(d, key)` that returns True if "
            "dictionary d contains the given key, False otherwise.\n\n"
            "Example:\n  has_key({'a': 1, 'b': 2}, 'a') -> True"
        ),
        "test_cases": [
            {"input": "has_key({'a': 1, 'b': 2}, 'a')", "expected": "True"},
            {"input": "has_key({'a': 1}, 'b')", "expected": "False"},
            {"input": "has_key({}, 'a')", "expected": "False"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `get_value(d, key, default)` that returns "
            "d[key] if key exists in dict d, otherwise returns default.\n\n"
            "Example:\n  get_value({'a': 1}, 'a', 0) -> 1\n  get_value({'a': 1}, 'b', 0) -> 0"
        ),
        "test_cases": [
            {"input": "get_value({'a': 1}, 'a', 0)", "expected": "1"},
            {"input": "get_value({'a': 1}, 'b', 0)", "expected": "0"},
            {"input": "get_value({}, 'x', 42)", "expected": "42"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `merge_dicts(a, b)` that returns a new dictionary "
            "with all keys from both a and b. If a key exists in both, use the value from b.\n\n"
            "Example:\n  merge_dicts({'a': 1}, {'b': 2}) -> {'a': 1, 'b': 2}"
        ),
        "test_cases": [
            {
                "input": "merge_dicts({'a': 1}, {'b': 2})",
                "expected": "{'a': 1, 'b': 2}",
            },
            {"input": "merge_dicts({'a': 1}, {'a': 2})", "expected": "{'a': 2}"},
            {"input": "merge_dicts({}, {'a': 1})", "expected": "{'a': 1}"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `keys_list(d)` that returns a sorted list "
            "of all keys in dictionary d.\n\n"
            "Example:\n  keys_list({'b': 2, 'a': 1}) -> ['a', 'b']"
        ),
        "test_cases": [
            {"input": "keys_list({'b': 2, 'a': 1})", "expected": "['a', 'b']"},
            {"input": "keys_list({})", "expected": "[]"},
            {"input": "keys_list({'z': 1})", "expected": "['z']"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `values_list(d)` that returns a list of all "
            "values in dictionary d, sorted by their keys.\n\n"
            "Example:\n  values_list({'b': 2, 'a': 1}) -> [1, 2]"
        ),
        "test_cases": [
            {"input": "values_list({'b': 2, 'a': 1})", "expected": "[1, 2]"},
            {"input": "values_list({})", "expected": "[]"},
            {"input": "values_list({'x': 10})", "expected": "[10]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `has_required_keys(d, keys)` that returns True "
            "if dictionary d contains all keys in the list keys, False otherwise.\n\n"
            "Example:\n  has_required_keys({'a': 1, 'b': 2}, ['a', 'b']) -> True"
        ),
        "test_cases": [
            {
                "input": "has_required_keys({'a': 1, 'b': 2}, ['a', 'b'])",
                "expected": "True",
            },
            {"input": "has_required_keys({'a': 1}, ['a', 'b'])", "expected": "False"},
            {
                "input": "has_required_keys({'a': 1, 'b': 2, 'c': 3}, ['a'])",
                "expected": "True",
            },
            {"input": "has_required_keys({}, [])", "expected": "True"},
        ],
    },
    # --- Reward / scoring functions ---
    {
        "problem": (
            "Write a Python function called `binary_reward(predicted, expected)` that returns "
            "1.0 if predicted equals expected (both strings), 0.0 otherwise.\n\n"
            "Example:\n  binary_reward('42', '42') -> 1.0\n  binary_reward('41', '42') -> 0.0"
        ),
        "test_cases": [
            {"input": "binary_reward('42', '42')", "expected": "1.0"},
            {"input": "binary_reward('41', '42')", "expected": "0.0"},
            {"input": "binary_reward('', '')", "expected": "1.0"},
            {"input": "binary_reward('hello', 'Hello')", "expected": "0.0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `partial_reward(predicted, expected)` that returns "
            "a float between 0.0 and 1.0 representing the fraction of characters that match "
            "at the same position. Compare up to the length of the shorter string, "
            "then divide by the length of the longer string.\n\n"
            "Example:\n  partial_reward('abc', 'abc') -> 1.0\n  partial_reward('abc', 'axc') -> 0.6666666666666666"
        ),
        "test_cases": [
            {"input": "partial_reward('abc', 'abc')", "expected": "1.0"},
            {"input": "partial_reward('abc', 'xyz')", "expected": "0.0"},
            {"input": "partial_reward('ab', 'abcd')", "expected": "0.5"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `threshold_reward(score, threshold)` that returns "
            "1.0 if score >= threshold, 0.0 otherwise. Both are floats.\n\n"
            "Example:\n  threshold_reward(0.8, 0.5) -> 1.0\n  threshold_reward(0.3, 0.5) -> 0.0"
        ),
        "test_cases": [
            {"input": "threshold_reward(0.8, 0.5)", "expected": "1.0"},
            {"input": "threshold_reward(0.3, 0.5)", "expected": "0.0"},
            {"input": "threshold_reward(0.5, 0.5)", "expected": "1.0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `scale_reward(value, lo, hi)` that linearly "
            "scales value from [lo, hi] to [0.0, 1.0]. Clamp the result to [0.0, 1.0].\n\n"
            "Example:\n  scale_reward(5, 0, 10) -> 0.5\n  scale_reward(15, 0, 10) -> 1.0"
        ),
        "test_cases": [
            {"input": "scale_reward(5, 0, 10)", "expected": "0.5"},
            {"input": "scale_reward(0, 0, 10)", "expected": "0.0"},
            {"input": "scale_reward(10, 0, 10)", "expected": "1.0"},
            {"input": "scale_reward(15, 0, 10)", "expected": "1.0"},
            {"input": "scale_reward(-5, 0, 10)", "expected": "0.0"},
        ],
    },
    # --- Expression evaluation ---
    {
        "problem": (
            "Write a Python function called `safe_divide(a, b)` that returns a / b if b != 0, "
            "otherwise returns 0.0.\n\n"
            "Example:\n  safe_divide(10, 2) -> 5.0\n  safe_divide(10, 0) -> 0.0"
        ),
        "test_cases": [
            {"input": "safe_divide(10, 2)", "expected": "5.0"},
            {"input": "safe_divide(10, 0)", "expected": "0.0"},
            {"input": "safe_divide(0, 5)", "expected": "0.0"},
            {"input": "safe_divide(7, 2)", "expected": "3.5"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `power(base, exp)` that returns base raised "
            "to the power exp. Both are non-negative integers.\n\n"
            "Example:\n  power(2, 3) -> 8"
        ),
        "test_cases": [
            {"input": "power(2, 3)", "expected": "8"},
            {"input": "power(5, 0)", "expected": "1"},
            {"input": "power(3, 2)", "expected": "9"},
            {"input": "power(10, 1)", "expected": "10"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `factorial(n)` that returns the factorial of "
            "non-negative integer n. factorial(0) = 1.\n\n"
            "Example:\n  factorial(5) -> 120"
        ),
        "test_cases": [
            {"input": "factorial(5)", "expected": "120"},
            {"input": "factorial(0)", "expected": "1"},
            {"input": "factorial(1)", "expected": "1"},
            {"input": "factorial(3)", "expected": "6"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `fibonacci(n)` that returns the n-th Fibonacci "
            "number (0-indexed). fib(0)=0, fib(1)=1, fib(n)=fib(n-1)+fib(n-2).\n\n"
            "Example:\n  fibonacci(6) -> 8"
        ),
        "test_cases": [
            {"input": "fibonacci(0)", "expected": "0"},
            {"input": "fibonacci(1)", "expected": "1"},
            {"input": "fibonacci(6)", "expected": "8"},
            {"input": "fibonacci(10)", "expected": "55"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `gcd(a, b)` that returns the greatest common "
            "divisor of two positive integers a and b.\n\n"
            "Example:\n  gcd(12, 8) -> 4"
        ),
        "test_cases": [
            {"input": "gcd(12, 8)", "expected": "4"},
            {"input": "gcd(7, 3)", "expected": "1"},
            {"input": "gcd(100, 25)", "expected": "25"},
            {"input": "gcd(6, 6)", "expected": "6"},
        ],
    },
    # --- String formatting / parsing ---
    {
        "problem": (
            "Write a Python function called `to_upper(s)` that returns the string s "
            "converted to uppercase.\n\n"
            "Example:\n  to_upper('hello') -> 'HELLO'"
        ),
        "test_cases": [
            {"input": "to_upper('hello')", "expected": "HELLO"},
            {"input": "to_upper('Hello World')", "expected": "HELLO WORLD"},
            {"input": "to_upper('')", "expected": ""},
        ],
    },
    {
        "problem": (
            "Write a Python function called `to_lower(s)` that returns the string s "
            "converted to lowercase.\n\n"
            "Example:\n  to_lower('HELLO') -> 'hello'"
        ),
        "test_cases": [
            {"input": "to_lower('HELLO')", "expected": "hello"},
            {"input": "to_lower('Hello World')", "expected": "hello world"},
            {"input": "to_lower('')", "expected": ""},
        ],
    },
    {
        "problem": (
            "Write a Python function called `repeat_string(s, n)` that returns string s "
            "repeated n times.\n\n"
            "Example:\n  repeat_string('ab', 3) -> 'ababab'"
        ),
        "test_cases": [
            {"input": "repeat_string('ab', 3)", "expected": "ababab"},
            {"input": "repeat_string('x', 5)", "expected": "xxxxx"},
            {"input": "repeat_string('hi', 0)", "expected": ""},
            {"input": "repeat_string('', 10)", "expected": ""},
        ],
    },
    {
        "problem": (
            "Write a Python function called `truncate(s, max_len)` that returns the first "
            "max_len characters of string s. If s is shorter, return s as-is.\n\n"
            "Example:\n  truncate('hello world', 5) -> 'hello'"
        ),
        "test_cases": [
            {"input": "truncate('hello world', 5)", "expected": "hello"},
            {"input": "truncate('hi', 10)", "expected": "hi"},
            {"input": "truncate('', 5)", "expected": ""},
        ],
    },
    {
        "problem": (
            "Write a Python function called `count_words(s)` that returns the number "
            "of whitespace-separated words in string s.\n\n"
            "Example:\n  count_words('hello world') -> 2"
        ),
        "test_cases": [
            {"input": "count_words('hello world')", "expected": "2"},
            {"input": "count_words('one')", "expected": "1"},
            {"input": "count_words('')", "expected": "0"},
            {"input": "count_words('a b c d')", "expected": "4"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `join_strings(lst, sep)` that joins a list "
            "of strings with separator sep.\n\n"
            "Example:\n  join_strings(['a', 'b', 'c'], '-') -> 'a-b-c'"
        ),
        "test_cases": [
            {"input": "join_strings(['a', 'b', 'c'], '-')", "expected": "a-b-c"},
            {
                "input": "join_strings(['hello', 'world'], ' ')",
                "expected": "hello world",
            },
            {"input": "join_strings([], ',')", "expected": ""},
            {"input": "join_strings(['only'], ',')", "expected": "only"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `split_string(s, sep)` that splits string s "
            "by separator sep and returns a list of parts.\n\n"
            "Example:\n  split_string('a-b-c', '-') -> ['a', 'b', 'c']"
        ),
        "test_cases": [
            {"input": "split_string('a-b-c', '-')", "expected": "['a', 'b', 'c']"},
            {"input": "split_string('hello', ',')", "expected": "['hello']"},
            {"input": "split_string('a,b', ',')", "expected": "['a', 'b']"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `strip_whitespace(s)` that returns string s "
            "with leading and trailing whitespace removed.\n\n"
            "Example:\n  strip_whitespace('  hello  ') -> 'hello'"
        ),
        "test_cases": [
            {"input": "strip_whitespace('  hello  ')", "expected": "hello"},
            {"input": "strip_whitespace('no_space')", "expected": "no_space"},
            {"input": "strip_whitespace('  ')", "expected": ""},
            {"input": "strip_whitespace('')", "expected": ""},
        ],
    },
    {
        "problem": (
            "Write a Python function called `replace_char(s, old, new)` that returns a new "
            "string with all occurrences of character old replaced by character new.\n\n"
            "Example:\n  replace_char('hello', 'l', 'r') -> 'herro'"
        ),
        "test_cases": [
            {"input": "replace_char('hello', 'l', 'r')", "expected": "herro"},
            {"input": "replace_char('aaa', 'a', 'b')", "expected": "bbb"},
            {"input": "replace_char('xyz', 'a', 'b')", "expected": "xyz"},
        ],
    },
    # --- Number parsing ---
    {
        "problem": (
            "Write a Python function called `extract_numbers(s)` that takes a string "
            "and returns a list of all integers found in it.\n\n"
            "Example:\n  extract_numbers('I have 3 cats and 2 dogs') -> [3, 2]"
        ),
        "test_cases": [
            {
                "input": "extract_numbers('I have 3 cats and 2 dogs')",
                "expected": "[3, 2]",
            },
            {"input": "extract_numbers('no numbers here')", "expected": "[]"},
            {"input": "extract_numbers('42')", "expected": "[42]"},
            {"input": "extract_numbers('a1b2c3')", "expected": "[1, 2, 3]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `is_digit_string(s)` that returns True if "
            "all characters in string s are digits and s is non-empty, False otherwise.\n\n"
            "Example:\n  is_digit_string('123') -> True\n  is_digit_string('12a') -> False"
        ),
        "test_cases": [
            {"input": "is_digit_string('123')", "expected": "True"},
            {"input": "is_digit_string('12a')", "expected": "False"},
            {"input": "is_digit_string('')", "expected": "False"},
            {"input": "is_digit_string('0')", "expected": "True"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `to_int_or_none(s)` that tries to convert "
            "string s to an integer. Return the integer if successful, None otherwise.\n\n"
            "Example:\n  to_int_or_none('42') -> 42\n  to_int_or_none('abc') -> None"
        ),
        "test_cases": [
            {"input": "to_int_or_none('42')", "expected": "42"},
            {"input": "to_int_or_none('abc')", "expected": "None"},
            {"input": "to_int_or_none('-7')", "expected": "-7"},
            {"input": "to_int_or_none('')", "expected": "None"},
        ],
    },
    # --- Boolean / logic ---
    {
        "problem": (
            "Write a Python function called `all_true(lst)` that returns True if all "
            "elements in the list are True (or truthy), False otherwise. "
            "Returns True for an empty list.\n\n"
            "Example:\n  all_true([True, True, True]) -> True"
        ),
        "test_cases": [
            {"input": "all_true([True, True, True])", "expected": "True"},
            {"input": "all_true([True, False, True])", "expected": "False"},
            {"input": "all_true([])", "expected": "True"},
            {"input": "all_true([1, 2, 3])", "expected": "True"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `any_true(lst)` that returns True if any "
            "element in the list is True (or truthy), False otherwise. "
            "Returns False for an empty list.\n\n"
            "Example:\n  any_true([False, True, False]) -> True"
        ),
        "test_cases": [
            {"input": "any_true([False, True, False])", "expected": "True"},
            {"input": "any_true([False, False])", "expected": "False"},
            {"input": "any_true([])", "expected": "False"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `none_true(lst)` that returns True if none "
            "of the elements in the list are True (or truthy), False otherwise. "
            "Returns True for an empty list.\n\n"
            "Example:\n  none_true([False, False]) -> True"
        ),
        "test_cases": [
            {"input": "none_true([False, False])", "expected": "True"},
            {"input": "none_true([False, True])", "expected": "False"},
            {"input": "none_true([])", "expected": "True"},
        ],
    },
    # --- Format conversion ---
    {
        "problem": (
            "Write a Python function called `to_csv_line(lst)` that takes a list of strings "
            "and returns them joined by commas as a single string.\n\n"
            "Example:\n  to_csv_line(['a', 'b', 'c']) -> 'a,b,c'"
        ),
        "test_cases": [
            {"input": "to_csv_line(['a', 'b', 'c'])", "expected": "a,b,c"},
            {"input": "to_csv_line(['hello'])", "expected": "hello"},
            {"input": "to_csv_line([])", "expected": ""},
        ],
    },
    {
        "problem": (
            "Write a Python function called `from_csv_line(s)` that takes a comma-separated "
            "string and returns a list of the parts.\n\n"
            "Example:\n  from_csv_line('a,b,c') -> ['a', 'b', 'c']"
        ),
        "test_cases": [
            {"input": "from_csv_line('a,b,c')", "expected": "['a', 'b', 'c']"},
            {"input": "from_csv_line('hello')", "expected": "['hello']"},
            {"input": "from_csv_line('1,2')", "expected": "['1', '2']"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `dict_to_pairs(d)` that converts a dictionary "
            "to a sorted list of (key, value) tuples, sorted by key.\n\n"
            "Example:\n  dict_to_pairs({'b': 2, 'a': 1}) -> [('a', 1), ('b', 2)]"
        ),
        "test_cases": [
            {
                "input": "dict_to_pairs({'b': 2, 'a': 1})",
                "expected": "[('a', 1), ('b', 2)]",
            },
            {"input": "dict_to_pairs({})", "expected": "[]"},
            {"input": "dict_to_pairs({'x': 10})", "expected": "[('x', 10)]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `pairs_to_dict(pairs)` that converts a list "
            "of (key, value) tuples into a dictionary.\n\n"
            "Example:\n  pairs_to_dict([('a', 1), ('b', 2)]) -> {'a': 1, 'b': 2}"
        ),
        "test_cases": [
            {
                "input": "pairs_to_dict([('a', 1), ('b', 2)])",
                "expected": "{'a': 1, 'b': 2}",
            },
            {"input": "pairs_to_dict([])", "expected": "{}"},
        ],
    },
    # --- Range / sequence generation ---
    {
        "problem": (
            "Write a Python function called `make_range(n)` that returns a list of integers "
            "from 0 to n-1.\n\n"
            "Example:\n  make_range(5) -> [0, 1, 2, 3, 4]"
        ),
        "test_cases": [
            {"input": "make_range(5)", "expected": "[0, 1, 2, 3, 4]"},
            {"input": "make_range(0)", "expected": "[]"},
            {"input": "make_range(1)", "expected": "[0]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `make_range_from_to(start, end)` that returns "
            "a list of integers from start to end-1.\n\n"
            "Example:\n  make_range_from_to(2, 5) -> [2, 3, 4]"
        ),
        "test_cases": [
            {"input": "make_range_from_to(2, 5)", "expected": "[2, 3, 4]"},
            {"input": "make_range_from_to(0, 3)", "expected": "[0, 1, 2]"},
            {"input": "make_range_from_to(5, 5)", "expected": "[]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `enumerate_items(lst)` that returns a list of "
            "tuples (index, value) for each element in lst.\n\n"
            "Example:\n  enumerate_items(['a', 'b', 'c']) -> [(0, 'a'), (1, 'b'), (2, 'c')]"
        ),
        "test_cases": [
            {
                "input": "enumerate_items(['a', 'b', 'c'])",
                "expected": "[(0, 'a'), (1, 'b'), (2, 'c')]",
            },
            {"input": "enumerate_items([])", "expected": "[]"},
            {"input": "enumerate_items([42])", "expected": "[(0, 42)]"},
        ],
    },
    # --- Sorting ---
    {
        "problem": (
            "Write a Python function called `sort_ascending(lst)` that returns a new list "
            "with elements sorted in ascending order.\n\n"
            "Example:\n  sort_ascending([3, 1, 2]) -> [1, 2, 3]"
        ),
        "test_cases": [
            {"input": "sort_ascending([3, 1, 2])", "expected": "[1, 2, 3]"},
            {"input": "sort_ascending([])", "expected": "[]"},
            {"input": "sort_ascending([5, 5, 5])", "expected": "[5, 5, 5]"},
            {"input": "sort_ascending([1])", "expected": "[1]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `sort_descending(lst)` that returns a new list "
            "with elements sorted in descending order.\n\n"
            "Example:\n  sort_descending([3, 1, 2]) -> [3, 2, 1]"
        ),
        "test_cases": [
            {"input": "sort_descending([3, 1, 2])", "expected": "[3, 2, 1]"},
            {"input": "sort_descending([])", "expected": "[]"},
            {"input": "sort_descending([1])", "expected": "[1]"},
        ],
    },
    # --- Misc simple utility functions ---
    {
        "problem": (
            "Write a Python function called `identity(x)` that simply returns x unchanged.\n\n"
            "Example:\n  identity(42) -> 42\n  identity('hello') -> 'hello'"
        ),
        "test_cases": [
            {"input": "identity(42)", "expected": "42"},
            {"input": "identity('hello')", "expected": "hello"},
            {"input": "identity(None)", "expected": "None"},
            {"input": "identity([1, 2])", "expected": "[1, 2]"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `constant_zero()` that takes no arguments "
            "and always returns 0.\n\n"
            "Example:\n  constant_zero() -> 0"
        ),
        "test_cases": [
            {"input": "constant_zero()", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `negate(n)` that returns the negation of number n.\n\n"
            "Example:\n  negate(5) -> -5\n  negate(-3) -> 3"
        ),
        "test_cases": [
            {"input": "negate(5)", "expected": "-5"},
            {"input": "negate(-3)", "expected": "3"},
            {"input": "negate(0)", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `increment(n)` that returns n + 1.\n\n"
            "Example:\n  increment(4) -> 5"
        ),
        "test_cases": [
            {"input": "increment(4)", "expected": "5"},
            {"input": "increment(0)", "expected": "1"},
            {"input": "increment(-1)", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `decrement(n)` that returns n - 1.\n\n"
            "Example:\n  decrement(4) -> 3"
        ),
        "test_cases": [
            {"input": "decrement(4)", "expected": "3"},
            {"input": "decrement(0)", "expected": "-1"},
            {"input": "decrement(1)", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `swap(a, b)` that returns a tuple (b, a).\n\n"
            "Example:\n  swap(1, 2) -> (2, 1)"
        ),
        "test_cases": [
            {"input": "swap(1, 2)", "expected": "(2, 1)"},
            {"input": "swap('a', 'b')", "expected": "('b', 'a')"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `pair(a, b)` that returns a tuple (a, b).\n\n"
            "Example:\n  pair(1, 2) -> (1, 2)"
        ),
        "test_cases": [
            {"input": "pair(1, 2)", "expected": "(1, 2)"},
            {"input": "pair('x', 'y')", "expected": "('x', 'y')"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `is_empty(lst)` that returns True if the list "
            "is empty, False otherwise.\n\n"
            "Example:\n  is_empty([]) -> True\n  is_empty([1]) -> False"
        ),
        "test_cases": [
            {"input": "is_empty([])", "expected": "True"},
            {"input": "is_empty([1])", "expected": "False"},
            {"input": "is_empty([1, 2, 3])", "expected": "False"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `string_length(s)` that returns the number "
            "of characters in string s.\n\n"
            "Example:\n  string_length('hello') -> 5"
        ),
        "test_cases": [
            {"input": "string_length('hello')", "expected": "5"},
            {"input": "string_length('')", "expected": "0"},
            {"input": "string_length('a')", "expected": "1"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `sum_two(a, b)` that returns a + b where "
            "a and b are numbers.\n\n"
            "Example:\n  sum_two(3, 4) -> 7"
        ),
        "test_cases": [
            {"input": "sum_two(3, 4)", "expected": "7"},
            {"input": "sum_two(0, 0)", "expected": "0"},
            {"input": "sum_two(-1, 1)", "expected": "0"},
            {"input": "sum_two(100, 200)", "expected": "300"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `subtract(a, b)` that returns a - b.\n\n"
            "Example:\n  subtract(10, 3) -> 7"
        ),
        "test_cases": [
            {"input": "subtract(10, 3)", "expected": "7"},
            {"input": "subtract(0, 0)", "expected": "0"},
            {"input": "subtract(1, 5)", "expected": "-4"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `integer_divide(a, b)` that returns the integer "
            "division of a by b (floor division).\n\n"
            "Example:\n  integer_divide(7, 2) -> 3"
        ),
        "test_cases": [
            {"input": "integer_divide(7, 2)", "expected": "3"},
            {"input": "integer_divide(10, 5)", "expected": "2"},
            {"input": "integer_divide(1, 3)", "expected": "0"},
        ],
    },
    {
        "problem": (
            "Write a Python function called `remainder(a, b)` that returns the remainder "
            "when a is divided by b.\n\n"
            "Example:\n  remainder(7, 3) -> 1"
        ),
        "test_cases": [
            {"input": "remainder(7, 3)", "expected": "1"},
            {"input": "remainder(10, 5)", "expected": "0"},
            {"input": "remainder(10, 3)", "expected": "1"},
        ],
    },
]


SYSTEM_PROMPT = (
    "You are a Python coding assistant. Write clean, correct Python code. "
    "Output ONLY the function definition inside ```python\\n...\\n``` code blocks. "
    "Do not include any other text, explanation, or examples."
)


def build_dataset(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for task in TASKS:
            record = {
                "problem": task["problem"],
                "test_cases_json": json.dumps(task["test_cases"]),
            }
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(TASKS)} tasks to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "nemorl_code_tasks.jsonl")
    build_dataset(output_path)
