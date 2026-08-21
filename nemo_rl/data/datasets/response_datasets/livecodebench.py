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

"""LiveCodeBench dataset for SDPO.

LCBv6 is the Feb 2025 - May 2025 slice of LiveCodeBench (131 problems). The
paper splits each problem's private test cases 50/50 into a "public" half
(used as the in-training reward signal) and a "private" half (held out for
validation), so train and validation expose the same problems with disjoint
test halves.
"""

from __future__ import annotations

import base64
import json
import pickle
import random
import zlib
from datetime import date
from typing import Any

from datasets import Dataset
from huggingface_hub import hf_hub_download

from nemo_rl.data.datasets.raw_dataset import RawDataset


_LCBV6_START = date(2025, 2, 1)
_LCBV6_END = date(2025, 5, 31)

# `livecodebench/code_generation_lite` ships one cumulative JSONL per release.
# Datasets >=4.0 dropped support for the repo's loader script, so we pull the
# raw file via huggingface_hub and parse JSON ourselves.
_VERSION_TAG_TO_FILE = {
    "release_v1": "test.jsonl",
    "release_v2": "test2.jsonl",
    "release_v3": "test3.jsonl",
    "release_v4": "test4.jsonl",
    "release_v5": "test5.jsonl",
    "release_v6": "test6.jsonl",
}


def _parse_test_cases(raw: Any) -> list[dict[str, str]]:
    """Decode LCB test cases.

    Public tests are a JSON string. Private tests are base64 + zlib + pickle
    of a JSON string (the LCB-Lite encoding — pickle yields a str that itself
    needs json.loads). Lists pass through. Anything else falls back to an
    empty list so a single bad row doesn't kill the loader.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [dict(t) for t in raw]
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, list):
                return [dict(t) for t in decoded]
        except json.JSONDecodeError:
            pass
        try:
            inner = pickle.loads(zlib.decompress(base64.b64decode(raw.encode())))
            if isinstance(inner, str):
                inner = json.loads(inner)
            return [dict(t) for t in inner]
        except (ValueError, zlib.error, pickle.UnpicklingError, json.JSONDecodeError):
            return []
    return []


def _within_lcbv6(contest_date: Any) -> bool:
    if contest_date is None:
        return False
    if isinstance(contest_date, str):
        try:
            d = date.fromisoformat(contest_date.split("T")[0])
        except ValueError:
            return False
    elif hasattr(contest_date, "date"):
        d = contest_date.date()
    elif isinstance(contest_date, date):
        d = contest_date
    else:
        return False
    return _LCBV6_START <= d <= _LCBV6_END


def _split_tests(
    tests: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split a problem's tests into a 50/50 train/validation pair.

    With a fixed seed per problem so the same problem yields the same split
    in train and validation loaders.
    """
    if len(tests) <= 1:
        return tests, tests
    rng = random.Random(seed)
    indices = list(range(len(tests)))
    rng.shuffle(indices)
    half = max(1, len(tests) // 2)
    train_idx = sorted(indices[:half])
    val_idx = sorted(indices[half:]) or train_idx
    return [tests[i] for i in train_idx], [tests[i] for i in val_idx]


def _build_user_prompt(question: str, starter_code: str | None) -> str:
    parts = [question.strip()]
    if starter_code and starter_code.strip():
        parts.append(
            "Use the following starter code:\n\n"
            f"```python\n{starter_code.strip()}\n```"
        )
    parts.append(
        "Write your final solution as a single self-contained Python "
        "code block."
    )
    return "\n\n".join(parts)


class LiveCodeBenchDataset(RawDataset):
    """LiveCodeBench v6 (Feb-May 2025, 131 problems).

    Args:
        split: "train" exposes the train half of each problem's tests as the
            reward signal; "validation" exposes the held-out half.
        version_tag: HuggingFace config name. Defaults to "release_v6". Set
            to e.g. "release_latest" to get a newer LCB cut and re-filter on
            contest_date.
        test_split_seed: Seed used when splitting each problem's tests 50/50.
    """

    def __init__(
        self,
        split: str = "train",
        version_tag: str = "release_v6",
        test_split_seed: int = 42,
        **kwargs,
    ) -> None:
        self.task_name = "livecodebench_v6"
        self.split = split

        filename = _VERSION_TAG_TO_FILE.get(version_tag, version_tag)
        local_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=filename,
            repo_type="dataset",
        )
        with open(local_path, "r") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        records = []
        for row in raw:
            if not _within_lcbv6(row.get("contest_date")):
                continue

            tests = _parse_test_cases(row.get("public_test_cases")) + _parse_test_cases(
                row.get("private_test_cases")
            )
            if not tests:
                continue

            qid = str(row.get("question_id") or row.get("question_title") or "")
            train_tests, val_tests = _split_tests(
                tests, seed=hash((qid, test_split_seed)) & 0xFFFFFFFF
            )
            tests_for_split = train_tests if split == "train" else val_tests

            metadata = {
                "tests": tests_for_split,
                "starter_code": row.get("starter_code") or "",
                "function_name": _infer_function_name(row.get("starter_code") or ""),
                "platform": row.get("platform") or "",
                "question_id": qid,
            }

            user_prompt = _build_user_prompt(
                row.get("question_content") or "", row.get("starter_code") or ""
            )
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": ""},
                    ],
                    "metadata": json.dumps(metadata),
                    "task_name": self.task_name,
                }
            )

        if not records:
            raise ValueError(
                f"No LCBv6 problems found in livecodebench/code_generation_lite "
                f"version_tag={version_tag} within {_LCBV6_START}..{_LCBV6_END}."
            )

        self.dataset = Dataset.from_list(records)


def _infer_function_name(starter_code: str) -> str | None:
    r"""Pull the entrypoint method name out of an LCB starter snippet.

    LCB LeetCode-style starters look like
    ``class Solution:\n    def methodName(self, ...)``; we want ``methodName``.
    Returns None for stdin-style problems (empty/non-class starter).
    """
    import re

    if not starter_code:
        return None
    m = re.search(r"def\s+(\w+)\s*\(", starter_code)
    return m.group(1) if m else None
