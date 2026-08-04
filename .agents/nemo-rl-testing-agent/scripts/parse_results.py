#!/usr/bin/env python3
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

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Turn a suite log emitted by run_suite_remote.sh into per-test results JSON.

The JSON is the hand-off point between the mechanical part of the agent and the
judgement part: this script fills in name/status/rc/timing/error signature, and
the agent enriches each entry's ``comment`` before it is rendered by
``post_report.py``.

Usage:
    uv run --script parse_results.py <suite.log> --out results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TEST_BEGIN_RE = re.compile(
    r"===NRLTA_TEST_BEGIN name=(?P<name>\S+) suite=(?P<suite>\S+)"
)
TEST_END_RE = re.compile(
    r"===NRLTA_TEST_END name=(?P<name>\S+) rc=(?P<rc>-?\d+) secs=(?P<secs>\d+)"
)
TEST_SKIP_RE = re.compile(
    r"===NRLTA_TEST_SKIP name=(?P<name>\S+) reason=(?P<reason>\S+)"
)
PREP_FIELD_RE = re.compile(
    r"^(mcore_fetch_ref|mcore_sha|mcore_sha_before|mcore_subject|megatron_core_file"
    r"|bridge_fetch_ref|bridge_sha"
    r"|nemo_rl_fetch_ref|nemo_rl_sha|nemo_rl_env_sha|nemo_rl_mode)=(.*)$"
)

# Ordered by how specific the signal is; the first pattern that matches when
# scanning a failed test's output backwards becomes the reported signature.
ERROR_PATTERNS = [
    re.compile(r"^\s*(?:FAIL|FAILED)[:\s].*$"),
    re.compile(r"^.*AssertionError.*$"),
    re.compile(r"^.*\b(?:CUDA error|illegal memory access)\b.*$"),
    re.compile(r"^.*\b(?:OutOfMemoryError|CUDA out of memory)\b.*$"),
    re.compile(r"^\s*\S*(?:Error|Exception):.*$"),
    re.compile(r"^.*\b(?:Segmentation fault|core dumped|Killed)\b.*$"),
    re.compile(r"^.*srun: error.*$"),
]

JOB_ERROR_PATTERNS = [
    re.compile(r"^.*NRLTA_(?:PREP_)?FAIL:.*$"),
    re.compile(r"^.*srun: error:.*$"),
    re.compile(r"^.*slurmstepd:.*(?:CANCELLED|TIME LIMIT|OUT OF MEMORY).*$"),
    re.compile(r"^.*QOSMinGRES.*$"),
]

# Teardown artifacts that appear AFTER the real exception. Because signatures are
# found by scanning backwards, these otherwise win and point the reader at
# interpreter shutdown instead of the failure -- e.g. every one of three real
# `AttributeError: 'NoneType' object has no attribute 'tolist'` failures was
# reported as `PythonFinalizationError: preexec_fn not supported`.
NOISE_PATTERNS = [
    re.compile(r"PythonFinalizationError"),
    re.compile(r"preexec_fn is not supported|preexec_fn not supported"),
    re.compile(r"^\s*Exception ignored in:"),
    re.compile(r"atexit\._run_exitfuncs"),
    re.compile(r"Ray objects? .*lost|raylet.*died", re.IGNORECASE),
]

# Errors NeMo-RL catches per sample and carries on from, so the test still exits 0.
# `rollouts.py` wraps generation in `except Exception: print(...); break`, which
# means a backend broken for EVERY sample still yields a rollout, a metric, and a
# green test. Two async L1 tests passed this way while every generation raised
# `AttributeError: 'NoneType' object has no attribute 'tolist'`. A suite that
# cannot distinguish "worked" from "silently degraded" is worse than no suite.
SWALLOWED_ERROR_PATTERNS = [
    re.compile(r"^Error generating response for sample \d+:(?P<detail>.*)$"),
    re.compile(r"^Error processing sample \d+:(?P<detail>.*)$"),
]

# A failed metric check raises nothing: `check_metrics.py` prints a rich table and
# the suite just exits non-zero. That made an entire class of failure -- the
# numerics regressions these suites exist to catch -- come back with an empty
# signature, which in turn made them invisible to the known-issues registry.
METRIC_ROW_RE = re.compile(
    r"^\s*│\s*FAIL\s*│\s*(?P<check>.+?)\s*│\s*(?P<value>[^│]*?)\s*│",
)

MAX_TAIL_LINES = 25
MAX_LINE_CHARS = 400


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line.rstrip("\n"))


def is_noise(line: str) -> bool:
    return any(pattern.search(line) for pattern in NOISE_PATTERNS)


def find_metric_failure(block: list[str]) -> str:
    """Returns a signature for a failed metric assertion, if the block has one.

    The measured value is deliberately left out of the signature: it varies run to
    run for the same underlying bug, and including it would make every occurrence
    look like a different failure to anything matching on signatures.
    """
    for line in block:
        match = METRIC_ROW_RE.match(line)
        if match:
            return f"metric check failed: {match.group('check').strip()}"[
                :MAX_LINE_CHARS
            ]
    return ""


def find_metric_failure_detail(block: list[str]) -> str:
    """The same failure with its measured value, for humans reading the report."""
    for line in block:
        match = METRIC_ROW_RE.match(line)
        if match:
            value = match.group("value").strip()
            check = match.group("check").strip()
            return f"{check} (measured {value})" if value else check
    return ""


def find_error_signature(block: list[str]) -> str:
    """Returns the most specific-looking error line in a failed test's output."""
    metric = find_metric_failure(block)
    if metric:
        return metric
    candidates = [line for line in block if not is_noise(line)]
    for pattern in ERROR_PATTERNS:
        for line in reversed(candidates):
            if pattern.match(line):
                return line.strip()[:MAX_LINE_CHARS]
    return ""


def find_swallowed_errors(block: list[str]) -> tuple[int, str]:
    """Counts caught-and-continued sample errors, and returns the first signature.

    The signature comes from the traceback following the first occurrence, since
    the caught line itself often only carries the exception's first line.
    """
    count = 0
    first_index = -1
    for index, line in enumerate(block):
        for pattern in SWALLOWED_ERROR_PATTERNS:
            if pattern.match(line):
                count += 1
                if first_index < 0:
                    first_index = index
                break

    if first_index < 0:
        return 0, ""

    # Scan forward from the first hit: the exception type lands a few lines into
    # the traceback that Ray prints inline after the message.
    window = block[first_index : first_index + 40]
    signature = ""
    for pattern in ERROR_PATTERNS:
        for line in window:
            if pattern.match(line) and not is_noise(line):
                signature = line.strip()[:MAX_LINE_CHARS]
                break
        if signature:
            break
    return count, signature or block[first_index].strip()[:MAX_LINE_CHARS]


def tail_of(block: list[str]) -> list[str]:
    non_empty = [line.strip()[:MAX_LINE_CHARS] for line in block if line.strip()]
    return non_empty[-MAX_TAIL_LINES:]


def parse_log(log_path: Path, artifact_dir: str) -> dict[str, Any]:
    lines = [
        strip_ansi(line) for line in log_path.read_text(errors="replace").splitlines()
    ]

    prep: dict[str, str] = {}
    tests: list[dict[str, Any]] = []
    job_errors: list[str] = []

    current: dict[str, Any] | None = None
    block: list[str] = []

    for line in lines:
        prep_match = PREP_FIELD_RE.match(line)
        if prep_match and prep_match.group(1) not in prep:
            prep[prep_match.group(1)] = prep_match.group(2).strip()

        for pattern in JOB_ERROR_PATTERNS:
            if pattern.match(line):
                stripped = line.strip()[:MAX_LINE_CHARS]
                if stripped not in job_errors:
                    job_errors.append(stripped)
                break

        skip_match = TEST_SKIP_RE.search(line)
        if skip_match:
            tests.append(
                {
                    "name": skip_match.group("name"),
                    "suite": "",
                    "status": "not run",
                    "rc": None,
                    "secs": None,
                    "error_signature": "",
                    "tail": [],
                    "log": "",
                    "comment": "",
                }
            )
            continue

        begin_match = TEST_BEGIN_RE.search(line)
        if begin_match:
            current = {
                "name": begin_match.group("name"),
                "suite": begin_match.group("suite"),
                "status": "incomplete",
                "rc": None,
                "secs": None,
                "error_signature": "",
                "tail": [],
                "log": f"{artifact_dir.rstrip('/')}/{begin_match.group('name')}.log"
                if artifact_dir
                else "",
                "comment": "",
            }
            block = []
            continue

        end_match = TEST_END_RE.search(line)
        if end_match and current is not None:
            rc = int(end_match.group("rc"))
            current["rc"] = rc
            current["secs"] = int(end_match.group("secs"))
            current["status"] = "pass" if rc == 0 else "fail"
            if rc != 0:
                current["error_signature"] = find_error_signature(block)
                detail = find_metric_failure_detail(block)
                if detail:
                    current["metric_failure"] = detail
                current["tail"] = tail_of(block)
            else:
                swallowed, signature = find_swallowed_errors(block)
                if swallowed:
                    current["status"] = "pass (suspect)"
                    current["swallowed_errors"] = swallowed
                    current["error_signature"] = signature
                    current["tail"] = tail_of(block)
            tests.append(current)
            current = None
            block = []
            continue

        if current is not None:
            block.append(line)

    if current is not None:
        # A test that started but never printed its END marker: the job was
        # killed, hit the Slurm time limit, or the node died.
        current["status"] = "incomplete"
        current["error_signature"] = (
            find_error_signature(block)
            or "no completion marker (job killed or timed out)"
        )
        current["tail"] = tail_of(block)
        tests.append(current)

    return {
        "log": str(log_path),
        "artifact_dir": artifact_dir,
        "prep": prep,
        "job_errors": job_errors,
        "tests": tests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", type=Path, help="Suite log containing the NRLTA markers."
    )
    parser.add_argument(
        "--artifact-dir", default="", help="Cluster dir holding the per-test logs."
    )
    parser.add_argument("--out", type=Path, help="Write JSON here instead of stdout.")
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"parse_results: no such log file: {args.log}", file=sys.stderr)
        return 1

    results = parse_log(args.log, args.artifact_dir)
    payload = json.dumps(results, indent=2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
        print(
            f"parse_results: wrote {len(results['tests'])} test result(s) to {args.out}",
            file=sys.stderr,
        )
    else:
        print(payload)

    if not results["tests"]:
        print(
            "parse_results: WARNING - no test markers found; the job likely died during prep",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
