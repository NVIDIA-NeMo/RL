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

"""Build and verify the byte-identical 1,920-row Super TROPD proxy slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

EXPECTED_SHA256 = "1fa08931c7321a149172bd5fb87ba024b68d1e880859792875b5d53a84c0548d"
EXPECTED_ROWS = 1_920
EXCLUDED_AGENT = "math_with_judge_simple_agent"
EXPECTED_AGENT_COUNTS = Counter(
    {
        "single_step_tool_use_with_argument_comparison_agent": 1_019,
        "code_gen_simple_agent": 220,
        "swe_pivot_single_step_tool_use_with_argument_comparison_agent": 180,
        "instruction_following_simple_agent": 122,
        "reasoning_gym_simple_agent": 105,
        "search_pivot_single_step_tool_use_with_argument_comparison_agent": 86,
        "structured_outputs_v3_simple_agent": 48,
        "terminus_judge_string_only_simple_agent": 39,
        "ns_tools_simple_agent": 25,
        "calendar_simple_agent": 25,
        "citation_format_simple_agent": 23,
        "freeform_formatting_simple_agent": 18,
        "toolcall_schema_single_step_tool_use_with_argument_comparison_agent": 10,
    }
)


def _agent_name(raw_line: bytes, line_number: int) -> str:
    """Extract one agent name while retaining the source bytes for output."""
    try:
        row = json.loads(raw_line)
        agent_ref = row["agent_ref"]
        name = agent_ref["name"] if isinstance(agent_ref, dict) else agent_ref
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid agent_ref on input line {line_number}") from error
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid agent name on input line {line_number}: {name!r}")
    return name


def build_proxy_dataset(source: Path, output: Path) -> None:
    """Stream the filtered prefix, validate its contract, then publish atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    accepted = 0

    temporary_path: Path | None = None
    try:
        with (
            source.open("rb") as source_file,
            tempfile.NamedTemporaryFile(
                mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
            ) as temporary_file,
        ):
            temporary_path = Path(temporary_file.name)
            for line_number, raw_line in enumerate(source_file, start=1):
                name = _agent_name(raw_line, line_number)
                if name == EXCLUDED_AGENT:
                    continue
                temporary_file.write(raw_line)
                digest.update(raw_line)
                counts[name] += 1
                accepted += 1
                if accepted == EXPECTED_ROWS:
                    break

        actual_sha256 = digest.hexdigest()
        if accepted != EXPECTED_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_ROWS} accepted rows, found only {accepted}"
            )
        if counts != EXPECTED_AGENT_COUNTS:
            raise ValueError(
                "Proxy agent distribution mismatch: "
                f"expected {dict(EXPECTED_AGENT_COUNTS)}, got {dict(counts)}"
            )
        if actual_sha256 != EXPECTED_SHA256:
            raise ValueError(
                f"Proxy SHA-256 mismatch: expected {EXPECTED_SHA256}, "
                f"got {actual_sha256}"
            )

        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Wrote {accepted} rows to {output}")
    print(f"SHA-256: {EXPECTED_SHA256}")
    for name, count in EXPECTED_AGENT_COUNTS.most_common():
        print(f"{count:4d} {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_proxy_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
