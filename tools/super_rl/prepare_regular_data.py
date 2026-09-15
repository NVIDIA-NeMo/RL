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

"""Cap an ordered regular curriculum and attach historical effort profiles.

Never copy effort instructions/labels from the profile variant. Preserve source
prompts, routes, historical measurements and their original profiling cap.
Publication is atomic and refuses to overwrite an existing dataset.
"""

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import tempfile


def prepare(
    source: Path,
    profiles: Path,
    output: Path,
    source_sha: str,
    profile_sha: str,
    cap: int,
    expected_rows: int,
) -> dict:
    if cap <= 0 or expected_rows <= 0:
        raise ValueError("Cap and row count must be positive")
    digests = [hashlib.sha256(), hashlib.sha256()]
    result_hash = hashlib.sha256()
    seen = set()
    routes = Counter()
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with (
            os.fdopen(fd, "wb") as target,
            source.open("rb") as rows,
            profiles.open("rb") as profiled,
        ):
            for ordinal, pair in enumerate(zip_longest(rows, profiled)):
                if None in pair:
                    raise ValueError("Curriculum variants have different row counts")
                for digest, line in zip(digests, pair):
                    digest.update(line)
                row, profile_row = map(json.loads, pair)
                identity = row["metadata"]["curriculum"]
                if (
                    identity != profile_row["metadata"]["curriculum"]
                    or identity["ordinal"] != ordinal
                ):
                    raise ValueError("Curriculum identity/order mismatch")
                key = identity["task_sha256"]
                if key in seen or "reasoning_effort" in row:
                    raise ValueError("Duplicate task or non-regular source")
                seen.add(key)
                profile = profile_row["reasoning_effort_profile"]
                if (
                    "reasoning_effort_profile" in row
                    and row["reasoning_effort_profile"] != profile
                ):
                    raise ValueError("Conflicting historical effort profiles")
                row["reasoning_effort_profile"] = profile
                row["responses_create_params"]["max_output_tokens"] = cap
                routes[row["agent_ref"]["name"]] += 1
                payload = (
                    json.dumps(
                        row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    )
                    + "\n"
                ).encode()
                target.write(payload)
                result_hash.update(payload)
            if len(seen) != expected_rows or [d.hexdigest() for d in digests] != [
                source_sha,
                profile_sha,
            ]:
                raise ValueError("Source checksum or unique row count mismatch")
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {
        "complete": True,
        "path": str(output),
        "sha256": result_hash.hexdigest(),
        "rows": len(seen),
        "routes": dict(routes),
        "source_sha256": source_sha,
        "profile_sha256": profile_sha,
        "max_output_tokens": cap,
        "effort_shaping": False,
        "historical_profiles_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.profiles,
                args.output,
                args.source_sha256,
                args.profile_sha256,
                args.cap,
                args.rows,
            )
        )
    )


if __name__ == "__main__":
    main()
