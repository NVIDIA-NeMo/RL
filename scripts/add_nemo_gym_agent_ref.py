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

"""Stamp an ``agent_ref`` onto NeMo-Gym JSONL rows that are missing one.

NeMo-RL routes every row to a Gym agent by reading ``agent_ref.name``
(``nemo_rl/environments/nemo_gym.py``), so a row without it fails the rollout
batch with ``KeyError: 'agent_ref'``. Some Gym generators omit the field even
though the ``example.jsonl`` they ship beside declares it — for instance
``resources_servers/circle_count/generate_data.py``.

Rewrites files in place and leaves rows that already declare an ``agent_ref``
untouched, so it is safe to re-run.

Usage:
    python scripts/add_nemo_gym_agent_ref.py \\
        --agent-name circle_count_simple_agent \\
        path/to/train.jsonl path/to/val.jsonl
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+", help="JSONL files to rewrite in place."
    )
    parser.add_argument(
        "--agent-name",
        required=True,
        help=(
            "Agent key from the Gym resources server config, e.g. "
            "circle_count_simple_agent."
        ),
    )
    parser.add_argument(
        "--agent-type",
        default="responses_api_agents",
        help="Gym agent type. Only responses_api_agents exists today.",
    )
    args = parser.parse_args()

    agent_ref = {"type": args.agent_type, "name": args.agent_name}

    for path in args.paths:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        stamped = 0
        for row in rows:
            if "agent_ref" not in row:
                row["agent_ref"] = agent_ref
                stamped += 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"{path}: stamped {stamped} of {len(rows)} row(s)")


if __name__ == "__main__":
    main()
