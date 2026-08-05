#!/usr/bin/env python
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
"""Print a side-by-side throughput comparison of NeMo-RL runs from W&B.

Runs are addressed by **run id**, not display name: the two SingleController
arms of the SWE A/B share one EXP_NAME, so a display-name lookup cannot tell
them apart. Defaults are the three runs in SWE_TEACHER_SC_TQ_RUNBOOK.md.

Usage (from a NETWORKED shell — the sandbox blocks wandb.ai):
    source /lustre/.../users/zhiyul/secrets.sh
    python wandb_compare.py                       # the three reference runs
    python wandb_compare.py <run-id> [<run-id> …]
"""

import statistics as st
import sys

import wandb

ENTITY = "nvidia"
PROJECT = "nemorl-dataplane-zhiyul"

# Default to the reference A/B; override with run ids on the command line.
DEFAULT_RUNS = ["rb88wuo0", "0pta4j34", "0e4e72g8"]

# Throughput rates moved from timing/train/ to performance/; runs recorded
# before that rename keep the old prefix, so try both and report which hit.
THRPT_KEYS = [
    "performance/tokens_per_sec_per_gpu",
    "timing/train/total_tokens_per_sec_per_gpu",
    "performance/valid_tokens_per_sec_per_gpu",
    "timing/train/valid_tokens_per_sec_per_gpu",
]
STEPT = "timing/train/total_step_time"


def main(run_ids: list[str]) -> None:
    api = wandb.Api(timeout=30)
    print(f"entity={ENTITY} project={PROJECT}\n")

    for run_id in run_ids:
        try:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        except Exception as e:  # noqa: BLE001 - surface and keep going
            print(f"{run_id}: not fetched ({type(e).__name__}: {e})\n")
            continue

        # One history pass for every key of interest: each scan_history() call
        # costs its own GraphQL + init RPC, and rows carry all requested columns.
        keys = [*THRPT_KEYS, STEPT]
        rows = list(run.scan_history(keys=keys))
        series = {k: [r[k] for r in rows if r.get(k) is not None] for k in keys}

        thrpt_key = next((k for k in THRPT_KEYS if series[k]), None)
        print(f"=== {run.name}  [{run_id}]  state={run.state} ===")

        if thrpt_key is None:
            print("  no throughput metric found\n")
            continue

        vals = series[thrpt_key]
        print(f"  {thrpt_key}  ({len(vals)} steps)")
        print(f"    per step: {[round(x, 2) for x in vals]}")
        print(
            f"    mean={st.mean(vals):.2f}  median={st.median(vals):.2f}  max={max(vals):.2f}"
        )
        if series[STEPT]:
            steps = series[STEPT]
            print(
                f"  total_step_time (s): {[round(x, 1) for x in steps]}  mean={st.mean(steps):.1f}"
            )
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_RUNS)
