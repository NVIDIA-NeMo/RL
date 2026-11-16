# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
"""Reject incomplete or non-bitwise Qwen precision gates before releasing long runs.

Usage: verify_startup.py RUN_DIR MODE [MIN_STEPS] [EXPECTED_MAX_AGE].
VPG and baseline smoke gates check at least five steps and can verify the requested lag.
"""

import math
import re
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(sys.argv[1])
mode = sys.argv[2]
assert mode in ("sync", "vpg", "baseline"), mode
required_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
expected_max_age = int(sys.argv[4]) if len(sys.argv) > 4 else None
assert required_steps >= 5, "training gates require at least five steps"
if expected_max_age is not None:
    assert expected_max_age >= 1
    assert required_steps >= expected_max_age + 2, (
        "check a post-refit batch at the requested lag"
    )
latest = max((root / "logs").glob("exp_*"), key=lambda p: p.stat().st_mtime)
steps = (1, 2) if mode == "sync" else ((1,) if mode == "vpg" else ())
checker = Path(__file__).with_name("logprob_match.py")
for step in steps:
    data = latest / f"train_data_step{step}.jsonl"
    assert data.is_file(), f"missing {data}"
    result = subprocess.check_output(
        [sys.executable, str(checker), str(data)], text=True
    )
    print(result, flush=True)
    assert "BITWISE MATCH" in result and re.search(r"tokens [1-9][0-9]* ", result), (
        result
    )
if mode in ("vpg", "baseline"):
    metrics = {}
    for event in sorted(latest.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            metrics.setdefault(tag, {}).update(
                {e.step: e.value for e in accumulator.Scalars(tag)}
            )
    for tag in ("train/loss", "train/grad_norm"):
        values = metrics[tag]
        assert set(range(1, required_steps + 1)).issubset(values), (tag, values)
        assert all(math.isfinite(values[s]) for s in range(1, required_steps + 1)), (
            tag,
            values,
        )
    if mode == "vpg":
        for suffix, expected in (
            ("version_point_applied", 1),
            ("version_point_parity", 0),
            ("version_point_no_snapshot", 0),
        ):
            tags = [tag for tag in metrics if tag.split("/")[-1] == suffix]
            assert len(tags) == 1, f"missing or ambiguous {suffix}: {tags}"
            values = metrics[tags[0]]
            assert set(range(1, required_steps + 1)).issubset(values), (suffix, values)
            assert all(values[s] == expected for s in range(1, required_steps + 1)), (
                suffix,
                values,
            )
    else:
        applied = metrics.get("train/version_point_applied", {})
        assert all(value == 0 for value in applied.values()), applied
    if expected_max_age is not None:
        tags = [tag for tag in metrics if tag.split("/")[-1] == "version_point_age"]
        assert len(tags) == 1, f"missing or ambiguous trajectory age: {tags}"
        ages = metrics[tags[0]]
        assert set(range(1, required_steps + 1)).issubset(ages), ages
        observed = [ages[s] for s in range(1, required_steps + 1)]
        assert all(0 <= age <= expected_max_age for age in observed), observed
        assert max(observed) == expected_max_age, observed
print("GATE VERIFIED", flush=True)
