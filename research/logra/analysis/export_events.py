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

"""Export native GRPO scalar events without smoothing or step renumbering."""

import argparse
import csv
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = {
    "memory_gib": "train/memory/mean_peak_allocated_gib",
    "accuracy": "validation/accuracy",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    audit = {}
    for method, label in [("dense", "Dense AdamW"), ("logra", "LoGRA")]:
        for seed in (42, 43):
            events = {}
            files = sorted(
                (args.results / f"{method}-s{seed}").rglob("events.out.tfevents.*")
            )
            for file in files:
                accumulator = EventAccumulator(
                    str(file), size_guidance={"scalars": 0}
                ).Reload()
                for tag in accumulator.Tags()["scalars"]:
                    for event in accumulator.Scalars(tag):
                        key = (tag, event.step)
                        if key not in events or event.wall_time > events[key].wall_time:
                            events[key] = event
            audit[f"{method}-s{seed}"] = {
                "files": [str(p) for p in files],
                "tags": sorted({tag for tag, step in events}),
                "last_values": {
                    tag: max(
                        (e for (t, s), e in events.items() if t == tag),
                        key=lambda e: e.step,
                    ).value
                    for tag in sorted({t for t, s in events})
                },
            }
            steps = sorted({s for (tag, s) in events if tag in TAGS.values()})
            for step in steps:
                row = dict(method=label, seed=seed, step=step)
                for metric, tag in TAGS.items():
                    row[metric] = (
                        events[tag, step].value if (tag, step) in events else ""
                    )
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "seed", "step", *TAGS])
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
