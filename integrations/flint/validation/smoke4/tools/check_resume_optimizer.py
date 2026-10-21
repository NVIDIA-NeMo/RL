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
"""Read only small Adam step tensors to verify checkpoint optimizer continuity."""

import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


def optimizer_step(path):
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata()
    keys = sorted(
        key
        for key in metadata.state_dict_metadata
        if "lm_head" in key and key.endswith(".step")
    )
    if not keys:
        raise ValueError("No saved LM-head Adam step tensor")
    key = keys[0]
    spec = metadata.state_dict_metadata[key]
    state = {key: torch.empty(spec.size, dtype=spec.properties.dtype)}
    dcp.load(
        state,
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    return {"key": key, "value": state[key].item(), "checkpoint": str(path)}


root = Path("/flint-work/results/synthetic")
records = [
    optimizer_step(root / f"step_{step}/policy/optimizer/optim") for step in (2, 3)
]
assert [item["value"] for item in records] == [2, 3], records
report = {
    "status": "passed",
    "scope": "LM-head Adam step restored and incremented",
    "records": records,
}
path = Path("/flint-work/evidence/optimizer-resume-check.json")
path.write_text(json.dumps(report, indent=2) + "\n")
path.chmod(0o600)
print(json.dumps(report))
