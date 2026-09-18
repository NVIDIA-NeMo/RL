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
"""Read-only checkpoint gate for a serialized continuation of this run."""

import json
import os
import sys
from pathlib import Path

import torch
from nemo_rl.utils.checkpoint import _load_megatron_common_state_dict
from nemo_rl.experience.interfaces import FRONTIER_ORDINAL_KEY, RESUME_BASE_ORDINAL_KEY

root, receipt = map(Path, sys.argv[1:])
steps = [p for p in root.glob("step_*") if p.is_dir() and p.name[5:].isdigit()]
assert steps, "No finalized checkpoint: refusing a fresh start"
checkpoint = max(steps, key=lambda p: int(p.name[5:]))
step = int(checkpoint.name[5:])
info = json.loads((checkpoint / "training_info.json").read_text())
status = json.loads((root / "latest_checkpoint_status.json").read_text())
assert info["current_step"] == step == status["last_checkpoint_step"]
assert 0 < step <= 100
for relative in (
    "config.yaml",
    "train_dataloader.pt",
    "rollouts.pt",
    "replay_buffer.pt",
):
    assert (checkpoint / relative).stat().st_size > 0, relative
iteration = checkpoint / "policy/weights/iter_0000000"
for name in (".metadata", "metadata.json", "run_config.yaml", "train_state.pt"):
    assert (iteration / name).stat().st_size > 0, name
shards = list(iteration.glob("*.distcp"))
assert len(shards) == 64 and all(p.stat().st_size > 0 for p in shards)
# Only the small common state is loaded, not the 1.5 TB tensor payload.
common = _load_megatron_common_state_dict(iteration)
assert "optimizer" in common, "Missing embedded optimizer"
# These are trusted artifacts from this exact training lineage.
loader = torch.load(
    checkpoint / "train_dataloader.pt", map_location="cpu", weights_only=False
)
rollouts = torch.load(
    checkpoint / "rollouts.pt", map_location="cpu", weights_only=False
)
assert isinstance(loader, dict) and loader
assert isinstance(rollouts, dict) and rollouts
assert rollouts.get(FRONTIER_ORDINAL_KEY) is not None, "Missing trained frontier"
assert rollouts.get(RESUME_BASE_ORDINAL_KEY) is not None, "Missing data rewind anchor"
result = {
    "complete": True,
    "checkpoint": str(checkpoint),
    "current_step": step,
    "optimizer_present": True,
    "dataloader_present": True,
    "shards": len(shards),
    "load_replay_buffer": False,
    "wandb_run_id": os.environ["WANDB_RUN_ID"],
    "wandb_resume": "must",
    "restore_tested": False,
}
receipt.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
