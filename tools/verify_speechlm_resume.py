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

"""Read native checkpoint scalar state and verify the CER400 20+2 test resume.

Reads only optimizer steps, group LRs and scheduler bytes from DCP, never
optimizer moments or model weights. The cursor probe restores the saved
TorchData state onto integer row IDs with the recipe's ordered loader shape.
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torchdata.stateful_dataloader import StatefulDataLoader


def checkpoint_scalar_state(checkpoint: Path) -> dict:
    reader = dcp.FileSystemReader(checkpoint / "policy/optimizer/optim")
    metadata = reader.read_metadata()
    scalars = {}
    for key, value in metadata.state_dict_metadata.items():
        if key.startswith("sched.") or (
            key.startswith("optim.") and key.rsplit(".", 1)[-1] in {"step", "lr"}
        ):
            scalars[key] = (
                torch.empty(value.size, dtype=value.properties.dtype)
                if isinstance(value, TensorStorageMetadata)
                else None
            )
    if not scalars:
        raise ValueError("Checkpoint contains no optimizer/scheduler scalar state")
    dcp.load(
        scalars,
        storage_reader=reader,
        planner=DefaultLoadPlanner(
            flatten_state_dict=False, flatten_sharded_tensors=False
        ),
    )
    return {
        key: value.tolist() if isinstance(value, torch.Tensor) else value
        for key, value in scalars.items()
    }


def next_source_rows(
    checkpoint: Path, *, rows: int, batch_size: int, num_workers: int
) -> list[int]:
    state = torch.load(
        checkpoint / "train_dataloader.pt", map_location="cpu", weights_only=False
    )
    loader = StatefulDataLoader(
        range(rows),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    loader.load_state_dict(state)
    # A fresh iterator restores the actual sampler/worker snapshot; no model/audio
    # data or custom cursor correction is involved.
    return next(iter(loader)).tolist()


def checkpoint_receipt(output: Path, step: int) -> dict:
    checkpoint = output / "checkpoints" / f"step_{step}"
    assert (checkpoint / "policy/weights/model/.metadata").is_file()
    info = json.loads((checkpoint / "training_info.json").read_text())
    assert info["total_steps"] == step and info["consumed_samples"] == step * 20, info
    scalar_state = checkpoint_scalar_state(checkpoint)
    optimizer_steps = {
        float(value)
        for key, value in scalar_state.items()
        if key.startswith("optim.") and key.endswith(".step")
    }
    learning_rates = {
        float(value)
        for key, value in scalar_state.items()
        if key.startswith("optim.") and key.endswith(".lr")
    }
    assert optimizer_steps == {float(step)}, optimizer_steps
    assert learning_rates == {1e-6}, learning_rates
    assert scalar_state["sched.last_epoch"] == step, scalar_state
    next_rows = next_source_rows(checkpoint, rows=20000, batch_size=40, num_workers=1)
    assert next_rows == list(range(next_rows[0], next_rows[0] + 40)), next_rows
    return {
        "training_info": info,
        "optimizer_scheduler_scalars": scalar_state,
        "next_source_rows": next_rows,
    }


def verify(output: Path, *, initial_only: bool) -> dict:
    before = checkpoint_receipt(output, 20)
    if initial_only:
        return {"initial": before}
    after = checkpoint_receipt(output, 22)
    steps = [
        int(step)
        for step in re.findall(
            r"={25} Step (\d+)/22 ={25}", (output / "resume2.log").read_text()
        )
    ]
    assert steps == [21, 22], steps
    assert (
        after["training_info"]["total_valid_tokens"]
        > before["training_info"]["total_valid_tokens"]
    )
    assert after["next_source_rows"][0] >= before["next_source_rows"][0] + 80
    # Selected groups may skip constant-reward recordings; they must come from
    # the restored cursor onward rather than restarting the original stream.
    logs = list((output / "logs").glob("exp_*/train_data_step21.jsonl"))
    assert len(logs) == 1, logs
    log = logs[0]
    source_rows = [
        int(json.loads(line)["source_idx"][0]) for line in log.read_text().splitlines()
    ]
    assert source_rows and min(source_rows) >= before["next_source_rows"][0], (
        source_rows
    )
    return {
        "initial": before,
        "resumed": after,
        "resumed_update_steps": steps,
        "first_resumed_source_rows": sorted(set(source_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-only", action="store_true")
    args = parser.parse_args()
    result = verify(args.output_dir, initial_only=args.initial_only)
    filename = (
        "training_initial_receipt.json"
        if args.initial_only
        else "training_resume_receipt.json"
    )
    (args.output_dir / filename).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
