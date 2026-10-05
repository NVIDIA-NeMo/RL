# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect a trusted local Nano checkpoint before a cross-job continuation.

Run with python-MegatronPolicyWorker. This inspects saved state; successful
model/optimizer loading and prompt coverage must still be verified in E2E.
"""

import argparse
import json
import re
from pathlib import Path

import torch
import yaml

from nemo_rl.experience.interfaces import (
    FRONTIER_ORDINAL_KEY,
    NEXT_NEMO_GYM_TASK_INDEX_KEY,
    RESUME_BASE_ORDINAL_KEY,
    TRAINED_TASK_INDICES_KEY,
)
from nemo_rl.utils.checkpoint import CheckpointManager


def inspect_checkpoint(checkpoint_root: Path) -> dict:
    complete = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and re.fullmatch(r"step_\d+", path.name)
    ]
    if not complete:
        raise ValueError(f"No completed checkpoints in {checkpoint_root}")
    checkpoint = max(complete, key=lambda path: int(path.name.split("_")[1]))
    step = int(checkpoint.name.split("_")[1])
    required = (
        "training_info.json",
        "config.yaml",
        "train_dataloader.pt",
        "rollouts.pt",
        "replay_buffer.pt",
    )
    for name in required:
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
            raise FileNotFoundError(checkpoint / name)
    info = json.loads((checkpoint / "training_info.json").read_text())
    saved_config = yaml.safe_load((checkpoint / "config.yaml").read_text())
    planned_steps = int(saved_config["grpo"]["max_num_steps"])
    if info["current_step"] != step or info["total_steps"] != step:
        raise ValueError("Checkpoint directory and saved training step disagree")

    # These are trusted artifacts from this run, not downloaded pickle files.
    rollouts = torch.load(
        checkpoint / "rollouts.pt", map_location="cpu", weights_only=False
    )
    keys = (FRONTIER_ORDINAL_KEY, RESUME_BASE_ORDINAL_KEY, TRAINED_TASK_INDICES_KEY)
    if not all(key in rollouts for key in keys):
        raise ValueError(
            "Legacy checkpoint lacks frontier metadata; start the upgraded run fresh"
        )
    cut = int(rollouts[FRONTIER_ORDINAL_KEY])
    base = int(rollouts[RESUME_BASE_ORDINAL_KEY])
    next_index = int(rollouts[NEXT_NEMO_GYM_TASK_INDEX_KEY])
    trained = [int(index) for index in rollouts[TRAINED_TASK_INDICES_KEY]]
    if not (0 <= base <= cut <= next_index):
        raise ValueError(
            f"Invalid checkpoint cursor ordering: {base=}, {cut=}, {next_index=}"
        )
    if trained != sorted(set(trained)) or any(
        index < cut or index >= next_index for index in trained
    ):
        raise ValueError("Invalid trained ordinals above the checkpoint cut")
    dataloader = torch.load(
        checkpoint / "train_dataloader.pt", map_location="cpu", weights_only=False
    )
    if not isinstance(dataloader, dict) or not dataloader:
        raise ValueError("Missing dataloader snapshot")
    weights, optimizer = CheckpointManager.get_resume_paths(checkpoint)
    if weights is None or optimizer is None:
        raise ValueError("A full model and optimizer checkpoint is required")
    shards = list(weights.rglob("*.distcp"))
    if not shards or any(path.stat().st_size == 0 for path in shards):
        raise ValueError("Missing or empty Megatron distributed checkpoint shards")
    return {
        "status": "metadata-passed",
        "checkpoint": str(checkpoint.resolve()),
        "step": step,
        "planned_training_steps": planned_steps,
        "training_complete": step >= planned_steps,
        "resume_base_ordinal": base,
        "frontier_ordinal": cut,
        "next_ng_task_index": next_index,
        "trained_task_indices": trained,
        "optimizer_detected": True,
        "distributed_shards": len(shards),
        "files": {
            name: {
                "size": (checkpoint / name).stat().st_size,
                "mtime_ns": (checkpoint / name).stat().st_mtime_ns,
            }
            for name in required
        },
        "qualification": "Metadata check only; real restore and coverage verification remain required",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_checkpoint(args.checkpoint_root)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)
