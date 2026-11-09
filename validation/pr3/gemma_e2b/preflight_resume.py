# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that a Gemma continuation keeps its original training configuration."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    validate_single_controller_config,
)
from nemo_rl.data.utils import update_single_dataset_config
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

HERE = Path(__file__).resolve().parent


def main() -> None:
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    assert torch.cuda.device_count() == 8
    register_omegaconf_resolvers()
    path = HERE / "resume650.yaml"
    cfg = OmegaConf.to_container(load_config(str(path)), resolve=True)
    validate_single_controller_config(MasterConfig(**cfg))
    original = json.loads((HERE / "results/resolved-config.json").read_text())
    expected = copy.deepcopy(original)
    expected["grpo"]["max_num_steps"] = 650
    expected["logger"]["wandb"]["name"] = cfg["logger"]["wandb"]["name"]
    assert cfg == expected, "Continuation changed settings beyond target/display name"
    status_path = (
        HERE / "results/streaming-la0/checkpoints/latest_checkpoint_status.json"
    )
    step = json.loads(status_path.read_text())["last_checkpoint_step"]
    assert 150 <= step < 650, step
    control = json.loads((HERE / "slurm/resume650/state.json").read_text())
    stages = [
        stage
        for stage in control["stages"]
        if stage["job_id"] == os.environ["SLURM_JOB_ID"]
    ]
    assert len(stages) == 1 and stages[0]["resume_from"] == step, (
        "Checkpoint changed outside the recorded continuation"
    )
    checkpoint = status_path.parent / f"step_{step}"
    info = json.loads((checkpoint / "training_info.json").read_text())
    assert (
        info["total_steps"]
        == info["trainer_version"]
        == info["sampler_dispatch_index"]
        == step
    )
    assert info["consumed_samples"] == step * 32 and info["current_epoch"] == 0
    for name in (
        "policy/weights",
        "policy/optimizer",
        "train_dataloader.pt",
        "data_plane",
        "rollout_recovery.pt",
    ):
        assert (checkpoint / name).exists(), name
    sources = json.loads((HERE / "results/provenance.json").read_text())[
        "training_source_sha256"
    ]
    for name, expected_hash in sources.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected_hash, (
            name
        )
    # Match the same post-normalization dataset representation stored by W&B.
    model_hash = hashlib.sha256(
        (Path(cfg["policy"]["model_name"]) / "config.json").read_bytes()
    ).hexdigest()
    assert (
        model_hash
        == json.loads((HERE / "results/provenance.json").read_text())[
            "model_config_sha256"
        ]
    )
    data = copy.deepcopy(cfg["data"])
    if isinstance(data["train"], dict):
        data["train"] = [data["train"]]
    for dataset in data["train"]:
        update_single_dataset_config(dataset, data["default"])
    for baseline in json.loads((HERE / "results/baselines.json").read_text()).values():
        assert data == baseline["config"]["data"]
    provenance = {
        "job_id": os.environ["SLURM_JOB_ID"],
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "resume_checkpoint_step": step,
        "training_info": info,
        "training_source_sha256": sources,
        "recipe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_config_sha256": model_hash,
        "only_target_and_display_name_changed": True,
        "fixed_input_streaming_parity": "NOT ASSESSED; original Qwen BF16 gate remains FAILED",
    }
    (
        HERE / f"results/resume650-provenance-{os.environ['SLURM_JOB_ID']}.json"
    ).write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Gemma650 preflight passed; restoring checkpoint{step}, same config and W&B identity",
        flush=True,
    )


if __name__ == "__main__":
    main()
