# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the fixed Nano SWE recipe and record the inputs of a fresh run."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

RECIPE = (
    "examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml"
)


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(mode: str) -> None:
    run = Path(os.environ["NANO35_RUN_DIR"]).resolve()
    model = Path(os.environ["NANO35_MODEL"])
    data = Path(os.environ["NANO35_DATA"])
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if len(rows) != 403:
        raise ValueError(
            f"This recipe requires the recorded 403-row dataset, got {len(rows)}"
        )
    index = model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    for name in ["config.json", "tokenizer_config.json", *shards]:
        if not (model / name).is_file() or (model / name).stat().st_size == 0:
            raise FileNotFoundError(model / name)

    register_omegaconf_resolvers()
    cfg = MasterConfig(**OmegaConf.to_container(load_config(RECIPE), resolve=True))
    with CheckpointManager(cfg.checkpointing):
        pass
    cfg_dict = cfg.model_dump()
    if cfg.policy["generation"]["backend"] != "trtllm":
        raise ValueError("This launcher requires TRT-LLM")
    if not cfg.grpo.async_grpo.enabled or not cfg.checkpointing["save_optimizer"]:
        raise ValueError("Async GRPO and optimizer checkpoints must remain enabled")
    if cfg.logger["wandb_enabled"]:
        raise ValueError("This validation recipe records metrics locally")

    text = OmegaConf.to_yaml(OmegaConf.create(cfg_dict))
    manifest = {
        "run_dir": str(run),
        "model": str(model),
        "model_index_sha256": sha256(index),
        "model_config_sha256": sha256(model / "config.json"),
        "model_shards": len(shards),
        "data_path": str(data),
        "data_rows": len(rows),
        "data_sha256": sha256(data),
        "data_identity": "SWE-bench Verified 403-row local E2E dataset; not the official Ultra SWE blend",
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "recipe_sha256": sha256(Path(RECIPE)),
        "image_sha256": os.environ["NANO35_IMAGE_SHA256"],
        "container_fingerprint": Path("/opt/nemo_rl_container_fingerprint")
        .read_text()
        .strip(),
    }
    manifest_path = run / "run-manifest.json"
    config_path = run / "config.resolved.yaml"
    if mode == "fresh":
        if manifest_path.exists() or list((run / "checkpoints").glob("*step_*")):
            raise ValueError(
                "Fresh mode requires a new run directory with no checkpoint history"
            )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        config_path.write_text(text)
    else:
        previous = json.loads(manifest_path.read_text())
        if previous != manifest or config_path.read_text() != text:
            raise ValueError(
                "Resume inputs differ from the original run manifest; refusing to resume"
            )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fresh", "resume"))
    prepare(parser.parse_args().mode)
