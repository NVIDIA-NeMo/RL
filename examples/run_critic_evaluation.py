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

"""Evaluate one offline trajectory-value critic checkpoint without training."""

import argparse
import json
import os
import pprint
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from nemo_rl.algorithms.critic_pretraining import (
    MasterConfig,
    resolve_critic_checkpoint,
    setup_evaluation,
    validate,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a scalar trajectory-prefix critic checkpoint"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Exact step_N checkpoint directory or its value/weights directory",
    )
    parser.add_argument(
        "--dataset-name",
        default="test",
        help="Name used in metric and artifact prefixes",
    )
    return parser.parse_known_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    checkpoint = resolve_critic_checkpoint(args.checkpoint)
    if not args.config:
        checkpoint_config = checkpoint.checkpoint_path / "config.yaml"
        args.config = str(
            checkpoint_config
            if checkpoint_config.is_file()
            else Path(__file__).parent / "configs" / "critic_pretraining.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("the resolved critic-evaluation config must be a mapping")
    wandb_config = resolved.get("logger", {}).get("wandb")
    if isinstance(wandb_config, dict):
        for optional_key in ("id", "resume"):
            if wandb_config.get(optional_key) is None:
                wandb_config.pop(optional_key, None)
    master_config = MasterConfig.model_validate(resolved)
    master_config.logger["log_dir"] = get_next_experiment_dir(
        master_config.logger["log_dir"]
    )
    print("Final config:")
    pprint.pprint(master_config)

    init_ray()
    tokenizer = get_tokenizer(master_config.value["tokenizer"])
    _, validation_dataset = cast(
        tuple[
            AllTaskProcessedDataset | dict[str, AllTaskProcessedDataset],
            AllTaskProcessedDataset | None,
        ],
        setup_response_data(tokenizer, master_config.data, env_configs=None),
    )
    if validation_dataset is None:
        raise ValueError("critic evaluation requires data.validation")
    validation_datasets = {args.dataset_name: validation_dataset}

    value_model = None
    try:
        (
            value_model,
            _cluster,
            val_dataloader,
            logger,
            checkpoint,
            master_config,
        ) = setup_evaluation(
            master_config,
            tokenizer,
            validation_datasets,
            checkpoint,
        )
        metrics, timings = validate(
            value_model,
            val_dataloader,
            checkpoint.step,
            master_config,
            logger,
        )
        summary = {
            "status": "passed",
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "weights_path": str(checkpoint.weights_path),
            "checkpoint_step": checkpoint.step,
            "checkpoint_training_info": checkpoint.training_info,
            "dataset_name": args.dataset_name,
            "metrics": metrics,
            "timings": timings,
        }
        summary_path = Path(master_config.logger["log_dir"]) / "evaluation_summary.json"
        _atomic_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"Evaluation summary: {summary_path}", flush=True)
    finally:
        if value_model is not None:
            value_model.shutdown()


if __name__ == "__main__":
    main()
