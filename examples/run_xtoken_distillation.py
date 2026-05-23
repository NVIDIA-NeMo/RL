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
"""Single-teacher cross-tokenizer off-policy distillation entrypoint."""

from __future__ import annotations

import argparse
import os
import pprint

from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.xtoken_distillation import (
    MasterConfig,
    setup,
    xtoken_distillation_train,
)
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args; unknown args become Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Run single-teacher cross-tokenizer off-policy distillation"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "xtoken_distillation.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: "
            f"{config['checkpointing']['checkpoint_dir']}",
            flush=True,
        )

    init_ray()

    # Two tokenizers — one each for student and teacher.
    student_tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    teacher_tokenizer = get_tokenizer(config["teacher"]["tokenizer"])

    # Load arrow_text dataset directly (no env / no rollout path).
    train_data = load_response_dataset(config["data"]["train"])
    train_dataset = AllTaskProcessedDataset(
        train_data.dataset,
        student_tokenizer,
        train_data.task_spec,
        train_data.processor,
        max_seq_length=config["data"]["max_input_seq_length"],
    )
    val_dataset = None
    if config["data"].get("validation") is not None:
        val_data = load_response_dataset(config["data"]["validation"])
        val_dataset = AllTaskProcessedDataset(
            val_data.dataset,
            student_tokenizer,
            val_data.task_spec,
            val_data.processor,
            max_seq_length=config["data"]["max_input_seq_length"],
        )

    (
        student_policy,
        teacher_policy,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        off_policy_distillation_state,
        master_config,
    ) = setup(config, student_tokenizer, teacher_tokenizer, train_dataset, val_dataset)

    xtoken_distillation_train(
        student_policy,
        teacher_policy,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        off_policy_distillation_state,
        master_config,
    )


if __name__ == "__main__":
    main()
