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
"""Multi-teacher cross-tokenizer off-policy distillation entrypoint."""

from __future__ import annotations

import argparse
import os
import pprint
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.xtoken_off_policy_distillation import (
    CSDMasterConfig,
    MasterConfig,
    setup,
    validate_csd_packing_setup,
    validate_csd_tokenizer_reuse,
    xtoken_off_policy_distillation_train,
)
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from nemo_rl.utils.outdated_config_checks import check_outdated_config


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args; unknown args become Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Run multi-teacher cross-tokenizer off-policy distillation"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resolved-config-output",
        type=str,
        default=None,
        help="Optional path for the fully resolved, validated configuration",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main(default_config: str = "xtoken_off_policy_distillation.yaml") -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", default_config)

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    check_outdated_config(config)

    config = (
        CSDMasterConfig(**config)
        if config.get("algorithm", "xtoken") == "csd"
        else MasterConfig(**config)
    )
    validate_csd_packing_setup(config)
    if args.resolved_config_output:
        resolved_config_path = Path(args.resolved_config_output).resolve()
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            config=OmegaConf.create(config.model_dump(mode="json")),
            f=resolved_config_path,
        )
        print(f"Resolved config written to: {resolved_config_path}", flush=True)
    # X-Token selects cross-tokenizer teachers through projection_matrix_path;
    # CSD uses its sub-token table pair. CSD validates full tokenizer/template
    # equivalence before allowing a same-tokenizer teacher to reuse student IDs.

    print("Final config:")
    pprint.pprint(config)

    config.logger.log_dir = get_next_experiment_dir(config.logger.log_dir)
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}",
            flush=True,
        )

    # Student tokenizer + one tokenizer per teacher (same-tokenizer teachers
    # included — needed for vocab sizing in setup()).
    student_tokenizer = get_tokenizer(config.policy["tokenizer"])
    teacher_tokenizers = [
        get_tokenizer(teacher.policy_config()["tokenizer"])
        for teacher in config.teachers
    ]
    if config.algorithm == "csd":
        validate_csd_tokenizer_reuse(config, student_tokenizer, teacher_tokenizers)

    # `env_configs=None` skips the env-creation block (no rollout path);
    # `setup_response_data` then handles dataset construction, the optional
    # train/val split via `split_validation_size`, `data.default` merging,
    # and validation-from-config — features the prior manual route silently
    # dropped.
    train_dataset, val_dataset = setup_response_data(
        student_tokenizer, config.data, env_configs=None
    )

    init_ray()

    (
        student_policy,
        teacher_policies,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        off_policy_distillation_state,
        master_config,
    ) = setup(config, student_tokenizer, teacher_tokenizers, train_dataset, val_dataset)

    # The checkpointer owns background async-checkpoint finalization threads;
    # the context manager guarantees they are flushed (rename + delete) on exit.
    with checkpointer:
        xtoken_off_policy_distillation_train(
            student_policy,
            teacher_policies,
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
