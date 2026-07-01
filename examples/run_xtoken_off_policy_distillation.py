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

from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.xtoken_off_policy_distillation import (
    MasterConfig,
    setup,
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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI args; unknown args become Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Run multi-teacher cross-tokenizer off-policy distillation"
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
            os.path.dirname(__file__), "configs", "xtoken_off_policy_distillation.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)

    # Per-teacher same-vocab vs cross-tokenizer is determined solely by
    # `teachers[i].projection_matrix_path` (null => same-vocab direct KL; set =>
    # cross-tokenizer). The consistency check (a same-vocab teacher must
    # actually share the student's vocab) needs the real tokenizers and lives
    # in `setup()` — comparing tokenizer *names* here is wrong, e.g.
    # Llama-3.2-3B and Llama-3.2-1B have different names but the same vocab.

    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}",
            flush=True,
        )

    init_ray()

    # Student tokenizer + one tokenizer per teacher (same-tokenizer teachers
    # included — needed for vocab sizing in setup()).
    student_tokenizer = get_tokenizer(config.policy["tokenizer"])
    teacher_tokenizers = [
        get_tokenizer(teacher.policy_config()["tokenizer"])
        for teacher in config.teachers
    ]

    # `env_configs=None` skips the env-creation block (no rollout path);
    # `setup_response_data` then handles dataset construction, the optional
    # train/val split via `split_validation_size`, `data.default` merging,
    # and validation-from-config — features the prior manual route silently
    # dropped.
    train_dataset, val_dataset = setup_response_data(
        student_tokenizer, config.data, env_configs=None
    )

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
