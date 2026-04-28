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
"""Run cross-tokenizer off-policy distillation.

Mirrors examples/run_distillation.py: load a YAML config, build dataset and
policies, then call off_policy_distillation_train. Off-policy means student
training consumes fixed (prompt, response) pairs from the dataset rather
than newly generated rollouts; teacher logits are computed once per batch
and shipped to the student via CUDA IPC.
"""

import argparse
import os

from omegaconf import OmegaConf

from nemo_rl.algorithms.off_policy_distillation import (
    OffPolicyMasterConfig,
    off_policy_distillation_train,
    setup,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run off-policy distillation training with configuration"
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
            os.path.dirname(__file__), "configs", "off_policy_distillation.yaml"
        )

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    config: OffPolicyMasterConfig = OmegaConf.to_container(config, resolve=True)
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])

    init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    env_configs = config.get("env") or None
    if env_configs is None:
        dataset, val_dataset = setup_response_data(
            tokenizer, config["data"], None
        )
    else:
        (
            dataset,
            val_dataset,
            _task_to_env,
            _val_task_to_env,
        ) = setup_response_data(tokenizer, config["data"], env_configs)

    (
        student_policy,
        teacher_policies,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
        token_aligners,
        teacher_tokenizers,
    ) = setup(config, tokenizer, dataset, val_dataset)

    off_policy_distillation_train(
        student_policy,
        teacher_policies,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
        token_aligners=token_aligners,
        teacher_tokenizers=teacher_tokenizers,
    )


if __name__ == "__main__":
    main()
