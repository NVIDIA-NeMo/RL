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
"""Entrypoint for flow-GRPO training (Qwen-Image flow-grpo)."""

import argparse
import os
import pprint

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nemo_rl.algorithms.flow_grpo import (
    MasterConfig,
    flow_grpo_train,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data.datasets.text_to_image_prompt import (
    TextToImagePromptDataset,
    text_to_image_collate_fn,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.image_reward_environment import ImageRewardEnvironment
from nemo_rl.models.diffusion.flow_grpo_policy import FlowGRPOPolicy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import Logger, get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run flow-GRPO training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: examples/configs/flow_grpo_qwen_image_ocr.yaml)",
    )
    return parser.parse_known_args()


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "flow_grpo_qwen_image_ocr.yaml",
        )

    cfg = load_config(args.config)
    if overrides:
        cfg = parse_hydra_overrides(cfg, overrides)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print("Final config:")
    pprint.pprint(cfg)

    # Validate against the schema; field defaults live on the BaseModels
    # (config-conventions v2), so downstream code reads values directly.
    master = MasterConfig.model_validate(cfg)

    master.logger["log_dir"] = get_next_experiment_dir(master.logger["log_dir"])
    print(f"📊 log_dir: {master.logger['log_dir']}")

    # Seed the driver process too: DataLoader(shuffle=True) draws from the
    # global RNG, so without this the prompt order differs across runs.
    set_seed(master.flow_grpo.seed)

    init_ray()

    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[master.cluster["gpus_per_node"]]
        * master.cluster["num_nodes"],
        use_gpus=True,
        max_colocated_worker_groups=1,
    )

    policy = FlowGRPOPolicy(cluster=cluster, config=master.policy)

    env = ImageRewardEnvironment(master.env.image_reward)

    train_ds = TextToImagePromptDataset(master.data.train.prompt_file)
    val_ds = (
        TextToImagePromptDataset(master.data.val.prompt_file)
        if master.data.val is not None
        else None
    )

    n_gpus = int(master.cluster["gpus_per_node"]) * int(master.cluster["num_nodes"])
    if n_gpus > 1 and master.flow_grpo.num_prompts_per_step % n_gpus != 0:
        raise ValueError(
            f"flow_grpo.num_prompts_per_step={master.flow_grpo.num_prompts_per_step} "
            f"must be a multiple of the {n_gpus} DP workers, otherwise every "
            "rollout silently falls back to a single worker"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=master.flow_grpo.num_prompts_per_step,
        shuffle=True,
        # A short trailing batch would not split across DP workers.
        drop_last=True,
        collate_fn=text_to_image_collate_fn,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=master.flow_grpo.num_prompts_per_step,
            shuffle=False,
            collate_fn=text_to_image_collate_fn,
        )
        if val_ds is not None
        else None
    )

    logger = Logger(master.logger)
    checkpointer = CheckpointManager(master.checkpointing)

    # NotRequired in LoggerConfig: absent means "save no validation images".
    num_val_images = master.logger.get("num_val_samples_to_print")

    try:
        flow_grpo_train(
            policy=policy,
            env=env,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            master_config=master,
            logger=logger,
            checkpointer=checkpointer,
            val_image_dir=os.path.join(master.logger["log_dir"], "val_images"),
            num_val_images_to_save=int(num_val_images)
            if num_val_images is not None
            else 0,
        )
    finally:
        checkpointer.shutdown()
        env.shutdown()
        policy.shutdown()


if __name__ == "__main__":
    main()
