# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Build the inspection recipe from the saved, authored YAML chain."""

from pathlib import Path

from omegaconf import OmegaConf

from config_loader import load_config, register_omegaconf_resolvers

root = Path(__file__).resolve().parents[1]
source = root / "evidence/source-configs/examples/configs/recipes/vlm"
register_omegaconf_resolvers()
cfg = load_config(source / "super_vl_35_mixed_teachers_v2_test_batch4h.yaml")
cfg.policy.model_name = "${oc.env:MM_TRAINER_MODEL_PATH}"
cfg.policy.tokenizer.name = "${policy.model_name}"
cfg.policy.tokenizer.chat_template = "${policy.model_name}/chat_template.jinja"
cfg.checkpointing.checkpoint_dir = "${oc.env:MM_TRAINER_RESULTS_DIR}/checkpoints"
cfg.logger.log_dir = "${oc.env:MM_TRAINER_RESULTS_DIR}/logs"
cfg.logger.wandb.entity = "${oc.env:MM_TRAINER_WANDB_ENTITY,nvidia}"
cfg.logger.wandb.project = (
    "${oc.env:MM_TRAINER_WANDB_PROJECT,rohit-unified-teacher-supervl3p5}"
)
cfg.logger.wandb.name = "${oc.env:MM_TRAINER_WANDB_NAME}"
cfg.logger.wandb.id = "${oc.env:MM_TRAINER_WANDB_ID}"
cfg.env.nemo_gym.nemo_gym_log_dir = "${oc.env:MM_TRAINER_RESULTS_DIR}/logs/nemo_gym"
cfg.env.nemo_gym.image_tools_simple_agent.responses_api_agents.image_tools_agent.crop_dir = "${oc.env:MM_TRAINER_RESULTS_DIR}/image_tool_outputs"
dest = root / "nemo-rl/configs"
dest.mkdir(parents=True, exist_ok=True)
header = "# SuperVL3.5 V2 production: authored HSG recipe composed on 2026-10-04.\n# Paths and new-run identity come from the launch environment.\n"
with (dest / "supervl3p5-v2-production.yaml").open("w") as stream:
    stream.write(header + OmegaConf.to_yaml(cfg, resolve=False))
