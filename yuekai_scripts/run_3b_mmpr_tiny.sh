mkdir -p ./results
LOG_FILE=./results/vlm_grpo_3b_mmpr_tiny.log
exec > >(tee "${LOG_FILE}") 2>&1

uv run examples/run_vlm_grpo.py --config yuekai_scripts/configs/vlm_grpo_3b_mmpr_tiny.yaml \
    logger.wandb_enabled=true \
    logger.wandb.project=grpo-vlm \
    logger.wandb.name=vlm-grpo-3b-mmpr-tiny-64prompts \
    cluster.gpus_per_node=8
