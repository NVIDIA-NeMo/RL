#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=32
STEPS_PER_RUN=11  # about 12min/step (set to 11 just to make sure LR schedule doesn't crash)
MAX_STEPS=11
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
	--config examples/configs/recipes/llm/grpo-gym-nanov3-4n8g-megatron-slim.yaml \
	data.train_jsonl_fpath=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_nano/users/pjin/data/nano-v3-posttraining-data/curriculum_v7_acrid-teal_main_rename.train.jsonl \
	data.validation_jsonl_fpath=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_nano/users/pjin/data/nano-v3-posttraining-data/curriculum_v7_acrid-teal_main_rename.val.jsonl \
	policy.model_name=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_nano/users/pjin/checkpoints/nano-v3-sft-64gbs-nickel-capybara-5e-5-constant-wd-0-load-bal-1e-4-lcx3-pretool-base-temp1-iter-0013600-hf \
    \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/token_mult_prob_error"]) < 1.05' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.05"
fi

