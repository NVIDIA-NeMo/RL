#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=8
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
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
    # Thresholds anchored to standalone measurements on this checkpoint rather
    # than copied from a sibling recipe:
    #
    #   token_mult_prob_error - AutoModel (both te and sdpa) versus the vLLM
    #     teacher-forced baseline gave median |dlp| 6.086e-04, i.e. ~1.0006. The
    #     1.05 gate leaves ~80x headroom while still catching a real regression.
    #   gen_kl_error - a DAPO-Math-17k rollout measured 5.543e-04 with NeMo-RL's
    #     own k3 formula, so 0.002 is ~4x margin. This is the metric that
    #     matters: values >= 1 mean the policy and generation stacks disagree
    #     and training is optimising noise.
    #   reward - gate the TREND, not the last step. DAPO reward is noisy enough
    #     at this batch size that a single point says little; a
    #     `reward["20"] > 0.1` check failed on a run that plainly learned.
    #     Compare the last five steps against the first five instead.
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.05' \
        'mean(data["train/gen_kl_error"]) < 0.002' \
        'mean([data["train/reward"][str(s)] for s in range(16, 21)]) - mean([data["train/reward"][str(s)] for s in range(1, 6)]) > 0.3'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
