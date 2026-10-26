#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=40
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
    # gen_kl_error is the only metric in 20 steps at this lr that tells a working LoRA
    # refit from a broken one: with the adapters merged into the refit weights it sits
    # on the bf16 noise floor (0.0010-0.0011 for all 20 steps on 2 x 4 GB200 GPUs); when
    # the inference engine keeps the base weights it climbs monotonically as the adapter
    # drifts (0.0013 / 0.0014 / 0.0015 / 0.0015 over steps 17-20) while reward and the
    # 256-sample validation accuracy stay within noise. The previous 0.002 bound passed
    # both, so the refit bug fixed in #4470 was never caught here.
    uv run tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/gen_kl_error"], 20) < 0.002' \
        'data["train/gen_kl_error"]["20"] < 0.0013' \
        'max(data["train/gen_kl_error"]) < 0.0013' \
        'max(data["train/reward"]) > 0.35' \
        'mean(data["timing/train/total_step_time"], 2) < 80'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi