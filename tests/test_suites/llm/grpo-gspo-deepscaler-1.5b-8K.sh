#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=40
MAX_STEPS=40
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo_math.py \
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
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.1"

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi

# TODO: enable in subsequent PR to do a quick accuracy check
## Convert 8k checkpoint
#uv run examples/converters/convert_dcp_to_hf.py \
#  --config=$CKPT_DIR/step_${MAX_STEPS}/config.yaml \
#  --dcp-ckpt-path=$CKPT_DIR/step_${MAX_STEPS}/policy/weights \
#  --hf-ckpt-path=$CKPT_DIR/gspo-deepscaler-8k-${MAX_STEPS}-hf
#
## Run eval
#uv run examples/run_eval.py \
#    generation.model_name=$CKPT_DIR/gspo-deepscaler-8k-${MAX_STEPS}-hf \
#    data.prompt_file=examples/prompts/cot.txt \
#    generation.vllm_cfg.max_model_len=32768 2>&1 | tee ${RUN_LOG}.aime-8k
#
#cat ${RUN_LOG}.aime-8k       | grep "score=" | sed 's/.*score=\([^ ]*\).*/{"score": \1}/' > ${RUN_LOG}-8k-metric.json
# 
#uv run tests/check_metrics.py ${RUN_LOG}-8k-metric.json \
#  'data["score"] >= 0.25' \
#
##uv run examples/run_eval.py \
##    generation.model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
##    data.prompt_file=examples/prompts/cot.txt \
##    generation.vllm_cfg.max_model_len=32768 2>&1 | tee ${RUN_LOG}.aime-baseline
#
##cat ${RUN_LOG}.aime-baseline | grep "score=" | sed 's/.*score=\([^ ]*\).*/{"score": \1}/' > ${RUN_LOG}-baseline-metric.json
#
##uv run tests/check_metrics.py ${RUN_LOG}-baseline-metric.json \
##  'data["score"] == 0.2' \
