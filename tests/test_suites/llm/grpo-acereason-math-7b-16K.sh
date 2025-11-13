#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=100
MAX_STEPS=1000
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
        'mean(data["train/token_mult_prob_error"]) < 1.05' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.05"
fi

# Convert 16k checkpoint
uv run examples/converters/convert_dcp_to_hf.py \
  --config=$CKPT_DIR/step_${MAX_STEPS}/config.yaml \
  --dcp-ckpt-path=$CKPT_DIR/step_${MAX_STEPS}/policy/weights \
  --hf-ckpt-path=$CKPT_DIR/grpo-acereason-math-7b-16k-${MAX_STEPS}-hf

# Run eval on AceReason-Math dataset
uv run examples/run_eval.py \
    generation.model_name=$CKPT_DIR/grpo-acereason-math-7b-16k-${MAX_STEPS}-hf \
    data.prompt_file=examples/prompts/acemath_qwen_cot.txt \
    generation.vllm_cfg.max_model_len=16384 \
    generation.vllm_cfg.enforce_eager=True \
    generation.temperature=1.0 \
    eval.num_tests_per_prompt=16 \
    2>&1 | tee ${RUN_LOG}.acereason-eval

cat ${RUN_LOG}.acereason-eval | grep "score=" | sed 's/.*score=\([^ ]*\).*/{"score": \1}/' > ${RUN_LOG}-16k-metric.json

# Set baseline score for AceReason-Math evaluation (adjust based on expected performance)
uv run tests/check_metrics.py ${RUN_LOG}-16k-metric.json \
  'data["score"] >= 0.30'  # Baseline score to be adjusted based on actual performance

# Performance tracking comments
# ========================================================
# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B baseline performance
# ========================================================
# This section will be updated with baseline performance metrics
# after initial runs to establish proper thresholds
