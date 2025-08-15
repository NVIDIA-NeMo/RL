#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
NUM_NODES=${NUM_NODES:-1}
MAX_STEPS=${MAX_STEPS:-70}
NUM_MINUTES=240
TP_SIZE=${TP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
VLLM_TP=${VLLM_TP:-1}

SEQ_PACKING=false
if [[ $CP_SIZE -gt 1 ]]; then
    SEQ_PACKING=true
fi

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
    logger.wandb.name=$EXP_NAME-$MODEL_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    policy.model_name=$MODEL_NAME \
    cluster.num_nodes=$NUM_NODES \
    policy.generation.vllm_cfg.tensor_parallel_size=$VLLM_TP \
    policy.megatron_cfg.context_parallel_size=$CP_SIZE \
    policy.megatron_cfg.tensor_model_parallel_size=$TP_SIZE \
    policy.megatron_cfg.expert_model_parallel_size=$EP_SIZE \
    policy.sequence_packing.enabled=$SEQ_PACKING \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/token_mult_prob_error"], exclude_outliers=5) < 1.05'
fi
