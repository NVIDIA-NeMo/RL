#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
STEPS_PER_RUN=50  # about 3min/step
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

# Create dataset if doesn't exist
DATA_DIR=$(realpath ./nano-v3-posttraining-data)
if [[ ! -d $DATA_DIR ]]; then
    echo "Downloading dataset to $DATA_DIR"
    huggingface-cli download nvidia/Nemotron-3-Nano-RL-Training-Blend --repo-type dataset --local-dir $DATA_DIR
    chmod a+x $DATA_DIR/create_nanov3_jsonl.py
    $DATA_DIR/create_nanov3_jsonl.py --input $DATA_DIR/train.jsonl --output $DATA_DIR/train+dapo+skyrl.jsonl
    tail -n 10 $DATA_DIR/train+dapo+skyrl.jsonl > $DATA_DIR/valid+dapo+skyrl.jsonl
else
    echo "Dataset already exists in $DATA_DIR"
fi

# Run the experiment
cd $PROJECT_ROOT
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
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
    data.train_jsonl_fpath=$DATA_DIR/train+dapo+skyrl.jsonl \
    data.validation_jsonl_fpath=$DATA_DIR/valid+dapo+skyrl.jsonl \
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

## Convert 16k checkpoint
#uv run examples/converters/convert_dcp_to_hf.py \
#  --config=$CKPT_DIR/step_${MAX_STEPS}/config.yaml \
#  --dcp-ckpt-path=$CKPT_DIR/step_${MAX_STEPS}/policy/weights \
#  --hf-ckpt-path=$CKPT_DIR/grpo-deepscaler-16k-${MAX_STEPS}-hf
#
## Run eval
#uv run examples/run_eval.py \
#    generation.model_name=$CKPT_DIR/grpo-deepscaler-16k-${MAX_STEPS}-hf \
#    data.prompt_file=examples/prompts/cot.txt \
#    generation.vllm_cfg.max_model_len=32768 \
#    generation.vllm_cfg.enforce_eager=True \
#    generation.temperature=1.0 \
#    eval.num_tests_per_prompt=16 \
#    2>&1 | tee ${RUN_LOG}.aime-16k
#
#cat ${RUN_LOG}.aime-16k       | grep "score=" | sed 's/.*score=\([^ ]*\).*/{"score": \1}/' > ${RUN_LOG}-16k-metric.json
# 
## 240 step checkpoint 0.3
#uv run tests/check_metrics.py ${RUN_LOG}-16k-metric.json \
#  'data["score"] >= 0.2396'
#