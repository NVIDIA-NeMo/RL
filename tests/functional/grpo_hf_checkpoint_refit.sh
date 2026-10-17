#!/bin/bash
# Two-GPU Megatron -> managed async vLLM HF-checkpoint refit smoke.
# Runs ten training steps and checks policy/generation logprob agreement.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=${HF_REFIT_TEST_RUN_DIR:-$SCRIPT_DIR/$EXP_NAME}
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}


rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR
uv run python "$SCRIPT_DIR/prepare_refit_math_data.py" "$EXP_DIR/math.jsonl"

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    data.train.dataset_name=ResponseDataset \
    ++data.train.data_path=$EXP_DIR/math.jsonl \
    ++data.train.input_key=input \
    ++data.train.output_key=output \
    data.train.split_validation_size=0 \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=512 \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.dtensor_cfg.enabled=false \
    policy.generation.backend=vllm \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.refit_transport=hf_checkpoint \
    ++policy.generation.refit_cfg.hf_checkpoint.checkpoint_dir=$EXP_DIR/hf_exports \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=10 \
    grpo.val_period=0 \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# A broken refit corrupts the gen weights -> the importance-sampling ratio
# (train vs gen logprobs) explodes; assert it stays sane across all ten steps.
uv run tests/check_metrics.py $JSON_METRICS \
    'len(data["train/token_mult_prob_error"]) >= 10' \
    'max(data["train/token_mult_prob_error"]) < 1.05'
uv run python "$SCRIPT_DIR/check_hf_refit_exports.py" "$EXP_DIR/hf_exports" \
    --minimum-versions 10 --output "$EXP_DIR/refit-validation.json"
