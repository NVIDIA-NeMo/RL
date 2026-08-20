#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
CKPT_DIR=$EXP_DIR/ckpts
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

# clean up checkpoint directory on exit
trap "rm -rf $CKPT_DIR" EXIT

# async colocated, reshard mode: TE TP2 training; inference_optimized TP1
# generation on a dedicated model, resharded into on every wake.
cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.generation.backend=megatron \
    ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    ++policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    grpo.async_grpo.enabled=true \
    grpo.async_grpo.max_trajectory_age_steps=1 \
    grpo.async_grpo.in_flight_weight_updates=true \
    loss_fn.use_importance_sampling_correction=true \
    grpo.max_num_steps=3 \
    grpo.val_period=1 \
    grpo.max_val_samples=8 \
    grpo.val_batch_size=8 \
    cluster.gpus_per_node=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.save_period=2 \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Smoke-level threshold (matches grpo_megatron_generation_async_gym.sh); tighten after CI runs.
uv run tests/check_metrics.py $JSON_METRICS \
    'median(data["train/gen_kl_error"]) < 1.3' \
    '"3" in data["train/loss"]' \
    '"2" in data["validation/accuracy"]'

# The dedicated inference model must actually be built — guard against this
# leg silently degenerating to the matched-impl (reshardless) path.
if ! grep -q "\[colocated-reshard\] building dedicated inference model" $RUN_LOG; then
    echo "FAIL: dedicated-model build log line not found (reshard path not exercised)"
    exit 1
fi

# The non-save validation (step 1) wakes an already-serving engine; the
# worker must skip it (guards against redundant per-validation resharding).
if ! grep -q "prepare_for_generation: engine already serving, skipping" $RUN_LOG; then
    echo "FAIL: idempotent-wake skip log line not found (redundant reshard on validation?)"
    exit 1
fi

# The save-bound step must defer the engine wake past the checkpoint save.
# With `val_period=1`, the validation always intervenes before the save.
if ! grep -q "Keeping colocated engine asleep for checkpointing" $RUN_LOG; then
    echo "FAIL: deferred-wake log line not found (colocated checkpoint path not exercised)"
    exit 1
fi

if [[ ! -f $CKPT_DIR/step_2/replay_buffer.pt ]]; then
    echo "FAIL: replay_buffer.pt not found in step_2 checkpoint"
    exit 1
fi
