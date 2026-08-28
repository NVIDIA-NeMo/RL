#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

# tp_size=2 variant of grpo_sglang_nixl_non_colocated.sh.
#
# With one one-GPU engine the machinery this path exists for degenerates: a
# single stream has nothing to align, ``_validate_rank_batches`` compares
# nothing, and ``update_weights_from_tensor`` gets a one-payload list, so the
# payload-i -> TP-rank-i scatter contract never runs against a real server. As
# ``test_checkpoint_engine_payload_index_matches_sglang_rank`` notes, a
# transposed payload list would load every shard onto the wrong GPU while still
# succeeding; only tp_size>1 can catch that, and the token_mult_prob_error gate
# below is what would catch it.
#
# 3 GPUs: 2 for the SGLang engine, 1 for the trainer.

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="$SCRIPT_DIR/$EXP_NAME"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "$EXP_DIR" "$LOG_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR"

cd "$PROJECT_ROOT"
uv run --group test coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    "$PROJECT_ROOT/examples/run_grpo.py" \
    --config "$PROJECT_ROOT/examples/configs/grpo_math_1B_sglang.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=3 \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=2 \
    policy.generation.refit_transport=nixl \
    policy.generation.use_async_rollouts=false \
    policy.generation.sglang_cfg.tp_size=2 \
    policy.generation.sglang_cfg.sglang_server_config.num_gpus=2 \
    policy.generation.sglang_cfg.sglang_server_config.num_gpus_per_engine=2 \
    policy.generation.sglang_cfg.sglang_server_config.needs_offload=false \
    policy.generation.sglang_cfg.sglang_server_config.cpu_weight_backup=false \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

uv run tests/check_metrics.py "$JSON_METRICS" \
    'max(data["train/token_mult_prob_error"]) < 1.05'
