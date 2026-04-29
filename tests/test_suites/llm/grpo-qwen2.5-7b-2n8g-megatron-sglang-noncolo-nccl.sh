#!/bin/bash
# E2E smoke test for the SGLang non-colocated NCCL bridge refit.
#
# Layout (2 nodes, 8 GPUs each):
#   Training:  Megatron, TP=2, DP=4 on 1 node.
#   Inference: SGLang, 2 server replicas x TP=4 on 1 node.
#   Refit: per-bucket dist.broadcast on a bridge ProcessGroup spanning
#   train rank 0 + every SGLang TP rank; metadata POSTed to engines via
#   parallel HTTP threads.
#
# 7B / 2 nodes is the minimum that meaningfully exercises the bridge:
# ~14 GiB BF16 spans multiple 1-GiB buckets so bucket boundaries +
# flush_cache=True on the last bucket are both covered, and 30 steps is
# long enough to surface hangs / leaks.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=30
MAX_STEPS=30
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached

# Bridge env-var defaults (all overridable from the launcher):
#   NRL_SGLANG_BRIDGE_PORT, NRL_SGLANG_BRIDGE_GROUP_NAME — bridge
#       rendezvous TCPStore port + ProcessGroup name.
#   NRL_REFIT_BUCKET_BYTES — bytes per NCCL bucket (~1 GiB default).
#   NRL_SGLANG_{NCCL_HTTP,GENERATE}_TIMEOUT_S — engine-side request timeouts.
# NRL_SGLANG_NONCOLO_TRANSPORT is informational; today the non-colo path
# always takes the NCCL bridge.
export NRL_SGLANG_NONCOLO_TRANSPORT=${NRL_SGLANG_NONCOLO_TRANSPORT:-nccl}
export NRL_SGLANG_BRIDGE_PORT=${NRL_SGLANG_BRIDGE_PORT:-29500}
export NRL_SGLANG_BRIDGE_GROUP_NAME=${NRL_SGLANG_BRIDGE_GROUP_NAME:-nrl-sglang-bridge}
export NRL_REFIT_BUCKET_BYTES=${NRL_REFIT_BUCKET_BYTES:-1073741824}
export NRL_SGLANG_NCCL_HTTP_TIMEOUT_S=${NRL_SGLANG_NCCL_HTTP_TIMEOUT_S:-7200}
export NRL_SGLANG_GENERATE_TIMEOUT_S=${NRL_SGLANG_GENERATE_TIMEOUT_S:-1800}

# Lift NCCL collective watchdog from 600s to 9000s for slow first-step
# warmup (NCCL channel allocation etc.).
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-9000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-9000}

# Pick the python invocation method. Default ``uv`` (upstream-canonical)
# fully resolves+syncs deps. Override with ``NRL_RUNNER=python`` to use
# ``${NRL_PYTHON:-/opt/nemo_rl_venv/bin/python}`` directly (useful when
# uv-resolve fails inside a pre-baked container).
NRL_RUNNER=${NRL_RUNNER:-uv}
NRL_PYTHON=${NRL_PYTHON:-/opt/nemo_rl_venv/bin/python}
case "$NRL_RUNNER" in
  uv)
    RUN_PY=(uv run)
    ;;
  python|direct)
    RUN_PY=("$NRL_PYTHON")
    ;;
  *)
    echo "[ERROR] Unknown NRL_RUNNER='$NRL_RUNNER' (expected 'uv' or 'python')" >&2
    exit 1
    ;;
esac
echo "[runner] using NRL_RUNNER=$NRL_RUNNER -> ${RUN_PY[*]}"

# Run the experiment
cd $PROJECT_ROOT
"${RUN_PY[@]}" examples/run_grpo.py \
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
    "$@" \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
"${RUN_PY[@]}" tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Loose correctness gates (sized for H100/H200; relax for weaker GPUs):
# token_mult_prob_error < 1.5 catches a bridge that corrupted weights;
# mean(reward) > 0.0 is a not-NaN sanity check; step-time ceiling is the
# end-to-end performance floor.
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    "${RUN_PY[@]}" tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.5' \
        'mean(data["train/reward"]) > 0.0' \
        'mean(data["timing/train/total_step_time"], 2) < 240'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
