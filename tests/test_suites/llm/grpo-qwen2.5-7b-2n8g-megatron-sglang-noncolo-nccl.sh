#!/bin/bash
# E2E smoke test for the slime-style SGLang non-colocated NCCL bridge
# refit on the peng-grpo-sglang-no-colocate-v1 branch.
#
# Layout:
#   - 2 physical nodes (1 train + 1 infer), 8 GPUs each
#   - Training: Megatron, TP=2, DP=4 on 1 node
#   - Inference: SGLang, TP=8 single engine on 1 dedicated node
#   - Refit: per-bucket dist.broadcast on a bridge ProcessGroup spanning
#     train rank 0 + every SGLang TP rank (slime topology). Per-bucket
#     metadata POSTed to the SGLang HTTP server in parallel via threads.
#
# This is the LIGHTEST e2e test that meaningfully exercises the bridge:
#   - 2 nodes is the minimum for non-colocated;
#   - Qwen2.5-7B (~14 GiB BF16) is large enough to span multiple
#     1 GiB NCCL-bucket flushes per refit, validating bucket boundary
#     handling + flush_cache=True on the last bucket;
#   - 30 steps * (refit + 128-rollout generate + train) is enough to
#     surface NCCL hangs / deadlocks / progressive memory leaks.
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

# ----------------------------------------------------------------------
# NCCL-bridge env vars. The training-side grpo.py routing reads:
#   - NRL_SGLANG_BRIDGE_PORT: TCPStore port for the bridge rendezvous.
#   - NRL_SGLANG_BRIDGE_GROUP_NAME: bridge ProcessGroup name (defaults
#     to "nrl-sglang-bridge" on both sides).
# The training-side broadcast helper reads:
#   - NRL_REFIT_BUCKET_BYTES: bytes per NCCL bucket (default 1 GiB).
#   - NRL_SGLANG_NCCL_HTTP_TIMEOUT_S: per-bucket metadata HTTP timeout
#     (default 7200s).
# The SGLang side reads:
#   - NRL_SGLANG_GENERATE_TIMEOUT_S: aiohttp timeout for /generate
#     (default 1800s; bumped from upstream 300s to survive the slow
#     first-post-refit /generate that hits cuda graph rebuild etc.).
#
# NRL_SGLANG_NONCOLO_TRANSPORT is INFORMATIONAL on this branch (the
# code unconditionally takes the NCCL path when
# isinstance(policy_generation, SGLangGeneration) and not colocated);
# we still set it so the env reads obviously document the path.
# ----------------------------------------------------------------------
export NRL_SGLANG_NONCOLO_TRANSPORT=${NRL_SGLANG_NONCOLO_TRANSPORT:-nccl}
export NRL_SGLANG_BRIDGE_PORT=${NRL_SGLANG_BRIDGE_PORT:-29500}
export NRL_SGLANG_BRIDGE_GROUP_NAME=${NRL_SGLANG_BRIDGE_GROUP_NAME:-nrl-sglang-bridge}
export NRL_REFIT_BUCKET_BYTES=${NRL_REFIT_BUCKET_BYTES:-1073741824}
export NRL_SGLANG_NCCL_HTTP_TIMEOUT_S=${NRL_SGLANG_NCCL_HTTP_TIMEOUT_S:-7200}
export NRL_SGLANG_GENERATE_TIMEOUT_S=${NRL_SGLANG_GENERATE_TIMEOUT_S:-1800}

# Lift NCCL collective watchdog timeout from 600s to 9000s. The bridge
# refit shouldn't go anywhere near 600s in normal operation, but a slow
# first-step warmup (NCCL channel allocation etc.) is more comfortable
# with the lifted ceiling.
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-9000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-9000}

# Pick the python invocation method.
#
# Default: ``uv run`` (upstream-canonical). Best when running on a
# fresh checkout with internet — uv resolves & syncs deps before
# launching.
#
# Override: ``NRL_RUNNER=python`` invokes the in-container python
# directly (``${NRL_PYTHON:-/opt/nemo_rl_venv/bin/python}``), bypassing
# ``uv run`` entirely. Use this when:
#   - running inside a pre-baked container (deps are already installed
#     and ``uv sync`` would just spin / fail on offline nodes), or
#   - the project ``pyproject.toml`` pins a Python version that uv
#     cannot satisfy (we hit ``requires-python = ">=3.13.13"`` which
#     uv refuses), or
#   - megatron-core's CACHED_DEPENDENCIES drift from its submodule
#     pyproject blocks ``uv sync``.
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

# Only run metrics if the target step is reached.
# Bounds chosen to be loose: this is a CORRECTNESS smoke test, not a
# perf gate. The values are sized for an H100 H200 run; relax for
# weaker GPUs.
#
#   - token_mult_prob_error < 1.5: SGLang vs Megatron logprob mismatch
#     should stay close to 1 across refits. >1.5 indicates the NCCL
#     broadcast corrupted weights or skipped a tensor. (vLLM equivalent
#     suites use 1.1 but they generate on the same GPUs they trained
#     on; here the cross-node bridge introduces a tiny extra epsilon
#     so we widen the bound.)
#   - mean(reward) > 0.0: just a "training is making progress and
#     not NaN'ing" sanity check at 30 steps.
#   - mean(total_step_time) < 240s: end-to-end step ceiling. Bridge
#     init runs ONCE (~few s), refit per step is ~10-30 s for 7B,
#     128-rollout generate is ~30-90 s, train is ~30 s.
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    "${RUN_PY[@]}" tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.5' \
        'mean(data["train/reward"]) > 0.0' \
        'mean(data["timing/train/total_step_time"], 2) < 240'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
