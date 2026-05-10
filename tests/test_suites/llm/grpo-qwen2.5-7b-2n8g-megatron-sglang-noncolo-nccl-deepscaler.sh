#!/bin/bash
# E2E meaningful GRPO run for Qwen2.5-7B-Instruct on DeepScaler train +
# AIME2024 validation, exercising the SGLang non-colocated NCCL bridge.
# Sibling of grpo-qwen2.5-7b-2n8g-megatron-sglang-noncolo-nccl.sh (the
# 30-step smoke test); identical bridge env wiring, but longer runtime,
# real validation curve, and verify off by default for speed.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%s)}}"
EXP_DIR="$EXP_DIR/run-$RUN_TAG"
LOG_DIR="$EXP_DIR/logs"
CKPT_DIR="$EXP_DIR/ckpts"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
mkdir -p "$EXP_DIR" "$LOG_DIR" "$CKPT_DIR"
echo "[runner] run dir = $EXP_DIR"

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=200
MAX_STEPS=200
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=480
# ===== END CONFIG =====

exit_if_max_steps_reached

export NRL_SGLANG_NONCOLO_TRANSPORT=${NRL_SGLANG_NONCOLO_TRANSPORT:-nccl}
export NRL_SGLANG_BRIDGE_PORT=${NRL_SGLANG_BRIDGE_PORT:-29500}
export NRL_SGLANG_BRIDGE_GROUP_NAME=${NRL_SGLANG_BRIDGE_GROUP_NAME:-nrl-sglang-bridge}
export NRL_REFIT_BUCKET_BYTES=${NRL_REFIT_BUCKET_BYTES:-1073741824}
export NRL_SGLANG_NCCL_HTTP_TIMEOUT_S=${NRL_SGLANG_NCCL_HTTP_TIMEOUT_S:-7200}
export NRL_SGLANG_GENERATE_TIMEOUT_S=${NRL_SGLANG_GENERATE_TIMEOUT_S:-1800}

export NCCL_TIMEOUT=${NCCL_TIMEOUT:-9000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-9000}

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

"${RUN_PY[@]}" tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Loose end-of-run gates: bridge integrity (token_mult_prob_error well
# under control), real reward signal (DeepScaler verifier should give
# meaningful nonzero rewards once the policy improves), and bounded
# step time (catches any deadlock regression).
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    "${RUN_PY[@]}" tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.5' \
        'mean(data["train/reward"]) > 0.0' \
        'mean(data["timing/train/total_step_time"], 2) < 240'
fi
