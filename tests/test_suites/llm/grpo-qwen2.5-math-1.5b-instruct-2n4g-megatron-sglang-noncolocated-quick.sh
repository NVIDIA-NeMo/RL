#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=1
NUM_MINUTES=60
# ===== END CONFIG =====

cd "$PROJECT_ROOT"

REPO_TOP=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [[ "$REPO_TOP" == "$PROJECT_ROOT" ]]; then
    ROOT_COMMIT=$(git rev-parse HEAD)
    PINNED_GYM_COMMIT=$(git rev-parse HEAD:3rdparty/Gym-workspace/Gym)
    CHECKED_OUT_GYM_COMMIT=$(git -C 3rdparty/Gym-workspace/Gym rev-parse HEAD)
    if [[ "$CHECKED_OUT_GYM_COMMIT" != "$PINNED_GYM_COMMIT" ]]; then
        echo "[ERROR] NeMo-Gym checkout $CHECKED_OUT_GYM_COMMIT does not match the pinned commit $PINNED_GYM_COMMIT"
        exit 1
    fi
    if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
        echo "[ERROR] The reproducer requires a clean checkout; tracked files are modified"
        exit 1
    fi
else
    # tools/launch runs a git-tracked code snapshot without its own .git
    # directory. Use the immutable image commit and the source checkout's
    # gitlink for provenance in that case.
    if [[ -n "$REPO_TOP" ]]; then
        ROOT_COMMIT=$(git -C "$REPO_TOP" rev-parse HEAD)
        PINNED_GYM_COMMIT=$(git -C "$REPO_TOP" rev-parse HEAD:3rdparty/Gym-workspace/Gym)
        if [[ -n "$(git -C "$REPO_TOP" status --porcelain --untracked-files=no)" ]]; then
            echo "[ERROR] The source checkout used for the snapshot has modified tracked files"
            exit 1
        fi
    else
        ROOT_COMMIT=${NEMO_RL_COMMIT:-<unknown>}
        PINNED_GYM_COMMIT=${NEMO_GYM_COMMIT:-<unknown>}
    fi
    CHECKED_OUT_GYM_COMMIT=$PINNED_GYM_COMMIT
fi
if [[ "${NEMO_RL_COMMIT:-<unknown>}" != "$ROOT_COMMIT" ]]; then
    echo "[ERROR] Container commit ${NEMO_RL_COMMIT:-<unknown>} does not match checkout $ROOT_COMMIT"
    exit 1
fi
if [[ "$PINNED_GYM_COMMIT" == "<unknown>" ]]; then
    echo "[ERROR] Set NEMO_GYM_COMMIT when running a source snapshot without git metadata"
    exit 1
fi
if [[ -z "${NRL_MEGATRON_CHECKPOINT_DIR:-}" ]]; then
    echo "[ERROR] NRL_MEGATRON_CHECKPOINT_DIR must name a shared, writable conversion-cache directory"
    exit 1
fi
if [[ ! "${NRL_MODEL_REVISION:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[ERROR] NRL_MODEL_REVISION must be an immutable 40-character model commit"
    exit 1
fi
if [[ -z "${NRL_MODEL_PATH:-}" || ! -s "$NRL_MODEL_PATH/config.json" ]]; then
    echo "[ERROR] NRL_MODEL_PATH must contain the immutable model snapshot and config.json"
    exit 1
fi
if [[ "${NRL_IMAGE_REF:-}" != *@sha256:* ]]; then
    echo "[ERROR] NRL_IMAGE_REF must include an immutable sha256 image digest"
    exit 1
fi
mkdir -p "$NRL_MEGATRON_CHECKPOINT_DIR"
if [[ ! -d "$NRL_MEGATRON_CHECKPOINT_DIR" || ! -w "$NRL_MEGATRON_CHECKPOINT_DIR" ]]; then
    echo "[ERROR] NRL_MEGATRON_CHECKPOINT_DIR is not a writable directory"
    exit 1
fi

RUN_ID=${NRL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] NRL_RUN_ID may contain only letters, digits, '.', '_', and '-'"
    exit 1
fi
RUN_ROOT=$EXP_DIR/runs
EXP_DIR=$RUN_ROOT/$RUN_ID
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
PREFLIGHT_JSON=$EXP_DIR/preflight.json
VALIDATION_JSON=$EXP_DIR/validation.json
PROVENANCE_FILE=$EXP_DIR/provenance.txt
if [[ -e "$EXP_DIR" ]]; then
    echo "[ERROR] Refusing to reuse evidence directory $EXP_DIR"
    exit 1
fi
mkdir -p "$LOG_DIR" "$CKPT_DIR"

{
    printf 'nemo_rl_commit=%s\n' "$ROOT_COMMIT"
    printf 'nemo_gym_commit=%s\n' "$CHECKED_OUT_GYM_COMMIT"
    printf 'image_commit=%s\n' "$NEMO_RL_COMMIT"
    printf 'image_ref=%s\n' "$NRL_IMAGE_REF"
    printf 'model_revision=%s\n' "$NRL_MODEL_REVISION"
    printf 'model_path=%s\n' "$NRL_MODEL_PATH"
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'command='
    printf '%q ' "$0" "$@"
    printf '\n'
} > "$PROVENANCE_FILE"

echo "[INFO] Writing fresh-run evidence to $EXP_DIR"
uv run --frozen python tests/sglang_refit_checks.py preflight \
    --num-nodes "$NUM_NODES" \
    --gpus-per-node "$GPUS_PER_NODE" \
    --output "$PREFLIGHT_JSON"

uv run --frozen examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run --frozen tests/json_dump_tb_logs.py \
    "$LOG_DIR" \
    --output_path "$JSON_METRICS"
uv run --frozen python tests/sglang_refit_checks.py validate \
    --run-log "$RUN_LOG" \
    --metrics-json "$JSON_METRICS" \
    --max-step "$MAX_STEPS" \
    --world-size 5 \
    --engines 4 \
    --min-refit-successes 2 \
    --output "$VALIDATION_JSON"
