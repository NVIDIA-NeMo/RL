#!/usr/bin/env bash
#SBATCH --account=nemotron_n4_post
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=bird-eval-test
#SBATCH --output=/lustre/fsw/portfolios/nemotron/users/dmisra/Gym/results/slurm-logs/eval-bird-%j.log

set -euo pipefail

# GPUs on one HSG node (GB200, gpu:4); vLLM runs data-parallel across all of them.
NUM_GPUS="${NUM_GPUS:-4}"
# ARM=${1:?arm required: baseline, synthetic, synthetic_sqlite, bird, bird_lr1em6, bird_lr2p5em6, bird_lr1em5, or combined}
# Overridable via `sbatch --export=ALL,MAX_OUTPUT_TOKENS,TEMPERATURE,NUM_REPEATS`
# (see evaluate_all_checkpoints.sh).
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-24000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
NUM_REPEATS="${NUM_REPEATS:-3}"
export NUM_REPEATS
if [[ ! "$NUM_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "num repeats must be a positive integer" >&2
    exit 2
fi
# lightning (default): Nemotron-3.5-Lightning-30B-A3B hybrid Mamba MoE.
# dory: Nemotron-Labs-Dory1.0-4B (single model.safetensors, recurrent loops).
# Set by evaluate_all_checkpoints.sh via `--export`.
MODEL_FAMILY="${MODEL_FAMILY:-lightning}"
case "$MODEL_FAMILY" in
    lightning|dory) ;;
    *) echo "Unknown MODEL_FAMILY: $MODEL_FAMILY (expected lightning or dory)" >&2; exit 2 ;;
esac
# case "$ARM" in
#     baseline|synthetic|synthetic_sqlite|bird|bird_lr1em6|bird_lr2p5em6|bird_lr1em5|combined) ;;
#     *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
# esac
if [[ -n "$MAX_OUTPUT_TOKENS" && ! "$MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "max output tokens must be a positive integer" >&2
    exit 2
fi

# Resolve the physical path: /lustre/fsw/portfolios/nemotron/users/dmisra is a
# symlink, and pyxis's container chdir/mount can fail to resolve it from
# inside the container even though --container-mounts=/lustre/fsw:/lustre/fsw
# is set (see "couldn't chdir to ...: No such file or directory").
PROJECT_ROOT=$(cd /lustre/fsw/portfolios/nemotron/users/dmisra/Gym && pwd -P)
GYM_ROOT="$PROJECT_ROOT"
# Gym commit this script was adapted to (dipendra/birdspark; its CLI and file layout changed from the older Gym the
# original script targeted). Override with EXPECTED_COMMIT=<sha>, or set it empty to skip the check. The commit is
# always printed to driver.log. Note: this checks HEAD only, not uncommitted changes in the checkout.
EXPECTED_COMMIT="${EXPECTED_COMMIT-83c1624caaeb6f4933b9117b81235033b3439a4b}"

# MODEL_ROOT=$(cd /lustre/fsw/portfolios/nemotron/users/dmisra/offline-rl/results/oapl-14000seq-lr5e-6-beta-1-1e3-run2/checkpoints/step_50/hf_export && pwd -P)
# MODEL_ROOT=/lustre/fsw/portfolios/nemotron/users/dmisra/offline-rl/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9  # base model, for comparison
# MODEL_NAME=nemotron-3.5-lightning-oapl-14000seq-lr5e-6-beta-1-1e3-run2-step50

# Overridable via `sbatch --export=ALL,MODEL_ROOT,MODEL_NAME` (see
# evaluate_all_checkpoints.sh) -- defaults below are for a manual single-run edit.
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to a converted HF checkpoint dir, e.g. <nemo-rl-r080>/results/<run>/hf/step_50 (physical /lustre/fs1 path)}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME, e.g. nemotron-3.5-lightning-<run>-step_50}"

# if [[ "$ARM" == baseline ]]; then
#     MODEL_ROOT=/lustre/fsw/portfolios/llmservice/projects/llmservice_sdg_research/users/dnathawani/models/hf_models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-a9904d24
#     MODEL_NAME=nemotron-3.5-lightning-posttrained-baseline
# else
#     MODEL_ROOT="$PROJECT_ROOT/models/sft/nemotron-3.5-lightning-posttrained-$ARM-step-716-hf"
#     MODEL_NAME="nemotron-3.5-lightning-sft-$ARM"
# fi
# docker:// is pulled by enroot at job start, so the compute node needs registry access and
# an aarch64 build of the image. If it doesn't, import it once on a login node
# (enroot import -o <path>.sqsh docker://vllm/vllm-openai:v0.27.1) and export VLLM_IMAGE=<path>.sqsh.
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.27.1}"
# Model-family-specific `vllm serve` flags, spliced into the command below.
# Dory's --hf-overrides JSON goes through an exported env var so the inner
# `bash -lc` (srun forwards the environment) expands it with its quotes intact.
VLLM_PRE_CMD=""
SRUN_CONTAINER_FLAGS=""
case "$MODEL_FAMILY" in
    lightning)
        VLLM_FAMILY_ARGS="--mamba-backend flashinfer --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5"
        ;;
    dory)
        N_RECURRENT_LOOPS="${N_RECURRENT_LOOPS:-2}"
        export HF_OVERRIDES="{\"n_recurrent_loops\": ${N_RECURRENT_LOOPS}, \"recurrent_kv_cache_mode\": \"per_loop\"}"
        VLLM_FAMILY_ARGS='--hf-overrides "$HF_OVERRIDES"'
        # Dory is not in any released vLLM yet (through v0.32.0), so the image is
        # patched at job start by dory_vllm/apply_dory_overlay.py (vendored from the
        # vLLM build Dory was run on; see dory_vllm/README.md). v0.32.0 rather than
        # v0.27.1: Dory's config needs transformers >= 5.16, which v0.27.1 predates.
        # The overlay exits non-zero on a vLLM version mismatch, which fails the job.
        VLLM_IMAGE="${DORY_VLLM_IMAGE:-docker://vllm/vllm-openai:v0.32.0}"
        export DORY_OVERLAY_DIR="$GYM_ROOT/dory_vllm"
        VLLM_PRE_CMD='python3 "$DORY_OVERLAY_DIR/apply_dory_overlay.py"'
        # The overlay writes into the container's vllm package.
        SRUN_CONTAINER_FLAGS="--container-writable"
        ;;
esac
# BIRD_JSONL="$GYM_ROOT/benchmarks/birdbench/data/birdbench_dev_verified_100_v2.jsonl"
# Gym commit d42b3c36 renamed these: prompts/default.yaml -> prompts/default_<dialect>.yaml and
# data/birdbench_benchmark.jsonl -> data/birdbench_<dialect>_<split>.jsonl (written by
# `python -m benchmarks.birdbench.prepare`; BIRD dev, SQLite). Override with env vars if yours differ.
BIRD_JSONL="${BIRD_JSONL:-$GYM_ROOT/benchmarks/birdbench/data/birdbench_sqlite_dev.jsonl}"
PROMPT_CONFIG="${PROMPT_CONFIG:-$GYM_ROOT/benchmarks/birdbench/prompts/default_sqlite.yaml}"
if [[ -n "$MAX_OUTPUT_TOKENS" ]]; then
    RESULT_VARIANT="maxout-$MAX_OUTPUT_TOKENS"
else
    RESULT_VARIANT=maxout-unset
fi
# Temperature and repeats are always part of the directory name, e.g.
# maxout-16000-temp-0-num-repeats-1, maxout-24000-temp-1-num-repeats-3,
# maxout-24000-temp-0p6-num-repeats-1.
RESULT_VARIANT="$RESULT_VARIANT-temp-$(printf '%g' "$TEMPERATURE" | tr . p)-num-repeats-$NUM_REPEATS"
# Dory only: also encode the recurrent loop count (mirrors evaluate_all_checkpoints.sh).
if [[ "$MODEL_FAMILY" == dory ]]; then
    RESULT_VARIANT="$RESULT_VARIANT-loops-$N_RECURRENT_LOOPS"
fi
RUN_ROOT="$PROJECT_ROOT/results/$RESULT_VARIANT/$MODEL_NAME/job-${SLURM_JOB_ID}"
VLLM_LOG="$RUN_ROOT/vllm.log"
SMOKE_JSONL="$RUN_ROOT/smoke.jsonl"
FULL_JSONL="$RUN_ROOT/full.jsonl"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/driver.log") 2>&1

cd "$GYM_ROOT"
# $HOME/.local/bin isn't reliably visible on compute nodes; uv is copied onto
# shared storage instead (see driver.log: "uv: command not found" from the
# resources_server/agent/policy_model subprocesses, which inherit this PATH
# via run_command()'s environ.copy()).
export PATH="/lustre/fsw/portfolios/nemotron/users/dmisra/bin:$PATH"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export NLTK_DATA="$PROJECT_ROOT/cache/nltk"
export PYTHONPATH="$GYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
set -a
source "$PROJECT_ROOT/.env"
set +a

cleanup() {
    status=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "${VLLM_STEP_PID:-}" ]]; then
        kill "$VLLM_STEP_PID" 2>/dev/null || true
        wait "$VLLM_STEP_PID" 2>/dev/null || true
    fi
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "exit_status=$status"
    exit "$status"
}
trap cleanup EXIT INT TERM

# `test` under `set -e` exits silently; `require` says which check failed (stderr goes to driver.log too).
require() {
    if ! test "$@"; then
        echo "PREFLIGHT CHECK FAILED: test $*" >&2
        exit 1
    fi
}

echo "started_at=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "slurm_job_id=$SLURM_JOB_ID"
echo "slurm_account=nemotron_n4_post"
echo "slurm_partition=batch"
#echo "arm=$ARM"
echo "gym_commit=$(git rev-parse HEAD)"
echo "model_name=$MODEL_NAME"
echo "model_root=$MODEL_ROOT"
echo "num_gpus=$NUM_GPUS"
echo "model_family=$MODEL_FAMILY"
echo "vllm_family_args=$VLLM_FAMILY_ARGS"
echo "hf_overrides=${HF_OVERRIDES:-unset}"
echo "dory_overlay_dir=${DORY_OVERLAY_DIR:-unset}"
echo "vllm_image=$VLLM_IMAGE"
echo "client_max_output_tokens=${MAX_OUTPUT_TOKENS:-unset}"
echo "temperature=$TEMPERATURE"
echo "num_repeats=$NUM_REPEATS"
echo "result_variant=$RESULT_VARIANT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ -n "$EXPECTED_COMMIT" ]]; then
    require "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
fi
require -x .venv/bin/gym
require -f "$BIRD_JSONL"
# Derived from BIRD_JSONL itself instead of hardcoded 1534, so this script
# works against any subset (e.g. the 100-example overfitting-test file)
# without every downstream row-count check silently failing.
NUM_QUESTIONS=$(wc -l < "$BIRD_JSONL")
export NUM_QUESTIONS
require "$NUM_QUESTIONS" -gt 0
# sha256sum --check "$PROJECT_ROOT/manifests/bird-pr2898/birdbench_benchmark.jsonl.sha256"
require -f "$PROMPT_CONFIG"
require -f "$MODEL_ROOT/config.json"
require -f "$MODEL_ROOT/tokenizer.json"
if [[ "$MODEL_FAMILY" == dory ]]; then
    require -f "$MODEL_ROOT/model.safetensors"
    require -f "$DORY_OVERLAY_DIR/apply_dory_overlay.py"
else
    require -f "$MODEL_ROOT/model.safetensors.index.json"
    require "$(find -L "$MODEL_ROOT" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)" -eq 14
fi

# Mount the logical (/lustre/fsw) and physical (/lustre/fs1) trees, plus /scratch/fsw
# where it exists. PROJECT_ROOT and MODEL_ROOT are physical (/lustre/fs1/...) paths on
# HSG, so the physical tree has to be mounted too. Mounting a path that doesn't exist
# on the host makes enroot-mount fail before the job even starts, so each one is only
# requested when present on the node running this script.
CONTAINER_MOUNTS="/lustre/fsw:/lustre/fsw"
for extra in /lustre/fs1 /scratch/fsw; do
    if [ -d "$extra" ]; then
        CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${extra}:${extra}"
    fi
done

srun \
    --nodes=1 \
    --ntasks=1 \
    --gpus="$NUM_GPUS" \
    --container-image="$VLLM_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="$GYM_ROOT" \
    --no-container-mount-home \
    $SRUN_CONTAINER_FLAGS \
    bash -lc "
        set -euo pipefail
        $VLLM_PRE_CMD
        exec vllm serve '$MODEL_ROOT' \\
            --served-model-name '$MODEL_NAME' \\
            --host 0.0.0.0 \\
            --port 8000 \\
            --data-parallel-size $NUM_GPUS \\
            --data-parallel-size-local $NUM_GPUS \\
            --data-parallel-backend mp \\
            --api-server-count 1 \\
            --trust-remote-code \\
            --gpu-memory-utilization 0.9 \\
            --max-model-len 262144 \\
            --max-num-seqs 128 \\
            --enable-prefix-caching \\
            --async-scheduling \\
            $VLLM_FAMILY_ARGS \\
            --reasoning-parser nemotron_v3 \\
            --tool-call-parser qwen3_coder \\
            --enable-auto-tool-choice
    " >"$VLLM_LOG" 2>&1 &
VLLM_STEP_PID=$!

for attempt in $(seq 1 240); do
    if curl --fail --silent --show-error http://127.0.0.1:8000/v1/models >/dev/null; then
        echo "vllm_ready_after_attempt=$attempt"
        break
    fi
    if ! kill -0 "$VLLM_STEP_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready"
        tail -n 240 "$VLLM_LOG"
        exit 1
    fi
    if [[ "$attempt" -eq 240 ]]; then
        echo "Timed out waiting for vLLM"
        tail -n 240 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
done

# Same invocation as benchmarks/birdbench/eval.sh. `gym eval run` without --no-serve is end-to-end: it starts the
# Gym servers itself and reads the benchmark's prepared dataset from benchmarks/birdbench/config.yaml
# (data/birdbench_sqlite_dev.jsonl + prompts/default_sqlite.yaml, the BIRD dev set). In that mode --input is rejected
# (nemo_gym/rollout_collection.py:1263), and --agent / a dataset-overriding --config are redundant.
COMMON_ARGS=(
    --benchmark birdbench
    --model-type vllm_model
    --model "$MODEL_NAME"
    --model-url http://127.0.0.1:8000/v1
    --model-api-key dummy
    --split benchmark
    --temperature "$TEMPERATURE"
    ++port_range_low=63000
    ++port_range_high=64000
    ++overwrite_metrics_conflicts=true
)
if [[ -n "$MAX_OUTPUT_TOKENS" ]]; then
    COMMON_ARGS+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")
fi

echo "Starting three-question smoke evaluation"
.venv/bin/gym eval run \
    "${COMMON_ARGS[@]}" \
    --output "$SMOKE_JSONL" \
    --limit 3 \
    --num-repeats 1 \
    --concurrency 3

require "$(wc -l < "$SMOKE_JSONL")" -eq 3
.venv/bin/python - "$SMOKE_JSONL" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
assert len(rows) == 3
incomplete = 0
for index, row in enumerate(rows):
    response = row.get("response") or {}
    assert response.get("error") is None, (index, response.get("error"))
    status = response.get("status")
    assert status in {"completed", "incomplete"}, (index, status)
    incomplete += status == "incomplete"
    assert response.get("output"), (index, "empty response output")
print(f"smoke_validation=passed incomplete={incomplete}")
PY

echo "Starting full ${NUM_QUESTIONS}-question BIRD evaluation (${NUM_REPEATS} repeats each)"
.venv/bin/gym eval run \
    "${COMMON_ARGS[@]}" \
    --output "$FULL_JSONL" \
    --num-repeats "$NUM_REPEATS" \
    --concurrency 64

# One output row per (question, repeat).
EXPECTED_ROWS=$((NUM_QUESTIONS * NUM_REPEATS))
export EXPECTED_ROWS
require "$(wc -l < "$FULL_JSONL")" -eq "$EXPECTED_ROWS"
.venv/bin/python - "$FULL_JSONL" <<'PY'
import json
import os
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
assert len(rows) == int(os.environ["EXPECTED_ROWS"])
failed = [row for row in rows if (row.get("response") or {}).get("error") is not None]
print(f"full_response_errors={len(failed)}")
assert not failed
PY

.venv/bin/python - "$RUN_ROOT/full_aggregate_metrics.json" "$RUN_ROOT/metrics-summary.json" <<'PY'
import json
import os
import sys

source, destination = sys.argv[1:]
k = int(os.environ["NUM_REPEATS"])
entries = json.load(open(source))
assert len(entries) == 1
metrics = entries[0]["agent_metrics"]
keys = [
    "mean/input_tokens",
    "mean/output_tokens",
    f"pass@1[avg-of-{k}]/accuracy",
    "pass@1/accuracy",
    f"simple/pass@1[avg-of-{k}]/accuracy",
    f"moderate/pass@1[avg-of-{k}]/accuracy",
    f"challenging/pass@1[avg-of-{k}]/accuracy",
]
summary = {key: metrics[key] for key in keys}
# Across-repeat stats (std_dev/std_err of the per-repeat accuracy) only exist for k > 1.
summary.update({key: value for key, value in metrics.items() if key.startswith(f"pass@1[avg-of-{k}]/accuracy/")})
summary.update({key: value for key, value in metrics.items() if "no_sql" in key or "no_answer" in key})
with open(destination, "w") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "full_rollout_rows=${EXPECTED_ROWS}"
echo "full_output=$FULL_JSONL"
echo "metrics_summary=$RUN_ROOT/metrics-summary.json"
find "$RUN_ROOT" -maxdepth 1 -type f -printf '%f|%s\n' | sort
