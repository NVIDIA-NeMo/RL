#!/bin/bash
# Launch tau-bench GRPO training with local vLLM inference endpoints.
#
# Submits two or three SLURM jobs:
#   1. NeMo-RL training     (NUM_ACTOR_NODES nodes)  <- submitted FIRST
#   2. vLLM user simulator  (1 node, starts when training job begins running)
#   3. vLLM judge           (1 node, only when JUDGE_PORT != USER_PORT, same dependency)
#
# Submitting training first means the vLLM nodes are only allocated once the
# policy nodes have left the queue -- no wasted vLLM node time while waiting
# for a large training allocation.  The training job's command embeds the
# wait-for-coord-files loop, so it sits idle for the few minutes it takes
# vLLM to load the model, then starts training immediately.
#
# When JUDGE_PORT == USER_PORT the judge reuses the user simulator server
# (same model, same endpoint), saving a full node.
#
# Run from the root of the NeMo-RL repo.
#
# All configuration variables can be overridden by setting them as environment
# variables before invoking this script, e.g.:
#
#   ACCOUNT=my_account NEMO_CONTAINER=/path/to/image.squashfs ./examples/tau_bench/run_tau_local.sh

set -eoux pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

########################################################
# Configuration -- override any of these via environment variables
########################################################
NUM_ACTOR_NODES=${NUM_ACTOR_NODES:-16}

# Root of the NeMo-RL repo on a shared filesystem accessible from all SLURM
# nodes. Auto-detected from the script location (two directories up from
# examples/tau_bench/).
WORKDIR=${WORKDIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}

# HuggingFace model cache. Mounted into the container as /mnt/cache.
HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}

# Container image for NeMo-RL training.
NEMO_CONTAINER=${NEMO_CONTAINER:?Set NEMO_CONTAINER to the path of your NeMo-RL container image (e.g. /path/to/nemo-rl.squashfs)}

# Container image for vLLM servers. Defaults to NEMO_CONTAINER if vLLM is
# installed in the NeMo-RL image; set to a dedicated vLLM image if not.
VLLM_CONTAINER=${VLLM_CONTAINER:-$NEMO_CONTAINER}

# SLURM account and partitions.
ACCOUNT=${ACCOUNT:?Set ACCOUNT to your SLURM account name}
TRAIN_PARTITION=${TRAIN_PARTITION:-batch}        # long-running training job
VLLM_PARTITION=${VLLM_PARTITION:-batch}   # vLLM servers

# User simulator.
# TP=1 DP=8 -> 8 independent replicas, one per GPU. Better than TP=8 for
# inference because DP has no inter-GPU communication overhead.
USER_MODEL=${USER_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
USER_PORT=${USER_PORT:-8100}
USER_TP=${USER_TP:-1}
USER_DP=${USER_DP:-8}
USER_GPU_MEM=${USER_GPU_MEM:-0.90}
USER_MAX_LEN=${USER_MAX_LEN:-8192}       # must fit full conversation history, not just one turn

# Judge. Set JUDGE_MODEL to empty string to disable judge scoring entirely.
#
# Same server as user (default, saves one node):
#   JUDGE_MODEL == USER_MODEL, JUDGE_PORT == USER_PORT -> no separate job submitted.
#
# Separate server (better signal, costs one extra node):
#   Set JUDGE_PORT to a different value (e.g. 8101) and adjust TP/DP.
#   Example for 70B: JUDGE_MODEL="meta-llama/Llama-3.3-70B-Instruct" JUDGE_TP=4 JUDGE_DP=2
JUDGE_MODEL=${JUDGE_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
JUDGE_PORT=${JUDGE_PORT:-$USER_PORT}     # same as USER_PORT -> reuse the user simulator server
JUDGE_TP=${JUDGE_TP:-$USER_TP}
JUDGE_DP=${JUDGE_DP:-$USER_DP}
JUDGE_GPU_MEM=${JUDGE_GPU_MEM:-$USER_GPU_MEM}
JUDGE_MAX_LEN=${JUDGE_MAX_LEN:-$USER_MAX_LEN}

# Coordination files: written by each vLLM job with the IP of its node.
USER_COORD_FILE="${USER_COORD_FILE:-$WORKDIR/.vllm_user_host}"
JUDGE_COORD_FILE="${JUDGE_COORD_FILE:-$WORKDIR/.vllm_judge_host}"

# Training config
CONFIG=${CONFIG:-"examples/configs/recipes/llm/grpo_tau_bench_local.yaml"}
########################################################

MOUNTS="$WORKDIR:$WORKDIR,$WORKDIR:/opt/nemo-rl,$HF_HOME:/mnt/cache"

# Clean up stale coordination files from previous runs before submitting anything.
# (vLLM jobs won't start until the training job runs, so there is no race here.)
rm -f "$USER_COORD_FILE" "$JUDGE_COORD_FILE"

########################################################
# Determine judge configuration (known at submission time)
########################################################
SEPARATE_JUDGE=false
if [[ -n "$JUDGE_MODEL" && "$JUDGE_PORT" != "$USER_PORT" ]]; then
    SEPARATE_JUDGE=true
fi

# Model-name override is fixed at submission time; URL override is computed
# at runtime (inside the training job) once we know the vLLM host IP.
if [[ -n "$JUDGE_MODEL" ]]; then
    JUDGE_MODEL_OVERRIDE="env.tau_bench.judge_model=openai/${JUDGE_MODEL}"
else
    JUDGE_MODEL_OVERRIDE="env.tau_bench.judge_model=null"
fi

########################################################
# Build the training command with the vLLM wait loop embedded.
#
# ENDCMD is unquoted so the outer shell expands variables at capture time:
#   $var   -- expand NOW in run_tau_local.sh  (coord file paths, ports, config, ...)
#   \$var  -- expand LATER inside the running training job  (host IPs, loop vars)
#
# Do NOT use \\ line continuations inside the heredoc: SLURM inserts trailing
# spaces when passing multiline env vars, turning \<newline> into \ <newline>
# (escaped space), which becomes a spurious ' ' argument to the command.
########################################################
COMMAND=$(cat <<ENDCMD
set -eoux pipefail
export HF_HOME=/mnt/cache

# Wait for vLLM servers to write their coordination files.
# Allow up to 2 hours so jobs queued for a long time are still handled correctly.
echo "[INFO] Waiting for vLLM servers to come up..."
ELAPSED=0
WAIT_TIMEOUT=7200
while true; do
    USER_READY=false
    JUDGE_READY=false
    [[ -f "${USER_COORD_FILE}" ]] && USER_READY=true
    [[ "${SEPARATE_JUDGE}" == "false" || -f "${JUDGE_COORD_FILE}" ]] && JUDGE_READY=true
    if \$USER_READY && \$JUDGE_READY; then break; fi
    if [[ \$ELAPSED -ge \$WAIT_TIMEOUT ]]; then
        echo "[ERROR] Timed out waiting for vLLM coordination files after \${WAIT_TIMEOUT}s" >&2
        exit 1
    fi
    echo "[INFO] Still waiting (\${ELAPSED}s)... user_ready=\$USER_READY judge_ready=\$JUDGE_READY"
    sleep 10
    ELAPSED=\$((ELAPSED + 10))
done

USER_HOST=\$(cat "${USER_COORD_FILE}")
echo "[INFO] User simulator at \${USER_HOST}:${USER_PORT}"

JUDGE_HOST=""
JUDGE_URL_OVERRIDE=""
if [[ -n "${JUDGE_MODEL}" ]]; then
    if [[ "${SEPARATE_JUDGE}" == "true" ]]; then
        JUDGE_HOST=\$(cat "${JUDGE_COORD_FILE}")
    else
        JUDGE_HOST=\$USER_HOST
    fi
    echo "[INFO] Judge at \${JUDGE_HOST}:${JUDGE_PORT}"
    JUDGE_URL_OVERRIDE="env.tau_bench.judge_base_url=http://\${JUDGE_HOST}:${JUDGE_PORT}/v1"
fi

# Verify the servers are actually responding before starting training.
echo "[INFO] Verifying user simulator health (may take several minutes while model loads)..."
curl --retry 60 --retry-delay 10 --retry-connrefused --silent --fail "http://\${USER_HOST}:${USER_PORT}/v1/models" > /dev/null
echo "[INFO] User simulator is healthy."

if [[ "${SEPARATE_JUDGE}" == "true" ]]; then
    echo "[INFO] Verifying judge health (may take several minutes while model loads)..."
    curl --retry 60 --retry-delay 10 --retry-connrefused --silent --fail "http://\${JUDGE_HOST}:${JUDGE_PORT}/v1/models" > /dev/null
    echo "[INFO] Judge is healthy."
fi

export USER_SERVER_HOST=\${USER_HOST}
export JUDGE_SERVER_HOST=\${JUDGE_HOST:-}

uv run examples/run_grpo.py --config ${CONFIG} env.tau_bench.user_base_url=http://\${USER_HOST}:${USER_PORT}/v1 ${JUDGE_MODEL_OVERRIDE} \${JUDGE_URL_OVERRIDE} cluster.num_nodes=${NUM_ACTOR_NODES}
ENDCMD
)

########################################################
# Job 1: NeMo-RL training (submitted first to get its job ID)
########################################################
TRAIN_JOB_ID=$(
    COMMAND=$COMMAND \
    CONTAINER=$NEMO_CONTAINER \
    MOUNTS=$MOUNTS \
    sbatch \
        --nodes=$NUM_ACTOR_NODES \
        --account=$ACCOUNT \
        --job-name=${ACCOUNT}-llm:tau-local \
        --partition=$TRAIN_PARTITION \
        --time=3:30:0 \
        --gres=gpu:8 \
        --parsable \
        "$WORKDIR/ray.sub"
)
echo "[INFO] Training job: $TRAIN_JOB_ID"

########################################################
# Job 2: vLLM user simulator
# --dependency=after:<training_job_id> releases this job the moment the
# training job transitions from PENDING to RUNNING.
########################################################
USER_JOB_ID=$(
    MODEL=$USER_MODEL \
    PORT=$USER_PORT \
    TENSOR_PARALLEL=$USER_TP \
    DATA_PARALLEL=$USER_DP \
    GPU_MEMORY_UTIL=$USER_GPU_MEM \
    MAX_MODEL_LEN=$USER_MAX_LEN \
    COORD_FILE=$USER_COORD_FILE \
    CONTAINER=$VLLM_CONTAINER \
    MOUNTS=$MOUNTS \
    sbatch \
        --nodes=1 \
        --account=$ACCOUNT \
        --job-name=${ACCOUNT}-llm:vllm-user \
        --partition=$VLLM_PARTITION \
        --time=4:00:0 \
        --gres=gpu:8 \
        --dependency=after:${TRAIN_JOB_ID} \
        --parsable \
        "$SCRIPT_DIR/vllm.sub"
)
echo "[INFO] vLLM user simulator job: $USER_JOB_ID (released when training job $TRAIN_JOB_ID starts running)"

########################################################
# Job 3: vLLM judge (skipped when reusing the user simulator server)
########################################################
if [[ -n "$JUDGE_MODEL" && "$JUDGE_PORT" != "$USER_PORT" ]]; then
    JUDGE_JOB_ID=$(
        MODEL=$JUDGE_MODEL \
        PORT=$JUDGE_PORT \
        TENSOR_PARALLEL=$JUDGE_TP \
        DATA_PARALLEL=$JUDGE_DP \
        GPU_MEMORY_UTIL=$JUDGE_GPU_MEM \
        MAX_MODEL_LEN=$JUDGE_MAX_LEN \
        COORD_FILE=$JUDGE_COORD_FILE \
        CONTAINER=$VLLM_CONTAINER \
        MOUNTS=$MOUNTS \
        sbatch \
            --nodes=1 \
            --account=$ACCOUNT \
            --job-name=${ACCOUNT}-llm:vllm-judge \
            --partition=$VLLM_PARTITION \
            --time=4:00:0 \
            --gres=gpu:8 \
            --dependency=after:${TRAIN_JOB_ID} \
            --parsable \
            "$SCRIPT_DIR/vllm.sub"
    )
    echo "[INFO] vLLM judge job: $JUDGE_JOB_ID (released when training job $TRAIN_JOB_ID starts running)"
elif [[ -n "$JUDGE_MODEL" ]]; then
    echo "[INFO] Judge reusing user simulator server (same port $USER_PORT)"
else
    echo "[INFO] Judge disabled (JUDGE_MODEL is empty)"
fi

echo ""
echo "[INFO] All jobs submitted. Training will wait for vLLM servers before starting."
echo "[INFO] Training logs: ${TRAIN_JOB_ID}-logs/ray-driver.log"
echo ""
echo "To cancel all jobs:"
if [[ -n "$JUDGE_MODEL" && "$JUDGE_PORT" != "$USER_PORT" ]]; then
    echo "  scancel $TRAIN_JOB_ID $USER_JOB_ID $JUDGE_JOB_ID"
else
    echo "  scancel $TRAIN_JOB_ID $USER_JOB_ID"
fi
