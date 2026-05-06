#!/bin/bash
# Cluster auto-detect and shared paths for perf_test submissions.
# Sourced by exp_*.sh scripts.

detect_gpus_per_node() {
    local partition="${1:-batch}"
    local gres_gpus
    gres_gpus=$(sinfo -p "$partition" -h -o "%G" 2>/dev/null | grep -oP 'gpu:\d+' | grep -oP '\d+' | head -1 || true)
    if [[ -n "$gres_gpus" && "$gres_gpus" -gt 0 ]]; then
        echo "$gres_gpus"
    else
        echo "4"
    fi
}

setup_cluster_config() {
    local partition="${1:-batch}"
    PARTITION="${PARTITION:-$partition}"
    if [[ -z "${GPUS_PER_NODE:-}" ]]; then
        GPUS_PER_NODE=$(detect_gpus_per_node "$partition")
    fi
    if [[ "$GPUS_PER_NODE" -eq 8 ]]; then
        CLUSTER_TYPE="H100"
    else
        CLUSTER_TYPE="GB200"
    fi
    # GRES flag: Lyris rejects --gres=gpu:N, so allow explicit opt-out via
    # GRES_FLAG= (empty). Default is "--gres=gpu:${GPUS_PER_NODE}".
    if [[ -z "${GRES_FLAG+x}" ]]; then
        GRES_FLAG="--gres=gpu:${GPUS_PER_NODE}"
    fi

    BASE="${BASE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}"
    CONTAINER="${CONTAINER:-${BASE}/HybridEP_test/nemo_rl.sqsh}"
    MOUNTS="${MOUNTS:-/lustre:/lustre}"
    ACCOUNT="${ACCOUNT:-coreai_dlalgo_nemorl}"
    HF_HOME="${HF_HOME:-${BASE}/HybridEP_test/hf_home}"
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"

    echo "[INFO] Cluster: ${CLUSTER_TYPE}, GPUs/node: ${GPUS_PER_NODE}, Partition: ${PARTITION}"
    echo "[INFO] Account: ${ACCOUNT}, GRES: ${GRES_FLAG:-<none>}"
    echo "[INFO] Container: ${CONTAINER}"
    echo "[INFO] HF_HOME: ${HF_HOME}"
}

export_cluster_config() {
    export GPUS_PER_NODE CONTAINER GRES_FLAG CLUSTER_TYPE PARTITION
    export BASE MOUNTS ACCOUNT HF_HOME HF_DATASETS_CACHE
}

# Submit a single perf_test variant.
# Args: PROJECT_ROOT CONFIG_REL NUM_NODES JOB_NAME [EXTRA_ENV]
# Example: submit_variant "$BASE/RL-selective-recompute" "perf_test/qwen3_30ba3b/recompute_00_no_ckpt" 4 "nrl-recompute-qwen-no-ckpt"
submit_variant() {
    local project_root="$1"
    local config_rel="$2"
    local num_nodes="$3"
    local job_name="$4"
    local extra_env="${5:-}"

    local log_dir="${project_root}/logs/${config_rel}"
    mkdir -p "$log_dir"

    local uv_extra=""
    if [[ "$project_root" == *"moe-compute-opts"* ]] || [[ "$project_root" == *"high-priority-streams"* ]]; then
        uv_extra="--extra mcore"
    fi

    # NRL_FORCE_REBUILD_VENVS=false forces uv to re-sync venvs against the
    # current pyproject.toml/uv.lock on every run. NEMO_RL_VENV_DIR pins the
    # venv location into the project dir so each worktree has its own venv.
    # Both must be exported inside the container, not just the login shell.
    local command="cd ${project_root} && export NRL_IGNORE_VERSION_MISMATCH=1 NRL_FORCE_REBUILD_VENVS=false NEMO_RL_VENV_DIR=${project_root}/venvs CUDA_HOME=/usr/local/cuda HF_HOME=${HF_HOME} HF_DATASETS_CACHE=${HF_DATASETS_CACHE} HF_HUB_OFFLINE=1 ${extra_env} && uv run ${uv_extra} examples/run_grpo.py --config examples/configs/${config_rel}.yaml"

    # TIME_LIMIT override lets longer-running experiments use longer partitions
    # (Lyris gb200 has a 5h cap vs OCI-HSG's shorter slots). Default 1:30:00
    # covers the original short sweeps.
    local time_limit="${TIME_LIMIT:-1:30:00}"

    echo "[SUBMIT] ${job_name} (nodes=${num_nodes}, time=${time_limit})"
    CONTAINER="$CONTAINER" MOUNTS="$MOUNTS" GPUS_PER_NODE="$GPUS_PER_NODE" \
        COMMAND="$command" BASE_LOG_DIR="$log_dir" \
        sbatch \
            --nodes="$num_nodes" \
            $GRES_FLAG \
            --time="$time_limit" \
            --segment="$num_nodes" \
            -A "$ACCOUNT" -p "$PARTITION" \
            --job-name="$job_name" \
            --output="${log_dir}/slurm-%j.out" \
            "${project_root}/ray.sub"
}
