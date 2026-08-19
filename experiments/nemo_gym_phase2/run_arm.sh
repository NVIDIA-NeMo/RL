#!/bin/bash

set -euo pipefail

: "${PHASE2_REPO:?PHASE2_REPO must be set}"
: "${PHASE2_RUN_ROOT:?PHASE2_RUN_ROOT must be set}"
: "${PHASE2_ARM:?PHASE2_ARM must be set}"
: "${PHASE2_RUN_ID:?PHASE2_RUN_ID must be set}"
: "${PHASE2_REPEAT_ID:?PHASE2_REPEAT_ID must be set}"
: "${PHASE2_NUM_PROMPTS:?PHASE2_NUM_PROMPTS must be set}"
: "${PHASE2_NUM_GENERATIONS:?PHASE2_NUM_GENERATIONS must be set}"
: "${PHASE2_RUNTIME_ENV:?PHASE2_RUNTIME_ENV must be set}"
: "${PHASE2_NEMO_GYM_ENV:?PHASE2_NEMO_GYM_ENV must be set}"
: "${PHASE2_VLLM_ENV_VERIFICATION:?PHASE2_VLLM_ENV_VERIFICATION must be set}"
: "${PHASE2_NEMO_GYM_ENV_VERIFICATION:?PHASE2_NEMO_GYM_ENV_VERIFICATION must be set}"
: "${PHASE2_PYTHON_INSTALL_DIR:?PHASE2_PYTHON_INSTALL_DIR must be set}"
: "${PHASE2_VENV_DIR:?PHASE2_VENV_DIR must be set}"
: "${PHASE2_UV_CACHE_DIR:?PHASE2_UV_CACHE_DIR must be set}"
: "${PHASE2_RL_INSIGHT_SOURCE:?PHASE2_RL_INSIGHT_SOURCE must be set}"
: "${PHASE2_PROMETHEUS_BIN:?PHASE2_PROMETHEUS_BIN must be set}"
: "${PHASE2_UV_BIN_DIR:?PHASE2_UV_BIN_DIR must be set}"
: "${PHASE2_CONTAINER_DIGEST:?PHASE2_CONTAINER_DIGEST must be set}"
: "${PHASE2_MODEL_SNAPSHOT:?PHASE2_MODEL_SNAPSHOT must be set}"
: "${PHASE2_MODEL_REVISION:?PHASE2_MODEL_REVISION must be set}"

# ray.sub intentionally removes SLURM_* variables before starting Ray so MPI
# libraries imported by Ray actors do not mistake the Ray process tree for an
# MPI job. Resolve the allocation identity from the write-once launch sidecar
# when the driver therefore cannot inherit SLURM_JOB_ID.
PHASE2_SLURM_JOB_ID=${SLURM_JOB_ID:-}
if [[ -z $PHASE2_SLURM_JOB_ID ]]; then
  PHASE2_JOB_ID_FILE=$PHASE2_RUN_ROOT/job_id
  [[ -s $PHASE2_JOB_ID_FILE ]] || {
    echo "Missing Phase 2 Slurm job identity: $PHASE2_JOB_ID_FILE" >&2
    exit 1
  }
  PHASE2_SLURM_JOB_ID=$(<"$PHASE2_JOB_ID_FILE")
  # sbatch --parsable may append a cluster name after a semicolon.
  PHASE2_SLURM_JOB_ID=${PHASE2_SLURM_JOB_ID%%;*}
fi
[[ $PHASE2_SLURM_JOB_ID =~ ^[0-9]+$ ]] || {
  echo "Invalid Phase 2 Slurm job identity: $PHASE2_SLURM_JOB_ID" >&2
  exit 1
}

MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
MODEL_PATH=$PHASE2_MODEL_SNAPSHOT
SEED=42
WARMUP_REQUESTS=1
CONFIG=$PHASE2_REPO/examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
WORKLOAD=$PHASE2_RUN_ROOT/workload.jsonl
WARMUP_WORKLOAD=$PHASE2_RUN_ROOT/warmup-workload.jsonl
NEMO_LOG_DIR=$PHASE2_RUN_ROOT/nemo-logs
DRIVER_LOG=$PHASE2_RUN_ROOT/driver.log
RL_INSIGHT_CONFIG=$PHASE2_REPO/examples/nemo_gym/rl_insight_phase2/config.yaml

mkdir -p "$PHASE2_RUN_ROOT"
exec > >(tee -a "$DRIVER_LOG") 2>&1

export PATH=$PHASE2_UV_BIN_DIR:$PHASE2_RUNTIME_ENV/bin:$PATH
export PYTHONPATH=$PHASE2_REPO:$PHASE2_REPO/3rdparty/Gym-workspace/Gym:$PHASE2_RL_INSIGHT_SOURCE
export UV_PYTHON_INSTALL_DIR=$PHASE2_PYTHON_INSTALL_DIR
# The container image has its own /opt/ray_venvs default. Restore the audited,
# lock-specific shared root before NeMo RL resolves any actor environment.
export NEMO_RL_VENV_DIR=$PHASE2_VENV_DIR
export UV_CACHE_DIR_OVERRIDE=$PHASE2_UV_CACHE_DIR
export DG_USE_LOCAL_VERSION=0
unset NEMO_RL_PY_EXECUTABLES_SYSTEM NEMO_RL_SYSTEM_PY_EXECUTABLE
export PHASE2_RL_INSIGHT_ROOT=$PHASE2_RUN_ROOT/rl-insight
export PHASE2_PROMETHEUS_BIN
export NEMO_RL_RUN_ID=$PHASE2_RUN_ID
export NEMO_RL_PHASE2_WARMUP_REQUESTS=$WARMUP_REQUESTS
export NEMO_RL_PHASE2_WARMUP_RESULT_PATH=$PHASE2_RUN_ROOT/warmup-results.jsonl
export NEMO_RL_PHASE2_MODEL_CALL_CAPTURE_DIR=$NEMO_LOG_DIR/nemo_gym_monitoring/model_call_capture
export NEMO_RL_PHASE2_WARMUP_SETTLE_SECONDS=4
export NEMO_RL_PHASE2_WARMUP_WORKLOAD_SHA256
export NEMO_RL_PHASE2_RESOLVED_CONFIG=$PHASE2_RUN_ROOT/resolved-config.json
NEMO_RL_PHASE2_WARMUP_WORKLOAD_SHA256=$(sha256sum "$WARMUP_WORKLOAD")
NEMO_RL_PHASE2_WARMUP_WORKLOAD_SHA256=${NEMO_RL_PHASE2_WARMUP_WORKLOAD_SHA256%% *}

# RL-Insight probes service versions with a five-second timeout. Starting the
# audited binary directly from Lustre can exceed that timeout under metadata or
# I/O load, so stage the byte-identical executable on the allocated node.
PROMETHEUS_SOURCE_BIN=$PHASE2_PROMETHEUS_BIN
PROMETHEUS_LOCAL_ROOT=${SLURM_TMPDIR:-/tmp}/nemo-gym-phase2-$PHASE2_SLURM_JOB_ID
PROMETHEUS_LOCAL_BIN=$PROMETHEUS_LOCAL_ROOT/prometheus
mkdir -p "$PROMETHEUS_LOCAL_ROOT"
cp --preserve=mode,timestamps "$PROMETHEUS_SOURCE_BIN" "$PROMETHEUS_LOCAL_BIN"
PROMETHEUS_SOURCE_SHA256=$(sha256sum "$PROMETHEUS_SOURCE_BIN")
PROMETHEUS_SOURCE_SHA256=${PROMETHEUS_SOURCE_SHA256%% *}
PROMETHEUS_LOCAL_SHA256=$(sha256sum "$PROMETHEUS_LOCAL_BIN")
PROMETHEUS_LOCAL_SHA256=${PROMETHEUS_LOCAL_SHA256%% *}
[[ $PROMETHEUS_LOCAL_SHA256 == "$PROMETHEUS_SOURCE_SHA256" ]] || {
  echo "Node-local Prometheus copy differs from the audited source binary" >&2
  exit 1
}
export PHASE2_PROMETHEUS_BIN=$PROMETHEUS_LOCAL_BIN

NODE_IP=$(hostname -I)
NODE_IP=${NODE_IP%% *}
export NO_PROXY=${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$NODE_IP
export no_proxy=$NO_PROXY
export RL_INSIGHT_SERVER_URL=http://$NODE_IP:18080

stop_rl_insight() {
  "$PHASE2_RUNTIME_ENV/bin/python" -m rl_insight.cli server stop \
    --config "$RL_INSIGHT_CONFIG" || true
}
trap stop_rl_insight EXIT TERM INT

echo "PHASE2_RUN_ID=$PHASE2_RUN_ID"
echo "PHASE2_ARM=$PHASE2_ARM"
echo "PHASE2_SLURM_JOB_ID=$PHASE2_SLURM_JOB_ID"
echo "PHASE2_EXPERIMENT_COMMIT=$(git -C "$PHASE2_REPO" rev-parse HEAD)"
echo "PHASE2_GYM_COMMIT=$(git -C "$PHASE2_REPO/3rdparty/Gym-workspace/Gym" rev-parse HEAD)"
echo "PHASE2_PROMETHEUS_SOURCE_BIN=$PROMETHEUS_SOURCE_BIN"
echo "PHASE2_PROMETHEUS_BINARY_SHA256=$PROMETHEUS_LOCAL_SHA256"
UV_VERSION=$(uv --version)
RL_INSIGHT_VERSION=$("$PHASE2_RUNTIME_ENV/bin/python" -c \
  'import rl_insight; print(rl_insight.__version__)')
NEMO_GYM_VERSION=$("$PHASE2_NEMO_GYM_ENV/bin/python" -c \
  'import importlib.metadata as m; print(m.version("nemo-gym"))')
VLLM_ROUTER_VERSION=$("$PHASE2_NEMO_GYM_ENV/bin/python" -c \
  'import importlib.metadata as m; print(m.version("vllm-router"))')
echo "$UV_VERSION"
"$PHASE2_PROMETHEUS_BIN" --version
"$PHASE2_RUNTIME_ENV/bin/python" -c \
  'import nemo_rl, ray, rl_insight, vllm; print("nemo_rl=" + nemo_rl.__file__); print("driver ray=" + ray.__version__ + " vllm=" + vllm.__version__ + " rl_insight=" + rl_insight.__version__)'
"$PHASE2_NEMO_GYM_ENV/bin/python" -c \
  'import importlib.metadata as m, nemo_gym, ray, vllm_router; print("nemo_gym=" + nemo_gym.__file__); print("gym ray=" + ray.__version__ + " router=" + m.version("vllm-router"))'

"$PHASE2_RUNTIME_ENV/bin/python" -m rl_insight.cli server start \
  --config "$RL_INSIGHT_CONFIG" --detach
"$PHASE2_RUNTIME_ENV/bin/python" -c \
  'import os, time, urllib.request; url=os.environ["RL_INSIGHT_SERVER_URL"]+"/healthz"; deadline=time.monotonic()+30; error=None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                break
    except Exception as exc:
        error=exc
    time.sleep(0.2)
else:
    raise RuntimeError(f"RL-Insight health timeout: {error}")'

ROUTER_ENABLED=false
ROUTER_POLICY=consistent_hash
ROUTER_CACHE_METRICS_MODE=native
case "$PHASE2_ARM" in
  direct) ;;
  cache_aware)
    ROUTER_ENABLED=true
    ROUTER_POLICY=cache_aware
    ROUTER_CACHE_METRICS_MODE=debug_log_compat
    ;;
  consistent_hash)
    ROUTER_ENABLED=true
    ROUTER_POLICY=consistent_hash
    ;;
  *)
    echo "Unsupported Phase 2 arm: $PHASE2_ARM" >&2
    exit 2
    ;;
esac

OVERRIDES=(
  "policy.model_name=$MODEL_PATH"
  "policy.tokenizer.name=$MODEL_PATH"
  "policy.max_total_sequence_length=32768"
  "policy.generation.max_new_tokens=256"
  "policy.generation.temperature=1.0"
  "policy.generation.top_p=1.0"
  "policy.generation.top_k=null"
  "policy.generation.vllm_cfg.tensor_parallel_size=1"
  "policy.generation.vllm_cfg.enforce_eager=true"
  "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.tool_parser=hermes"
  "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.enable_auto_tools=true"
  "+policy.generation.vllm_cfg.enable_prefix_caching=true"
  "+policy.generation.vllm_kwargs.max_num_seqs=256"
  "+policy.generation.vllm_kwargs.max_num_batched_tokens=8192"
  "policy.generation.colocated.enabled=false"
  "policy.generation.colocated.resources.gpus_per_node=8"
  "policy.generation.colocated.resources.num_nodes=1"
  "~policy.generation.vllm_kwargs.compilation_config"
  "~policy.generation.vllm_kwargs.mamba_ssm_cache_dtype"
  "grpo.num_prompts_per_step=$PHASE2_NUM_PROMPTS"
  "grpo.num_generations_per_prompt=$PHASE2_NUM_GENERATIONS"
  "grpo.seed=$SEED"
  "data.shuffle=false"
  "data.validation.data_path=$WORKLOAD"
  "logger.log_dir=$NEMO_LOG_DIR"
  "logger.wandb_enabled=false"
  "logger.tensorboard_enabled=false"
  "logger.mlflow_enabled=false"
  "logger.swanlab_enabled=false"
  "logger.monitor_gpus=false"
  "logger.num_val_samples_to_print=0"
  "cluster.num_nodes=1"
  "cluster.gpus_per_node=8"
  "env.nemo_gym.vllm_router.enabled=$ROUTER_ENABLED"
  "env.nemo_gym.vllm_router.policy=$ROUTER_POLICY"
  "env.nemo_gym.vllm_router.cache_metrics_mode=$ROUTER_CACHE_METRICS_MODE"
  "env.nemo_gym.vllm_router.cache_threshold=0.3"
  "env.nemo_gym.prometheus.enabled=true"
  "env.nemo_gym.prometheus.required=true"
  "env.nemo_gym.prometheus.scrape_interval_s=1"
  "env.nemo_gym.prometheus.initial_scrape_wait_s=2"
  "env.nemo_gym.prometheus.final_scrape_wait_s=2"
  "env.nemo_gym.prometheus.target_lifecycle=dedicated"
  "++env.nemo_gym.prometheus.server_url=$RL_INSIGHT_SERVER_URL"
)

time "$PHASE2_RUNTIME_ENV/bin/python" -u \
  "$PHASE2_REPO/examples/nemo_gym/run_grpo_rollout_benchmark.py" \
  --config "$CONFIG" "${OVERRIDES[@]}"

METADATA=$PHASE2_RUN_ROOT/experiment-metadata.json
"$PHASE2_RUNTIME_ENV/bin/python" \
  "$PHASE2_REPO/experiments/nemo_gym_phase2/create_metadata.py" \
  --repo "$PHASE2_REPO" \
  --output "$METADATA" \
  --workload "$WORKLOAD" \
  --warmup "$WARMUP_WORKLOAD" \
  --model-snapshot "$PHASE2_MODEL_SNAPSHOT" \
  --model-name "$MODEL_PATH" \
  --model-repo-id "$MODEL_NAME" \
  --model-revision "$PHASE2_MODEL_REVISION" \
  --container-digest "$PHASE2_CONTAINER_DIGEST" \
  --rl-insight-source "$PHASE2_RL_INSIGHT_SOURCE" \
  --prometheus-bin "$PHASE2_PROMETHEUS_BIN" \
  --uv-bin "$PHASE2_UV_BIN_DIR/uv" \
  --runtime-env "$PHASE2_RUNTIME_ENV" \
  --nemo-gym-env "$PHASE2_NEMO_GYM_ENV" \
  --runtime-verification "$PHASE2_VLLM_ENV_VERIFICATION" \
  --nemo-gym-verification "$PHASE2_NEMO_GYM_ENV_VERIFICATION" \
  --launch-id "slurm-${PHASE2_SLURM_JOB_ID}-${PHASE2_RUN_ID}" \
  --routing-policy "$PHASE2_ARM" \
  --seed "$SEED" \
  --num-prompts "$PHASE2_NUM_PROMPTS" \
  --num-generations "$PHASE2_NUM_GENERATIONS" \
  --warmup-requests "$WARMUP_REQUESTS"

"$PHASE2_RUNTIME_ENV/bin/python" \
  "$PHASE2_REPO/tools/nemo_gym_phase2_report.py" \
  --prometheus-targets "$NEMO_LOG_DIR/nemo_gym_monitoring/prometheus-targets.json" \
  --driver-log "$DRIVER_LOG" \
  --eval-results "$NEMO_LOG_DIR/nemo_gym_eval_results.jsonl" \
  --workload-file "$WORKLOAD" \
  --warmup-workload-file "$WARMUP_WORKLOAD" \
  --workload-seed "$SEED" \
  --repeat-id "$PHASE2_REPEAT_ID" \
  --command-file "$PHASE2_RUN_ROOT/command.txt" \
  --experiment-metadata "$METADATA" \
  --config "$CONFIG" \
  --config "$NEMO_RL_PHASE2_RESOLVED_CONFIG" \
  --version "rl_insight=$RL_INSIGHT_VERSION" \
  --version "nemo_gym=$NEMO_GYM_VERSION" \
  --version "vllm_router=$VLLM_ROUTER_VERSION" \
  --version "uv=$UV_VERSION" \
  --prometheus-url "http://$NODE_IP:19090" \
  --range-step-s 1 \
  --output-dir "$PHASE2_RUN_ROOT/report"

trap - EXIT TERM INT
stop_rl_insight
