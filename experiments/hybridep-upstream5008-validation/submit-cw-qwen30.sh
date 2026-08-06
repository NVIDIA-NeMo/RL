#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null || die "missing command: $1"; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_PATH=${SOURCE_PATH:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-upstream5008-validation/cw-h100}
HF_CACHE=${HF_CACHE:-$OUTPUT_ROOT/hf-cache}
ACCOUNT=${ACCOUNT:?Set ACCOUNT after checking FairShare immediately before submission}
CONTAINER=${CONTAINER:?Set CONTAINER to an immutable image reference or local squashfs}
DEEPEP_WHEEL=${DEEPEP_WHEEL:?Set DEEPEP_WHEEL to the HybridEP 17cfb817 wheel}
DEEPEP_COMMIT_FILE=${DEEPEP_COMMIT_FILE:-$DEEPEP_WHEEL.commit}
FORK_REMOTE=${FORK_REMOTE:-fork}
FORK_BRANCH=${FORK_BRANCH:-sna/hybridep-always-pad-uneven-20260805}
TEST_ONLY=${TEST_ONLY:-0}
EXPECTED_TASK1_COMMIT=5de79fc3dd117f6bd6a9ccc3fb38732d9f624ea4
EXPECTED_BRIDGE_COMMIT=573e088c9c6740082c39744e03dc5b009e730ed4
EXPECTED_MCORE_COMMIT=6513e3e23d6b5eda6a1c934990b15e804237732b
MCORE_5008_COMMIT=81770cb015eab05785ecd540ba929d1400a52f67
EXPECTED_DEEPEP_COMMIT=17cfb817bccec3a9c247013360cc550c2bac441e
EXPECTED_GPU_MODEL=H100
EXPECTED_GPUS_PER_NODE=8
RECIPE=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml

for command_name in git sbatch sshare sha256sum; do require_command "$command_name"; done
[[ "$TEST_ONLY" == 0 || "$TEST_ONLY" == 1 ]] || die "TEST_ONLY must be 0 or 1"
[[ -f "$SOURCE_PATH/$RECIPE" && -f "$SOURCE_PATH/ray.sub" ]] || die "invalid SOURCE_PATH: $SOURCE_PATH"
[[ -f "$DEEPEP_WHEEL" && -f "$DEEPEP_COMMIT_FILE" ]] || die "DeepEP wheel or commit sidecar is missing"
[[ $(tr -d '[:space:]' < "$DEEPEP_COMMIT_FILE") == "$EXPECTED_DEEPEP_COMMIT" ]] || die "DeepEP artifact commit mismatch"
git -C "$SOURCE_PATH" diff --quiet && git -C "$SOURCE_PATH" diff --cached --quiet || die "tracked source is dirty"
git -C "$SOURCE_PATH" merge-base --is-ancestor "$EXPECTED_TASK1_COMMIT" HEAD || die "Task 1 commit is absent"
LOCAL_HEAD=$(git -C "$SOURCE_PATH" rev-parse HEAD)
PUSHED_HEAD=$(git -C "$SOURCE_PATH" ls-remote "$FORK_REMOTE" "refs/heads/$FORK_BRANCH" | cut -f1)
[[ -n "$PUSHED_HEAD" && "$LOCAL_HEAD" == "$PUSHED_HEAD" ]] || die "HEAD is not the pushed fork branch commit"

if [[ -f "$CONTAINER" ]]; then
  CONTAINER_SHA256=${CONTAINER_SHA256:?Set CONTAINER_SHA256 for a local container image}
  [[ $(sha256sum "$CONTAINER" | cut -d' ' -f1) == "$CONTAINER_SHA256" ]] || die "container checksum mismatch"
elif [[ ! "$CONTAINER" =~ @sha256:[0-9a-f]{64}$ ]]; then
  die "CONTAINER must be a local checksum-verified image or a digest-pinned reference"
fi

mkdir -p "$OUTPUT_ROOT" "$HF_CACHE"
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
FAIRSHARE_LOG=$OUTPUT_ROOT/fairshare-$RUN_STAMP.txt
sshare -A "$ACCOUNT" -u "$USER" -o Cluster,Account,User,FairShare | tee "$FAIRSHARE_LOG"
grep -F "$ACCOUNT" "$FAIRSHARE_LOG" >/dev/null || die "ACCOUNT is absent from FairShare output"
DEEPEP_SHA256=$(sha256sum "$DEEPEP_WHEEL" | cut -d' ' -f1)
PROVENANCE_ROOT=$OUTPUT_ROOT/provenance-$RUN_STAMP
mkdir -p "$PROVENANCE_ROOT"
printf 'nemo_rl_commit=%s\naccount=%s\ncontainer=%s\ndeepep_wheel=%s\ndeepep_sha256=%s\nrecipe=%s\n' \
  "$LOCAL_HEAD" "$ACCOUNT" "$CONTAINER" "$DEEPEP_WHEEL" "$DEEPEP_SHA256" "$RECIPE" > "$PROVENANCE_ROOT/submission.txt"

export SOURCE_PATH OUTPUT_ROOT HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets"
export EXPECTED_NEMO_RL_COMMIT="$LOCAL_HEAD" EXPECTED_BRIDGE_COMMIT EXPECTED_MCORE_COMMIT
export MCORE_5008_COMMIT EXPECTED_DEEPEP_COMMIT DEEPEP_WHEEL DEEPEP_COMMIT_FILE DEEPEP_SHA256
export EXPECTED_GPU_MODEL EXPECTED_GPUS_PER_NODE PROVENANCE_ROOT RECIPE
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NVLINK_DOMAIN_SIZE=8 USE_MNNVL=0 GPUS_PER_NODE=8 BASE_LOG_DIR="$OUTPUT_ROOT"
EXTRA_MOUNTS=${MOUNTS:-}
DEEPEP_DIR=$(dirname "$DEEPEP_WHEEL")
export MOUNTS="$SOURCE_PATH:$SOURCE_PATH,$OUTPUT_ROOT:$OUTPUT_ROOT,$HF_CACHE:$HF_CACHE,$DEEPEP_DIR:$DEEPEP_DIR${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
export CONTAINER

read -r -d '' SETUP_COMMAND <<'SETUP' || true
set -euo pipefail
cd "$SOURCE_PATH"
RUN_PYTHON=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
[[ $("$RUN_PYTHON" -c 'import platform; print(platform.python_version())') == 3.13.14 ]]
GPU_MODELS=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $(printf '%s\n' "$GPU_MODELS" | sed '/^$/d' | wc -l) -eq "$EXPECTED_GPUS_PER_NODE" ]]
[[ "$GPU_MODELS" == *"$EXPECTED_GPU_MODEL"* ]]
[[ $(tr -d '[:space:]' < "$DEEPEP_COMMIT_FILE") == "$EXPECTED_DEEPEP_COMMIT" ]]
[[ $(sha256sum "$DEEPEP_WHEEL" | cut -d' ' -f1) == "$DEEPEP_SHA256" ]]
uv pip install --python "$RUN_PYTHON" --no-deps --reinstall "$DEEPEP_WHEEL"
{
  printf 'host=%s\npython=%s\ngpu_count=%s\ngpu_models=%s\n' "$(hostname)" "$RUN_PYTHON" "$EXPECTED_GPUS_PER_NODE" "$GPU_MODELS"
  "$RUN_PYTHON" -c 'import deep_ep, importlib.metadata; print("deep_ep_module=" + deep_ep.__file__); print("deep_ep_version=" + importlib.metadata.version("deep-ep"))'
} > "$PROVENANCE_ROOT/node-$(hostname).txt"
SETUP
export SETUP_COMMAND

read -r -d '' COMMAND <<'DRIVER' || true
set -euo pipefail
cd "$SOURCE_PATH"
[[ $(git rev-parse HEAD) == "$EXPECTED_NEMO_RL_COMMIT" ]]
BRIDGE=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE=$BRIDGE/3rdparty/Megatron-LM
[[ $(git -C "$BRIDGE" rev-parse HEAD) == "$EXPECTED_BRIDGE_COMMIT" ]]
[[ $(git -C "$MCORE" rev-parse HEAD) == "$EXPECTED_MCORE_COMMIT" ]]
[[ $(git -C "$BRIDGE" remote get-url origin) == https://github.com/NVIDIA-NeMo/Megatron-Bridge.git ]]
[[ $(git -C "$MCORE" remote get-url origin) == https://github.com/NVIDIA/Megatron-LM.git ]]
git -C "$MCORE" merge-base --is-ancestor "$MCORE_5008_COMMIT" HEAD
uv run --no-sync python - <<'PY'
import os
from types import SimpleNamespace
from nemo_rl.models.megatron.setup import _apply_moe_config
assert os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "8"
assert os.environ["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == "128"
assert os.environ["NVLINK_DOMAIN_SIZE"] == "8"
assert os.environ["USE_MNNVL"] == "0"
cfg = {"megatron_cfg": {"expert_tensor_parallel_size": 1, "expert_model_parallel_size": 8, "moe_router_dtype": "float32", "moe_router_load_balancing_type": "none", "moe_router_bias_update_rate": 0.0, "moe_permute_fusion": True, "moe_enable_deepep": False, "moe_token_dispatcher_type": "flex", "moe_shared_expert_overlap": True, "moe_flex_dispatcher_backend": "hybridep"}}
model_cfg = SimpleNamespace(moe_hybridep_pad_uneven_dispatch_inputs=False)
_apply_moe_config(model_cfg, cfg)
assert model_cfg.moe_hybridep_pad_uneven_dispatch_inputs is True
print("moe_hybridep_pad_uneven_dispatch_inputs=True")
PY
git -C "$BRIDGE" remote get-url origin > "$PROVENANCE_ROOT/megatron-bridge-origin.txt"
git -C "$MCORE" remote get-url origin > "$PROVENANCE_ROOT/megatron-lm-origin.txt"
uv run --no-sync examples/run_grpo.py --config "$RECIPE" \
  grpo.max_num_steps=20 checkpointing.enabled=false \
  policy.megatron_cfg.moe_token_dispatcher_type=flex \
  policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep \
  policy.megatron_cfg.moe_hybridep_num_sms=32 \
  policy.sequence_packing.enabled=true \
  logger.log_dir="$OUTPUT_ROOT/training-$SLURM_JOB_ID" 2>&1 | tee "$OUTPUT_ROOT/training-$SLURM_JOB_ID.log"
DRIVER
export COMMAND

SBATCH_ARGS=(--nodes=4 --gres=gpu:8 --segment=4 --account="$ACCOUNT" --partition=batch
  --time=01:00:00 --exclusive --job-name="$ACCOUNT:hybridep5008-cw"
  --output="$OUTPUT_ROOT/slurm-%j.out" --error="$OUTPUT_ROOT/slurm-%j.out" --export=ALL)
[[ "$TEST_ONLY" == 1 ]] && SBATCH_ARGS+=(--test-only)
sbatch "${SBATCH_ARGS[@]}" "$SOURCE_PATH/ray.sub"
