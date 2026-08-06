#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null || die "missing command: $1"; }
assert_source_clean() {
  [[ -z $(git -C "$SOURCE_PATH" status --porcelain --untracked-files=all) ]] || die "NeMo-RL source is dirty"
  [[ -z $(git -C "$SOURCE_PATH" submodule foreach --recursive --quiet 'dirty=$(git status --porcelain --untracked-files=all); if [ -n "$dirty" ]; then printf "%s\n" "$displaypath"; fi') ]] || die "recursive submodule source is dirty"
  ! git -C "$SOURCE_PATH" submodule status --recursive | grep -Eq '^[+-U]' || die "recursive submodule checkout mismatch"
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_PATH=${SOURCE_PATH:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-upstream5008-validation/oci-gb200}
HF_CACHE=${HF_CACHE:-$OUTPUT_ROOT/hf-cache}
ACCOUNT=${ACCOUNT:?Set ACCOUNT after checking FairShare immediately before submission}
CONTAINER=${CONTAINER:?Set CONTAINER to an immutable image reference or local squashfs}
DEEPEP_WHEEL=${DEEPEP_WHEEL:?Set DEEPEP_WHEEL to the HybridEP 17cfb817 wheel}
DEEPEP_METADATA=${DEEPEP_METADATA:?Set DEEPEP_METADATA to build-generated wheel metadata JSON}
FORK_REMOTE=${FORK_REMOTE:-fork}
FORK_BRANCH=${FORK_BRANCH:-sna/hybridep-always-pad-uneven-20260805}
TEST_ONLY=${TEST_ONLY:-0}
EXPECTED_TASK1_COMMIT=5de79fc3dd117f6bd6a9ccc3fb38732d9f624ea4
EXPECTED_BRIDGE_COMMIT=573e088c9c6740082c39744e03dc5b009e730ed4
EXPECTED_MCORE_COMMIT=6513e3e23d6b5eda6a1c934990b15e804237732b
MCORE_5008_COMMIT=81770cb015eab05785ecd540ba929d1400a52f67
EXPECTED_DEEPEP_COMMIT=17cfb817bccec3a9c247013360cc550c2bac441e
EXPECTED_GPU_MODEL=GB200
EXPECTED_GPUS_PER_NODE=4
EXPECTED_ARCHITECTURE=aarch64
RECIPE=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml

for command_name in git python3 sbatch sshare sha256sum; do require_command "$command_name"; done
[[ "$TEST_ONLY" == 0 || "$TEST_ONLY" == 1 ]] || die "TEST_ONLY must be 0 or 1"
[[ -f "$SOURCE_PATH/$RECIPE" && -f "$SOURCE_PATH/ray.sub" ]] || die "invalid SOURCE_PATH: $SOURCE_PATH"
[[ -f "$DEEPEP_WHEEL" && -f "$DEEPEP_METADATA" ]] || die "DeepEP wheel or build metadata is missing"
mapfile -t DEEPEP_META < <(python3 - "$DEEPEP_METADATA" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)
for key in ("commit", "platform", "architecture", "wheel", "sha256"):
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"missing DeepEP metadata field: {key}")
    print(value)
PY
)
[[ ${#DEEPEP_META[@]} -eq 5 ]] || die "invalid DeepEP build metadata"
[[ ${DEEPEP_META[0]} == "$EXPECTED_DEEPEP_COMMIT" && ${DEEPEP_META[1]} == linux && ${DEEPEP_META[2]} == "$EXPECTED_ARCHITECTURE" ]] || die "DeepEP build metadata platform or commit mismatch"
[[ ${DEEPEP_META[3]} == "$DEEPEP_WHEEL" || ${DEEPEP_META[3]} == "$(basename "$DEEPEP_WHEEL")" ]] || die "DeepEP build metadata wheel mismatch"
[[ ${DEEPEP_META[4]} =~ ^[0-9a-f]{64}$ ]] || die "invalid DeepEP metadata SHA256"
assert_source_clean
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
[[ "$DEEPEP_SHA256" == "${DEEPEP_META[4]}" ]] || die "DeepEP wheel checksum does not match build metadata"
PROVENANCE_ROOT=$OUTPUT_ROOT/provenance-$RUN_STAMP
mkdir -p "$PROVENANCE_ROOT"
printf 'nemo_rl_commit=%s\naccount=%s\ncontainer=%s\ndeepep_commit=%s\ndeepep_platform=%s\ndeepep_architecture=%s\ndeepep_wheel=%s\ndeepep_metadata=%s\ndeepep_sha256=%s\nrecipe=%s\n' \
  "$LOCAL_HEAD" "$ACCOUNT" "$CONTAINER" "$EXPECTED_DEEPEP_COMMIT" "${DEEPEP_META[1]}" "${DEEPEP_META[2]}" "$DEEPEP_WHEEL" "$DEEPEP_METADATA" "$DEEPEP_SHA256" "$RECIPE" > "$PROVENANCE_ROOT/submission.txt"

export SOURCE_PATH OUTPUT_ROOT HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets"
export EXPECTED_NEMO_RL_COMMIT="$LOCAL_HEAD" EXPECTED_BRIDGE_COMMIT EXPECTED_MCORE_COMMIT
export MCORE_5008_COMMIT EXPECTED_DEEPEP_COMMIT DEEPEP_WHEEL DEEPEP_METADATA DEEPEP_SHA256
export EXPECTED_GPU_MODEL EXPECTED_GPUS_PER_NODE PROVENANCE_ROOT RECIPE
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NVLINK_DOMAIN_SIZE=72 USE_MNNVL=1 GPUS_PER_NODE=4 BASE_LOG_DIR="$OUTPUT_ROOT"
export NEMO_RL_VENV_DIR="$OUTPUT_ROOT/venvs/$LOCAL_HEAD"
export DEEPEP_OVERLAY_DIR="/tmp/nemo-rl-deepep-overlay-$DEEPEP_SHA256"
export PYTHONPATH="$DEEPEP_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR/deep_ep${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EXTRA_MOUNTS=${MOUNTS:-}
DEEPEP_DIR=$(dirname "$DEEPEP_WHEEL")
export MOUNTS="$SOURCE_PATH:$SOURCE_PATH,$OUTPUT_ROOT:$OUTPUT_ROOT,$HF_CACHE:$HF_CACHE,$DEEPEP_DIR:$DEEPEP_DIR${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
export CONTAINER

read -r -d '' SETUP_COMMAND <<'SETUP' || true
set -euo pipefail
cd "$SOURCE_PATH"
RUN_PYTHON=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
PYTHON_VERSION=$("$RUN_PYTHON" -c 'import platform; print(platform.python_version())')
[[ "$PYTHON_VERSION" == 3.13.14 ]]
GPU_MODELS=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $(printf '%s\n' "$GPU_MODELS" | sed '/^$/d' | wc -l) -eq "$EXPECTED_GPUS_PER_NODE" ]]
[[ "$GPU_MODELS" == *"$EXPECTED_GPU_MODEL"* ]]
[[ $(sha256sum "$DEEPEP_WHEEL" | cut -d' ' -f1) == "$DEEPEP_SHA256" ]]
rm -rf "$DEEPEP_OVERLAY_DIR"
mkdir -p "$DEEPEP_OVERLAY_DIR"
uv pip install --target "$DEEPEP_OVERLAY_DIR" --no-deps --reinstall "$DEEPEP_WHEEL"
{
  printf 'host=%s\npython_executable=%s\npython_version=%s\ngpu_count=%s\ngpu_models=%s\noverlay=%s\n' "$(hostname)" "$RUN_PYTHON" "$PYTHON_VERSION" "$EXPECTED_GPUS_PER_NODE" "$GPU_MODELS" "$DEEPEP_OVERLAY_DIR"
} > "$PROVENANCE_ROOT/node-$(hostname).txt"
SETUP
export SETUP_COMMAND

read -r -d '' COMMAND <<'DRIVER' || true
set -euo pipefail
cd "$SOURCE_PATH"
[[ $(git rev-parse HEAD) == "$EXPECTED_NEMO_RL_COMMIT" ]]
[[ -z $(git status --porcelain --untracked-files=all) ]]
[[ -z $(git submodule foreach --recursive --quiet 'dirty=$(git status --porcelain --untracked-files=all); if [ -n "$dirty" ]; then printf "%s\n" "$displaypath"; fi') ]]
! git submodule status --recursive | grep -Eq '^[+-U]'
BRIDGE=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE=$BRIDGE/3rdparty/Megatron-LM
[[ $(git -C "$BRIDGE" rev-parse HEAD) == "$EXPECTED_BRIDGE_COMMIT" ]]
[[ $(git -C "$MCORE" rev-parse HEAD) == "$EXPECTED_MCORE_COMMIT" ]]
[[ $(git -C "$BRIDGE" remote get-url origin) == https://github.com/NVIDIA-NeMo/Megatron-Bridge.git ]]
[[ $(git -C "$MCORE" remote get-url origin) == https://github.com/NVIDIA/Megatron-LM.git ]]
git -C "$MCORE" merge-base --is-ancestor "$MCORE_5008_COMMIT" HEAD
BRIDGE_SHA=$(git -C "$BRIDGE" rev-parse HEAD)
MCORE_SHA=$(git -C "$MCORE" rev-parse HEAD)
BRIDGE_ORIGIN=$(git -C "$BRIDGE" remote get-url origin)
MCORE_ORIGIN=$(git -C "$MCORE" remote get-url origin)
printf 'nemo_rl_sha=%s\nbridge_sha=%s\nbridge_origin=%s\nmcore_sha=%s\nmcore_origin=%s\nmcore_5008_ancestor=true\n' "$EXPECTED_NEMO_RL_COMMIT" "$BRIDGE_SHA" "$BRIDGE_ORIGIN" "$MCORE_SHA" "$MCORE_ORIGIN" > "$PROVENANCE_ROOT/source.txt"
uv run --no-sync python - <<'PY'
import os
from types import SimpleNamespace
from nemo_rl.models.megatron.setup import _apply_moe_config
assert os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "16"
assert os.environ["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == "128"
assert os.environ["NVLINK_DOMAIN_SIZE"] == "72"
assert os.environ["USE_MNNVL"] == "1"
cfg = {"megatron_cfg": {"expert_tensor_parallel_size": 1, "expert_model_parallel_size": 16, "moe_router_dtype": "float32", "moe_router_load_balancing_type": "none", "moe_router_bias_update_rate": 0.0, "moe_permute_fusion": True, "moe_enable_deepep": False, "moe_token_dispatcher_type": "flex", "moe_shared_expert_overlap": True, "moe_flex_dispatcher_backend": "hybridep"}}
model_cfg = SimpleNamespace(moe_hybridep_pad_uneven_dispatch_inputs=False)
_apply_moe_config(model_cfg, cfg)
assert model_cfg.moe_hybridep_pad_uneven_dispatch_inputs is True
print("moe_hybridep_pad_uneven_dispatch_inputs=True")
PY
uv run --no-sync python - <<'PY' > "$PROVENANCE_ROOT/mcore-actor-overlay-probe.log"
import json
import os
import platform
import ray
import sys
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.utils.venvs import create_local_venv_on_each_node

ray.init(address="auto")
actor_fqn = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
actor_env = get_actor_python_env(actor_fqn)
assert actor_env.startswith("uv run --locked --extra mcore")
actor_python = create_local_venv_on_each_node(actor_env, actor_fqn)
probe_env = {key: os.environ[key] for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "DEEPEP_OVERLAY_DIR")}

@ray.remote(num_cpus=0)
def probe(expected_python: str):
    import deep_ep
    import deep_ep_cpp
    import hybrid_ep_cpp

    overlay = os.path.realpath(os.environ["DEEPEP_OVERLAY_DIR"])
    module_paths = {
        "deep_ep": os.path.realpath(deep_ep.__file__),
        "deep_ep_cpp": os.path.realpath(deep_ep_cpp.__file__),
        "hybrid_ep_cpp": os.path.realpath(hybrid_ep_cpp.__file__),
    }
    assert os.path.realpath(sys.executable) == os.path.realpath(expected_python)
    assert platform.python_version() == "3.13.14"
    assert all(os.path.commonpath((overlay, path)) == overlay for path in module_paths.values())
    return {"node_id": ray.get_runtime_context().get_node_id(), "python_executable": sys.executable, "python_version": platform.python_version(), **module_paths}

nodes = [node for node in ray.nodes() if node.get("Alive") and node.get("Resources", {}).get("CPU", 0) > 0]
refs = [probe.options(runtime_env={"py_executable": actor_python, "env_vars": probe_env}, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)).remote(actor_python) for node in nodes]
for result in ray.get(refs):
    print(json.dumps(result, sort_keys=True))
PY
uv run --no-sync examples/run_grpo.py --config "$RECIPE" \
  grpo.max_num_steps=20 checkpointing.enabled=false \
  policy.megatron_cfg.moe_token_dispatcher_type=flex \
  ++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep \
  ++policy.megatron_cfg.moe_hybridep_num_sms=32 \
  policy.sequence_packing.enabled=true \
  logger.log_dir="$OUTPUT_ROOT/training-$SLURM_JOB_ID" 2>&1 | tee "$OUTPUT_ROOT/training-$SLURM_JOB_ID.log"
DRIVER
export COMMAND

SBATCH_ARGS=(--nodes=4 --gres=gpu:4 --segment=4 --account="$ACCOUNT" --partition=batch
  --time=04:00:00 --exclusive --job-name="$ACCOUNT:hybridep5008-oci"
  --output="$OUTPUT_ROOT/slurm-%j.out" --error="$OUTPUT_ROOT/slurm-%j.out" --export=ALL)
[[ "$TEST_ONLY" == 1 ]] && SBATCH_ARGS+=(--test-only)
sbatch "${SBATCH_ARGS[@]}" "$SOURCE_PATH/ray.sub"
