#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null || die "missing command: $1"; }
require_canonical_lustre_path() {
  local resolved
  resolved=$(realpath -e -- "$2") || die "$1 does not exist: $2"
  [[ "$resolved" == "$2" && "$resolved" == /lustre/* ]] || die "$1 must be a canonical /lustre path: $2"
}
require_canonical_tmp_path() {
  local resolved
  resolved=$(realpath -m -- "$2") || die "$1 is invalid: $2"
  [[ "$resolved" == "$2" && "$resolved" == /tmp/* ]] || die "$1 must be a canonical /tmp path: $2"
}
validate_extra_mounts() {
  local entry host destination remainder
  local -a entries
  [[ -z "$1" ]] && return
  IFS=',' read -r -a entries <<< "$1"
  for entry in "${entries[@]}"; do
    host=${entry%%:*}
    remainder=${entry#*:}
    [[ "$remainder" != "$entry" ]] || die "invalid MOUNTS entry: $entry"
    destination=${remainder%%:*}
    require_canonical_lustre_path MOUNTS "$host"
    [[ "$destination" == /* && "$destination" != /home && "$destination" != /home/* && "$destination" != *'/../'* ]] || die "invalid MOUNTS destination: $destination"
  done
}
directory_tree_sha256() {
  python3 - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
tree_hash = hashlib.sha256()
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == ".tree-sha256":
        continue
    relative_path = path.relative_to(root).as_posix().encode()
    tree_hash.update(len(relative_path).to_bytes(8, "big"))
    tree_hash.update(relative_path)
    file_hash = hashlib.sha256()
    with path.open("rb") as file_object:
        for chunk in iter(lambda: file_object.read(1024 * 1024), b""):
            file_hash.update(chunk)
    tree_hash.update(file_hash.digest())
print(tree_hash.hexdigest())
PY
}

for SBATCH_VARIABLE in ${!SBATCH_@}; do
  unset "$SBATCH_VARIABLE"
done
unset RAY_ADDRESS RAY_NAMESPACE

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python3}
ARM=${ARM:?Set ARM to one of the names emitted by arm_matrix.py --list}
RENDER_ONLY=${RENDER_ONLY:-0}
TEST_ONLY=${TEST_ONLY:-0}
SEGMENT=4

[[ "$RENDER_ONLY" == 0 || "$RENDER_ONLY" == 1 ]] || die "RENDER_ONLY must be 0 or 1"
[[ "$TEST_ONLY" == 0 || "$TEST_ONLY" == 1 ]] || die "TEST_ONLY must be 0 or 1"

IFS=$'\t' read -r ARM_NAME DISPATCHER HYBRIDEP_BACKEND PAD_UNEVEN LEGACY_PREPADDING \
  EXPECTED_DEEPEP_COMMIT SOURCE_PROFILE EXPECTED_NEMO_RL_COMMIT \
  EXPECTED_BRIDGE_COMMIT EXPECTED_MCORE_COMMIT SOURCE_BRANCH CONTAINER \
  CONTAINER_SHA256 PREFLIGHT_MANIFEST_SHA256 RECIPE NODES GPUS_PER_NODE \
  MAX_STEPS < <(
    "$PYTHON" "$SCRIPT_DIR/arm_matrix.py" --arm "$ARM" --format tsv
  )
[[ "$ARM_NAME" == "$ARM" ]] || die "arm matrix resolution failed"

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-padding-ab-q30/cw-h100}
OUTPUT_ROOT=${OUTPUT_ROOT:-$EXPERIMENT_ROOT/$ARM}
SOURCE_PATH=${SOURCE_PATH:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
BATCH_SCRIPT=$SCRIPT_DIR/ray-nonexclusive.sub
ACCOUNT_FOR_RENDER=${ACCOUNT:-ACCOUNT_REQUIRED}
JOB_NAME=${JOB_NAME:-$ACCOUNT_FOR_RENDER:hybridep-q30-$ARM}
WANDB_ENABLED=${WANDB_ENABLED:-true}
WANDB_PROJECT=${WANDB_PROJECT:-sna-hybridep-padding-ab-h100}
WANDB_NAME=${WANDB_NAME:-q30-$ARM}
REQUIRES_DEEPEP_ARTIFACT=$HYBRIDEP_BACKEND
BACKEND_NAME=none
[[ "$HYBRIDEP_BACKEND" == 0 ]] || BACKEND_NAME=hybridep

RUN_ARGS=(uv run --no-sync examples/run_grpo.py --config "$RECIPE"
  "grpo.max_num_steps=$MAX_STEPS" checkpointing.enabled=false
  policy.sequence_packing.enabled=true
  "policy.megatron_cfg.moe_token_dispatcher_type=$DISPATCHER")
if [[ "$HYBRIDEP_BACKEND" == 1 ]]; then
  RUN_ARGS+=(++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep
    ++policy.megatron_cfg.moe_hybridep_num_sms=32
    "++policy.megatron_cfg.moe_hybridep_pad_uneven_dispatch_inputs=$([[ $PAD_UNEVEN == 1 ]] && printf true || printf false)")
fi
if [[ "$LEGACY_PREPADDING" == 1 ]]; then
  RUN_ARGS+=(++policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true)
fi
RUN_ARGS+=("logger.log_dir=$OUTPUT_ROOT/training-__SLURM_JOB_ID__"
  "logger.wandb_enabled=$WANDB_ENABLED" "logger.wandb.project=$WANDB_PROJECT"
  "logger.wandb.name=$WANDB_NAME")
printf -v TRAINING_COMMAND '%q ' "${RUN_ARGS[@]}"
TRAINING_COMMAND=${TRAINING_COMMAND% }

SBATCH_RENDER=(sbatch --nodes="$NODES" --gpus-per-node="$GPUS_PER_NODE" --segment="$SEGMENT"
  --account="$ACCOUNT_FOR_RENDER" --partition=batch --time=02:00:00
  --job-name="$JOB_NAME" --output="$OUTPUT_ROOT/slurm-%j.out"
  --error="$OUTPUT_ROOT/slurm-%j.out" --export=ALL)
[[ "$TEST_ONLY" == 0 ]] || SBATCH_RENDER+=(--test-only)
SBATCH_RENDER+=("$BATCH_SCRIPT")
printf -v SBATCH_COMMAND '%q ' "${SBATCH_RENDER[@]}"
SBATCH_COMMAND=${SBATCH_COMMAND% }

if [[ "$RENDER_ONLY" == 1 ]]; then
  printf 'arm=%s\n' "$ARM"
  printf 'recipe=%s\n' "$RECIPE"
  printf 'nodes=%s\n' "$NODES"
  printf 'gpus_per_node=%s\n' "$GPUS_PER_NODE"
  printf 'segment=%s\n' "$SEGMENT"
  printf 'max_steps=%s\n' "$MAX_STEPS"
  printf 'sequence_packing=1\n'
  printf 'dispatcher=%s\n' "$DISPATCHER"
  printf 'hybridep_backend=%s\n' "$BACKEND_NAME"
  printf 'pad_uneven_dispatch_inputs=%s\n' "$PAD_UNEVEN"
  printf 'legacy_prepadding=%s\n' "$LEGACY_PREPADDING"
  printf 'deepep_commit=%s\n' "$EXPECTED_DEEPEP_COMMIT"
  printf 'requires_deepep_artifact=%s\n' "$REQUIRES_DEEPEP_ARTIFACT"
  printf 'source_profile=%s\n' "$SOURCE_PROFILE"
  printf 'nemo_rl_commit=%s\n' "$EXPECTED_NEMO_RL_COMMIT"
  printf 'bridge_commit=%s\n' "$EXPECTED_BRIDGE_COMMIT"
  printf 'mcore_commit=%s\n' "$EXPECTED_MCORE_COMMIT"
  printf 'source_branch=%s\n' "$SOURCE_BRANCH"
  printf 'container=%s\n' "$CONTAINER"
  printf 'container_sha256=%s\n' "$CONTAINER_SHA256"
  printf 'preflight_manifest_sha256=%s\n' "$PREFLIGHT_MANIFEST_SHA256"
  printf 'batch_script=%s\n' "$BATCH_SCRIPT"
  printf 'sbatch_environment_sanitized=1\n'
  printf 'ray_environment_sanitized=1\n'
  printf 'job_name=%s\n' "$JOB_NAME"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
  printf 'training_command=%s\n' "$TRAINING_COMMAND"
  printf 'sbatch_command=%s\n' "$SBATCH_COMMAND"
  exit 0
fi

for command_name in du flock git mktemp mv python3 realpath sbatch sshare sha256sum stat uv; do
  require_command "$command_name"
done
: "${ACCOUNT:?Set ACCOUNT after checking FairShare immediately before submission}"
SOURCE_REMOTE=fork
MCORE_5008_COMMIT=81770cb015eab05785ecd540ba929d1400a52f67
EXPECTED_GPU_MODEL='NVIDIA H100 80GB HBM3'
EXPECTED_ARCHITECTURE=x86_64
PREFLIGHT_VENV=${PREFLIGHT_VENV:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/hybridep-upstream5008-validation/cw-h100/preflight-venv}

mkdir -p "$OUTPUT_ROOT"
require_canonical_lustre_path SOURCE_PATH "$SOURCE_PATH"
require_canonical_lustre_path OUTPUT_ROOT "$OUTPUT_ROOT"
require_canonical_lustre_path PREFLIGHT_VENV "$PREFLIGHT_VENV"
GIT_COMMON_DIR=$(git -C "$SOURCE_PATH" rev-parse --path-format=absolute --git-common-dir)
require_canonical_lustre_path GIT_COMMON_DIR "$GIT_COMMON_DIR"
[[ -f "$SOURCE_PATH/$RECIPE" && -f "$SOURCE_PATH/ray.sub" && -f "$BATCH_SCRIPT" ]] || die "invalid source or batch script"
CUDNN_HOST_PATH=$PREFLIGHT_VENV/lib/python3.13/site-packages/nvidia/cudnn
require_canonical_lustre_path CUDNN_HOST_PATH "$CUDNN_HOST_PATH"
URLLIB3_HOST_PATH=${URLLIB3_HOST_PATH:-$PREFLIGHT_VENV/lib/python3.13/site-packages/urllib3}
require_canonical_lustre_path URLLIB3_HOST_PATH "$URLLIB3_HOST_PATH"
[[ -z $(git -C "$SOURCE_PATH" status --porcelain --untracked-files=all) ]] || die "NeMo-RL source is dirty"
[[ -z $(git -C "$SOURCE_PATH" submodule foreach --recursive --quiet 'dirty=$(git status --porcelain --untracked-files=all); if [ -n "$dirty" ]; then printf "%s\n" "$displaypath"; fi') ]] || die "recursive submodule source is dirty"
! git -C "$SOURCE_PATH" submodule status --recursive | grep -Eq '^[+-U]' || die "recursive submodule checkout mismatch"
HARNESS_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
require_canonical_lustre_path HARNESS_ROOT "$HARNESS_ROOT"
[[ -z $(git -C "$HARNESS_ROOT" status --porcelain --untracked-files=all) ]] || die "experiment harness is dirty"
HARNESS_COMMIT=$(git -C "$HARNESS_ROOT" rev-parse HEAD)
LAUNCHER_SHA256=$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)
BATCH_SCRIPT_SHA256=$(sha256sum "$BATCH_SCRIPT" | cut -d' ' -f1)
MATRIX_SHA256=$(sha256sum "$SCRIPT_DIR/arm_matrix.py" | cut -d' ' -f1)
LOCAL_HEAD=$(git -C "$SOURCE_PATH" rev-parse HEAD)
[[ "$LOCAL_HEAD" == "$EXPECTED_NEMO_RL_COMMIT" ]] || die "NeMo-RL commit mismatch"
PUSHED_HEAD=$(git -C "$SOURCE_PATH" ls-remote "$SOURCE_REMOTE" "refs/heads/$SOURCE_BRANCH" | cut -f1)
[[ "$PUSHED_HEAD" == "$EXPECTED_NEMO_RL_COMMIT" ]] || die "frozen source branch was not pushed"

BRIDGE=$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE=$BRIDGE/3rdparty/Megatron-LM
[[ $(git -C "$BRIDGE" rev-parse HEAD) == "$EXPECTED_BRIDGE_COMMIT" ]] || die "Megatron-Bridge commit mismatch"
[[ $(git -C "$MCORE" rev-parse HEAD) == "$EXPECTED_MCORE_COMMIT" ]] || die "Megatron-Core commit mismatch"
if [[ "$SOURCE_PROFILE" == official ]]; then
  git -C "$MCORE" merge-base --is-ancestor "$MCORE_5008_COMMIT" HEAD || die "Megatron-Core PR 5008 is absent"
else
  [[ "$LOCAL_HEAD" == d833180b9847daedafedaed6d7d1da6a013f14d0 ]] || die "legacy NeMo pre-padding commit is absent"
fi

CONTAINER_STAT_FINGERPRINT=remote-digest
CONTAINER_CHECKSUM_CACHE=remote-digest
CONTAINER_CHECKSUM_MODE=remote-digest
if [[ -f "$CONTAINER" ]]; then
  require_canonical_lustre_path CONTAINER "$CONTAINER"
  CONTAINER_STAT_FINGERPRINT=$(stat --printf='%d:%i:%s:%Y:%Z' "$CONTAINER")
  CONTAINER_CHECKSUM_CACHE_DIR=${CONTAINER_CHECKSUM_CACHE_DIR:-$EXPERIMENT_ROOT/checksum-cache}
  mkdir -p "$CONTAINER_CHECKSUM_CACHE_DIR"
  require_canonical_lustre_path CONTAINER_CHECKSUM_CACHE_DIR "$CONTAINER_CHECKSUM_CACHE_DIR"
  CONTAINER_CACHE_KEY=$(printf '%s\t%s\t%s\n' "$CONTAINER" "$CONTAINER_STAT_FINGERPRINT" "$CONTAINER_SHA256" | sha256sum | cut -d' ' -f1)
  CONTAINER_CHECKSUM_CACHE=$CONTAINER_CHECKSUM_CACHE_DIR/$CONTAINER_CACHE_KEY.tsv
  CACHED_CONTAINER_PATH=
  CACHED_CONTAINER_STAT=
  CACHED_CONTAINER_SHA256=
  if [[ -f "$CONTAINER_CHECKSUM_CACHE" ]]; then
    IFS=$'\t' read -r CACHED_CONTAINER_PATH CACHED_CONTAINER_STAT CACHED_CONTAINER_SHA256 < "$CONTAINER_CHECKSUM_CACHE" || true
  fi
  if [[ "$CACHED_CONTAINER_PATH" == "$CONTAINER" && "$CACHED_CONTAINER_STAT" == "$CONTAINER_STAT_FINGERPRINT" && "$CACHED_CONTAINER_SHA256" == "$CONTAINER_SHA256" ]]; then
    CONTAINER_CHECKSUM_MODE=cache-hit
  else
    ACTUAL_CONTAINER_SHA256=$(sha256sum "$CONTAINER" | cut -d' ' -f1)
    [[ "$ACTUAL_CONTAINER_SHA256" == "$CONTAINER_SHA256" ]] || die "container checksum mismatch"
    CONTAINER_CHECKSUM_TEMP=$(mktemp "$CONTAINER_CHECKSUM_CACHE.tmp.XXXXXX")
    printf '%s\t%s\t%s\n' "$CONTAINER" "$CONTAINER_STAT_FINGERPRINT" "$ACTUAL_CONTAINER_SHA256" > "$CONTAINER_CHECKSUM_TEMP"
    mv -f "$CONTAINER_CHECKSUM_TEMP" "$CONTAINER_CHECKSUM_CACHE"
    CONTAINER_CHECKSUM_MODE=cache-miss-verified
  fi
elif [[ ! "$CONTAINER" =~ @sha256:[0-9a-f]{64}$ ]]; then
  die "CONTAINER must be a checksum-verified local image or digest-pinned reference"
fi

DEEPEP_WHEEL=none
DEEPEP_METADATA=none
DEEPEP_SHA256=none
DEEPEP_OVERLAY_DIR=none
DEEPEP_OVERLAY_BYTES=0
DEEPEP_OVERLAY_TREE_SHA256=none
if [[ "$REQUIRES_DEEPEP_ARTIFACT" == 1 ]]; then
  if [[ "$EXPECTED_DEEPEP_COMMIT" == 17cfb817bccec3a9c247013360cc550c2bac441e ]]; then
    DEEPEP_WHEEL=${DEEPEP_17CF_WHEEL:?Set DEEPEP_17CF_WHEEL}
    DEEPEP_METADATA=${DEEPEP_17CF_METADATA:?Set DEEPEP_17CF_METADATA}
  else
    DEEPEP_WHEEL=${DEEPEP_F725_WHEEL:?Set DEEPEP_F725_WHEEL}
    DEEPEP_METADATA=${DEEPEP_F725_METADATA:?Set DEEPEP_F725_METADATA}
  fi
  require_canonical_lustre_path DEEPEP_WHEEL "$DEEPEP_WHEEL"
  require_canonical_lustre_path DEEPEP_METADATA "$DEEPEP_METADATA"
  [[ -f "$DEEPEP_WHEEL" && -f "$DEEPEP_METADATA" ]] || die "DeepEP wheel or metadata is missing"
  IFS=$'\t' read -r META_COMMIT META_PLATFORM META_ARCH META_WHEEL META_SHA < <(
    python3 - "$DEEPEP_METADATA" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)
keys = ("commit", "platform", "architecture", "wheel", "sha256")
values = []
for key in keys:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"missing DeepEP metadata field: {key}")
    values.append(value)
print("\t".join(values))
PY
  )
  [[ "$META_COMMIT" == "$EXPECTED_DEEPEP_COMMIT" && "$META_PLATFORM" == linux && "$META_ARCH" == "$EXPECTED_ARCHITECTURE" ]] || die "DeepEP metadata platform or commit mismatch"
  [[ "$META_WHEEL" == "$DEEPEP_WHEEL" || "$META_WHEEL" == "$(basename "$DEEPEP_WHEEL")" ]] || die "DeepEP metadata wheel mismatch"
  DEEPEP_SHA256=$(sha256sum "$DEEPEP_WHEEL" | cut -d' ' -f1)
  [[ "$DEEPEP_SHA256" == "$META_SHA" ]] || die "DeepEP wheel checksum mismatch"

  DEEPEP_OVERLAY_ROOT="$EXPERIMENT_ROOT/artifacts/deepep-overlays"
  mkdir -p "$DEEPEP_OVERLAY_ROOT"
  require_canonical_lustre_path DEEPEP_OVERLAY_ROOT "$DEEPEP_OVERLAY_ROOT"
  DEEPEP_OVERLAY_DIR="$DEEPEP_OVERLAY_ROOT/$DEEPEP_SHA256-tree-v1"
  DEEPEP_OVERLAY_MANIFEST="$DEEPEP_OVERLAY_DIR/.wheel-sha256"
  DEEPEP_OVERLAY_LOCK="$DEEPEP_OVERLAY_ROOT/.$DEEPEP_SHA256.lock"
  exec 9>"$DEEPEP_OVERLAY_LOCK"
  flock 9
  if [[ ! -d "$DEEPEP_OVERLAY_DIR" ]]; then
    DEEPEP_OVERLAY_TEMP=$(mktemp -d "$DEEPEP_OVERLAY_ROOT/.$DEEPEP_SHA256.XXXXXX")
    cleanup_deepep_overlay_temp() { rm -rf -- "$DEEPEP_OVERLAY_TEMP"; }
    trap cleanup_deepep_overlay_temp EXIT
    UV_NO_CONFIG=1 uv pip install --python-version 3.13 \
      --python-platform x86_64-unknown-linux-gnu \
      --target "$DEEPEP_OVERLAY_TEMP" --no-deps --reinstall "$DEEPEP_WHEEL"
    compgen -G "$DEEPEP_OVERLAY_TEMP/deep_ep_cpp*.so" >/dev/null || die "DeepEP extension is absent from staged overlay"
    compgen -G "$DEEPEP_OVERLAY_TEMP/hybrid_ep_cpp*.so" >/dev/null || die "HybridEP extension is absent from staged overlay"
    printf '%s\n' "$DEEPEP_SHA256" > "$DEEPEP_OVERLAY_TEMP/.wheel-sha256"
    directory_tree_sha256 "$DEEPEP_OVERLAY_TEMP" > "$DEEPEP_OVERLAY_TEMP/.tree-sha256"
    mv "$DEEPEP_OVERLAY_TEMP" "$DEEPEP_OVERLAY_DIR"
    trap - EXIT
  fi
  flock -u 9
  exec 9>&-
  require_canonical_lustre_path DEEPEP_OVERLAY_DIR "$DEEPEP_OVERLAY_DIR"
  [[ -f "$DEEPEP_OVERLAY_MANIFEST" ]] || die "DeepEP overlay manifest is missing"
  [[ $(<"$DEEPEP_OVERLAY_MANIFEST") == "$DEEPEP_SHA256" ]] || die "DeepEP overlay checksum mismatch"
  [[ -f "$DEEPEP_OVERLAY_DIR/.tree-sha256" ]] || die "DeepEP overlay tree manifest is missing"
  DEEPEP_OVERLAY_TREE_SHA256=$(<"$DEEPEP_OVERLAY_DIR/.tree-sha256")
  [[ $(directory_tree_sha256 "$DEEPEP_OVERLAY_DIR") == "$DEEPEP_OVERLAY_TREE_SHA256" ]] || die "DeepEP overlay tree checksum mismatch"
  compgen -G "$DEEPEP_OVERLAY_DIR/deep_ep_cpp*.so" >/dev/null || die "DeepEP overlay extension is missing"
  compgen -G "$DEEPEP_OVERLAY_DIR/hybrid_ep_cpp*.so" >/dev/null || die "HybridEP overlay extension is missing"
  DEEPEP_OVERLAY_BYTES=$(du -sb "$DEEPEP_OVERLAY_DIR" | cut -f1)
fi

mkdir -p "$OUTPUT_ROOT"
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
FAIRSHARE_LOG=$OUTPUT_ROOT/fairshare-$RUN_STAMP.txt
sshare -A "$ACCOUNT" -u "$USER" -o Cluster,Account,User,FairShare | tee "$FAIRSHARE_LOG"
grep -F "$ACCOUNT" "$FAIRSHARE_LOG" >/dev/null || die "ACCOUNT is absent from FairShare output"

PROVENANCE_ROOT=$OUTPUT_ROOT/provenance-$RUN_STAMP
mkdir -p "$PROVENANCE_ROOT"
printf 'arm=%s\nsource_profile=%s\nnemo_rl_commit=%s\nbridge_commit=%s\nmcore_commit=%s\ndeepep_commit=%s\ndeepep_wheel=%s\ndeepep_sha256=%s\ncontainer=%s\ncontainer_sha256=%s\nrecipe=%s\nmax_steps=%s\n' \
  "$ARM" "$SOURCE_PROFILE" "$LOCAL_HEAD" "$EXPECTED_BRIDGE_COMMIT" "$EXPECTED_MCORE_COMMIT" \
  "$EXPECTED_DEEPEP_COMMIT" "$DEEPEP_WHEEL" "$DEEPEP_SHA256" "$CONTAINER" "${CONTAINER_SHA256:-digest-pinned}" "$RECIPE" "$MAX_STEPS" \
  > "$PROVENANCE_ROOT/submission.txt"
printf 'source_branch=%s\npreflight_manifest_sha256=%s\nbatch_script=%s\n' \
  "$SOURCE_BRANCH" "$PREFLIGHT_MANIFEST_SHA256" "$BATCH_SCRIPT" \
  >> "$PROVENANCE_ROOT/submission.txt"
printf 'git_common_dir=%s\n' "$GIT_COMMON_DIR" >> "$PROVENANCE_ROOT/submission.txt"
printf 'cudnn_host_path=%s\ncudnn_container_path=%s\n' \
  "$CUDNN_HOST_PATH" \
  "/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn" \
  >> "$PROVENANCE_ROOT/submission.txt"
printf 'urllib3_host_path=%s\nurllib3_container_path=%s\n' \
  "$URLLIB3_HOST_PATH" \
  "/opt/nemo_rl_venv/lib/python3.13/site-packages/urllib3" \
  >> "$PROVENANCE_ROOT/submission.txt"
printf 'container_stat_fingerprint=%s\ncontainer_checksum_cache=%s\ncontainer_checksum_mode=%s\n' \
  "$CONTAINER_STAT_FINGERPRINT" "$CONTAINER_CHECKSUM_CACHE" "$CONTAINER_CHECKSUM_MODE" \
  >> "$PROVENANCE_ROOT/submission.txt"
printf 'harness_commit=%s\nlauncher_sha256=%s\nbatch_script_sha256=%s\nmatrix_sha256=%s\n' \
  "$HARNESS_COMMIT" "$LAUNCHER_SHA256" "$BATCH_SCRIPT_SHA256" "$MATRIX_SHA256" \
  >> "$PROVENANCE_ROOT/submission.txt"
printf 'deepep_overlay_dir=%s\ndeepep_overlay_bytes=%s\ndeepep_overlay_tree_sha256=%s\n' \
  "$DEEPEP_OVERLAY_DIR" "$DEEPEP_OVERLAY_BYTES" "$DEEPEP_OVERLAY_TREE_SHA256" \
  >> "$PROVENANCE_ROOT/submission.txt"

export SOURCE_PATH OUTPUT_ROOT PROVENANCE_ROOT RECIPE MAX_STEPS ARM SOURCE_PROFILE
export EXPECTED_NEMO_RL_COMMIT EXPECTED_BRIDGE_COMMIT EXPECTED_MCORE_COMMIT
export EXPECTED_DEEPEP_COMMIT DEEPEP_WHEEL DEEPEP_METADATA DEEPEP_SHA256
export DEEPEP_OVERLAY_DIR DEEPEP_OVERLAY_BYTES DEEPEP_OVERLAY_TREE_SHA256
export EXPECTED_GPU_MODEL GPUS_PER_NODE DISPATCHER HYBRIDEP_BACKEND PAD_UNEVEN LEGACY_PREPADDING
export HF_HOME=${HF_CACHE:-$EXPERIMENT_ROOT/hf-cache}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-$EXPERIMENT_ROOT/uv-cache}
export NRL_NODE_LOCAL_UV_CACHE_DIR=${NRL_NODE_LOCAL_UV_CACHE_DIR:-/tmp/nemo-rl-uv-cache-$ARM}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/tmp/nemo-rl-venvs-$ARM-$LOCAL_HEAD}
export CACHE_ROOT=$EXPERIMENT_ROOT/caches
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export XDG_CACHE_HOME=$CACHE_ROOT/xdg
export TORCH_HOME=$CACHE_ROOT/torch
export WANDB_CACHE_DIR=$CACHE_ROOT/wandb
export TRITON_CACHE_DIR=/tmp/nemo-rl-triton-$ARM-$LOCAL_HEAD
export CUDA_CACHE_PATH=/tmp/nemo-rl-cuda-cache-$ARM-$LOCAL_HEAD
export CUDNN_CONTAINER_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_HOME=$CUDNN_CONTAINER_PATH
export CUDNN_PATH=$CUDNN_HOME
export URLLIB3_CONTAINER_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/urllib3
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$CUDNN_CONTAINER_PATH/lib:/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BRIDGE_SOURCE=$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src
export MCORE_SOURCE=$SOURCE_PATH/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
export PYTHONPATH="$SOURCE_PATH:$BRIDGE_SOURCE:$MCORE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_DATASETS_CACHE" "$UV_CACHE_DIR_OVERRIDE" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$WANDB_CACHE_DIR"
require_canonical_lustre_path HF_HOME "$HF_HOME"
require_canonical_lustre_path HF_DATASETS_CACHE "$HF_DATASETS_CACHE"
require_canonical_lustre_path UV_CACHE_DIR_OVERRIDE "$UV_CACHE_DIR_OVERRIDE"
require_canonical_lustre_path CACHE_ROOT "$CACHE_ROOT"
require_canonical_lustre_path PIP_CACHE_DIR "$PIP_CACHE_DIR"
require_canonical_lustre_path XDG_CACHE_HOME "$XDG_CACHE_HOME"
require_canonical_lustre_path TORCH_HOME "$TORCH_HOME"
require_canonical_lustre_path WANDB_CACHE_DIR "$WANDB_CACHE_DIR"
require_canonical_tmp_path NRL_NODE_LOCAL_UV_CACHE_DIR "$NRL_NODE_LOCAL_UV_CACHE_DIR"
require_canonical_tmp_path NEMO_RL_VENV_DIR "$NEMO_RL_VENV_DIR"
require_canonical_tmp_path TRITON_CACHE_DIR "$TRITON_CACHE_DIR"
require_canonical_tmp_path CUDA_CACHE_PATH "$CUDA_CACHE_PATH"
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NVLINK_DOMAIN_SIZE=8 USE_MNNVL=0
export CONTAINER

EXTRA_MOUNTS=${MOUNTS:-}
validate_extra_mounts "$EXTRA_MOUNTS"
MOUNTS_VALUE="$SOURCE_PATH:$SOURCE_PATH,$OUTPUT_ROOT:$OUTPUT_ROOT,$HF_HOME:$HF_HOME,$CACHE_ROOT:$CACHE_ROOT,$CUDNN_HOST_PATH:$CUDNN_CONTAINER_PATH,$URLLIB3_HOST_PATH:$URLLIB3_CONTAINER_PATH"
if [[ "$GIT_COMMON_DIR" != "$SOURCE_PATH" && "$GIT_COMMON_DIR" != "$SOURCE_PATH/"* ]]; then
  MOUNTS_VALUE="$MOUNTS_VALUE,$GIT_COMMON_DIR:$GIT_COMMON_DIR"
fi
if [[ "$REQUIRES_DEEPEP_ARTIFACT" == 1 ]]; then
  DEEPEP_DIR=$(dirname "$DEEPEP_WHEEL")
  export PYTHONPATH="$DEEPEP_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR/deep_ep${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  MOUNTS_VALUE="$MOUNTS_VALUE,$DEEPEP_DIR:$DEEPEP_DIR,$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR"
fi
export MOUNTS="$MOUNTS_VALUE${EXTRA_MOUNTS:+,$EXTRA_MOUNTS}"
export BASE_LOG_DIR="$OUTPUT_ROOT"
export PREFLIGHT_MANIFEST_SHA256
if [[ "$LEGACY_PREPADDING" == 1 ]]; then
  export NEMO_RL_HYBRIDEP_LOG_PACKING=1
  export NEMO_RL_HYBRIDEP_LOG_PACKING_MAX_CALLS=32
fi

read -r -d '' SETUP_COMMAND <<'SETUP' || true
set -euo pipefail
cd "$SOURCE_PATH"
RUN_PYTHON=/opt/nemo_rl_venv/bin/python
[[ $("$RUN_PYTHON" -c 'import platform; print(platform.python_version())') == 3.13.14 ]]
"$RUN_PYTHON" - <<'PY'
import importlib.util
import os
from pathlib import Path

import ray
import requests
import urllib3
import urllib3.exceptions

assert Path(urllib3.__file__).resolve().is_relative_to(
    Path(os.environ["URLLIB3_CONTAINER_PATH"]).resolve()
)
spec = importlib.util.find_spec("uvloop")
assert spec is None or hasattr(__import__("uvloop"), "install")
PY
VENV_MANIFEST="$PROVENANCE_ROOT/venv-$(hostname).txt"
env -u PYTHONPATH "$RUN_PYTHON" -m pip freeze | LC_ALL=C sort > "$VENV_MANIFEST"
[[ $(sha256sum "$VENV_MANIFEST" | cut -d' ' -f1) == "$PREFLIGHT_MANIFEST_SHA256" ]]
GPU_MODELS=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ $(printf '%s\n' "$GPU_MODELS" | sed '/^$/d' | wc -l) -eq "$GPUS_PER_NODE" ]]
[[ -z $(printf '%s\n' "$GPU_MODELS" | sed '/^$/d' | grep -Fvx "$EXPECTED_GPU_MODEL") ]]
if [[ "$HYBRIDEP_BACKEND" == 1 ]]; then
  [[ $(sha256sum "$DEEPEP_WHEEL" | cut -d' ' -f1) == "$DEEPEP_SHA256" ]]
  [[ $(<"$DEEPEP_OVERLAY_DIR/.wheel-sha256") == "$DEEPEP_SHA256" ]]
  [[ $(<"$DEEPEP_OVERLAY_DIR/.tree-sha256") == "$DEEPEP_OVERLAY_TREE_SHA256" ]]
  "$RUN_PYTHON" - <<'PY'
import os
from pathlib import Path

import deep_ep, deep_ep_cpp, hybrid_ep_cpp

overlay = Path(os.environ["DEEPEP_OVERLAY_DIR"]).resolve()
for module in (deep_ep, deep_ep_cpp, hybrid_ep_cpp):
    assert Path(module.__file__).resolve().is_relative_to(overlay)
PY
fi
SETUP
export SETUP_COMMAND

read -r -d '' COMMAND <<'DRIVER' || true
set -euo pipefail
: "${NRL_MATRIX_JOB_ID:?NRL_MATRIX_JOB_ID is required}"
cd "$SOURCE_PATH"
[[ $(git rev-parse HEAD) == "$EXPECTED_NEMO_RL_COMMIT" ]]
[[ -z $(git status --porcelain --untracked-files=all) ]]
BRIDGE=3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
MCORE=$BRIDGE/3rdparty/Megatron-LM
[[ $(git -C "$BRIDGE" rev-parse HEAD) == "$EXPECTED_BRIDGE_COMMIT" ]]
[[ $(git -C "$MCORE" rev-parse HEAD) == "$EXPECTED_MCORE_COMMIT" ]]

uv run --no-sync python - <<'PY'
import os
from types import SimpleNamespace

from nemo_rl.models.megatron.setup import _apply_moe_config

dispatcher = os.environ["DISPATCHER"]
hybridep = os.environ["HYBRIDEP_BACKEND"] == "1"
expected_padding = os.environ["PAD_UNEVEN"] == "1"
megatron_cfg = {
    "expert_tensor_parallel_size": 1,
    "expert_model_parallel_size": 8,
    "moe_router_dtype": "float32",
    "moe_router_load_balancing_type": "none",
    "moe_router_bias_update_rate": 0.0,
    "moe_permute_fusion": True,
    "moe_enable_deepep": False,
    "moe_token_dispatcher_type": dispatcher,
    "moe_shared_expert_overlap": True,
}
if hybridep:
    megatron_cfg.update(
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_num_sms=32,
        moe_hybridep_pad_uneven_dispatch_inputs=expected_padding,
    )
if os.environ["LEGACY_PREPADDING"] == "1":
    megatron_cfg["moe_hybridep_prepad_packed_inputs"] = True
model_cfg = SimpleNamespace(moe_hybridep_pad_uneven_dispatch_inputs=False)
_apply_moe_config(model_cfg, {"megatron_cfg": megatron_cfg})
assert model_cfg.moe_hybridep_pad_uneven_dispatch_inputs is expected_padding
if os.environ["LEGACY_PREPADDING"] == "1":
    from nemo_rl.models.megatron.data import get_hybridep_prepadding_contract

    assert get_hybridep_prepadding_contract() == {
        "enabled": True,
        "mcore_router_masks_padding": True,
    }
PY

if [[ "$HYBRIDEP_BACKEND" == 1 ]]; then
  uv run --no-sync python - <<'PY'
import os
from pathlib import Path

import deep_ep
import deep_ep_cpp
import hybrid_ep_cpp

overlay = Path(os.environ["DEEPEP_OVERLAY_DIR"]).resolve()
for module in (deep_ep, deep_ep_cpp, hybrid_ep_cpp):
    assert Path(module.__file__).resolve().is_relative_to(overlay)
PY
fi

TRAINING_COMMAND=${TRAINING_COMMAND//__SLURM_JOB_ID__/$NRL_MATRIX_JOB_ID}
eval "$TRAINING_COMMAND" 2>&1 | tee "$OUTPUT_ROOT/training-$NRL_MATRIX_JOB_ID.log"
DRIVER
export COMMAND TRAINING_COMMAND WANDB_ENABLED WANDB_PROJECT WANDB_NAME

SBATCH_ARGS=(--nodes="$NODES" --gpus-per-node="$GPUS_PER_NODE" --segment="$SEGMENT"
  --account="$ACCOUNT" --partition=batch --time=02:00:00
  --job-name="$JOB_NAME" --output="$OUTPUT_ROOT/slurm-%j.out"
  --error="$OUTPUT_ROOT/slurm-%j.out" --export=ALL)
[[ "$TEST_ONLY" == 0 ]] || SBATCH_ARGS+=(--test-only)
sbatch "${SBATCH_ARGS[@]}" "$BATCH_SCRIPT"
