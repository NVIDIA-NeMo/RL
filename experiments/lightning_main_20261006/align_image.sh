#!/bin/bash
set -euo pipefail
: "${SOURCE_ARCHIVE:?Set pinned source archive}"
: "${LOCAL_ROOT:?Set node-local build directory}"
source_root="${LOCAL_ROOT}/source"
mkdir -p "$source_root"
tar -xf "$SOURCE_ARCHIVE" -C "$source_root"

# Preserve the image's dependency cache and prebuilt venvs, not stale code.
for tree in nemo_rl examples tools tests .agents; do
  rsync -a --delete "${source_root}/${tree}/" "/opt/nemo-rl/${tree}/"
done
for tree in Automodel-workspace Gym-workspace Megatron-Bridge-workspace; do
  rsync -a --delete "${source_root}/3rdparty/${tree}/" "/opt/nemo-rl/3rdparty/${tree}/"
done
# The archive's manifest records VCS provenance; old image Git metadata does not.
rm -rf /opt/nemo-rl/.git
cp "${source_root}/pyproject.toml" "${source_root}/uv.lock" "${source_root}/.python-version" /opt/nemo-rl/
mkdir -p /opt/nemo-rl/experiments
cp -a "${source_root}/experiments/lightning_main_20261006" /opt/nemo-rl/experiments/
cd /opt/nemo-rl
unset PYTHONPATH PYTHONOPTIMIZE NRL_IGNORE_VERSION_MISMATCH
export UV_CACHE_DIR=/opt/nemo_rl_cache/uv
export UV_LINK_MODE=hardlink MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export UV_HTTP_TIMEOUT=180
export TMPDIR="${LOCAL_ROOT}/tmp"
export RAY_TMPDIR="/tmp/nr${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/inductor"
export TRITON_CACHE_DIR="${LOCAL_ROOT}/triton"
export VLLM_CACHE_ROOT="${LOCAL_ROOT}/vllm"
export FLA_TILELANG=0
export NO_VCS_VERSION=1
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv uv sync --locked --inexact
actors=(
  nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
  nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
  nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
  nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector
  nemo_rl.algorithms.async_utils.ReplayBuffer
  nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
)
for actor in "${actors[@]}"; do
  extras=(--extra vllm)
  case "$actor" in
    *MegatronPolicyWorker) extras=(--extra mcore --extra nemo_gym --group test) ;;
    *VllmGenerationWorker|*VllmAsyncGenerationWorker) extras=(--extra vllm --extra nemo_gym) ;;
  esac
  printf 'Syncing locked actor environment: %s\n' "$actor"
  UV_PROJECT_ENVIRONMENT="/opt/ray_venvs/${actor}" uv sync --locked --inexact "${extras[@]}"
done
printf '%s\n' "${actors[@]}" > /opt/nemo_rl_aligned_actor_envs.txt
for environment in /opt/nemo_rl_venv "${actors[@]/#//opt/ray_venvs/}"; do
  printf '\nEnvironment: %s\n' "$environment"
  uv pip freeze --python "${environment}/bin/python"
done > /opt/nemo_rl_aligned_packages.txt

# A fingerprint is published only after the actual locked syncs succeed.
cp "${SOURCE_ARCHIVE}.fingerprint.json" /opt/nemo_rl_container_fingerprint
cp "${SOURCE_ARCHIVE}.metadata.txt" /opt/nemo_rl_source_manifest.txt
export EXPECTED_FINGERPRINT="${SOURCE_ARCHIVE}.fingerprint.json"
export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
for role in driver policy vllm-sync vllm-async; do
  case "$role" in
    driver) executable=/opt/nemo_rl_venv/bin/python ;;
    policy) executable=/usr/local/bin/python-MegatronPolicyWorker ;;
    vllm-sync) executable=/usr/local/bin/python-VllmGenerationWorker ;;
    vllm-async) executable=/usr/local/bin/python-VllmAsyncGenerationWorker ;;
  esac
  "$executable" experiments/lightning_main_20261006/smoke.py "$role"
  (
    cd "$LOCAL_ROOT"
    SMOKE_SOURCE_ROOT=/opt/nemo-rl env -u PYTHONPATH "$executable" \
      /opt/nemo-rl/experiments/lightning_main_20261006/smoke.py "$role"
  )
done
/usr/local/bin/python-MegatronPolicyWorker experiments/lightning_main_20261006/ray_socket_probe.py
/usr/local/bin/python-MegatronPolicyWorker -m pytest -q --mcore-only \
  tests/unit/models/megatron/test_hybridep_data.py \
  tests/unit/models/megatron/test_group_experts.py::test_build_hf_to_local_param_map_train_side \
  tests/unit/models/policy/test_megatron_worker.py::test_refit_metadata_map_precedes_transport_init
