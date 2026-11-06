#!/bin/bash
#SBATCH --job-name=omega-polaris-buffer-resume
#SBATCH --account=nemotron_reason_code
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=02:00:00
set -euo pipefail

# Submit held; register with the shared workflow monitor before releasing.
[[ $# == 2 ]] || { echo 'Required: NEW_RUN SOURCE' >&2; exit 2; }
run_dir=$1
source_dir=$2
user_root=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/siddjain
image="$user_root/containers/nemo-rl-v0.8.0.rc4-amd64.sqsh"
runtime_root="$user_root/nemo-run/output/eiwor/math/qwen3-1.7b/polaris/omega/buffered-rollouts/dfw/attempts/001/dependencies/nemo-rl-rc4-amd64/mcore"
[[ "$run_dir" == /lustre/* && "$source_dir" == /lustre/* ]]
# Do not accidentally resume or replace evidence from a previous attempt.
mkdir -- "$run_dir"
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=128 --mpi=pmix \
    --container-image="$image" --container-mounts="/lustre:/lustre,$source_dir:/opt/nemo-rl" \
    --no-container-mount-home --container-workdir=/opt/nemo-rl \
    bash -seu -o pipefail -s -- "$run_dir" "$runtime_root" <<'CONTAINER'
export SMOKE_ROOT="$1"
export RUNTIME_ROOT="$2"
export UV_PROJECT_ENVIRONMENT="$RUNTIME_ROOT/mcore-venv"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv-cache"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH" PYTHONPATH=/opt/nemo-rl PYTHONDONTWRITEBYTECODE=1
export NRL_IGNORE_VERSION_MISMATCH=1
unset UV_OFFLINE UV_NO_SYNC VIRTUAL_ENV
export NEMO_RL_VENV_DIR=/opt/ray_venvs NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_DEDUP_LOGS=0 WANDB_MODE=disabled OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$SMOKE_ROOT/caches/huggingface"
export NRL_MEGATRON_CHECKPOINT_DIR="$SMOKE_ROOT/caches/megatron-checkpoints"
export TORCHINDUCTOR_CACHE_DIR="$SMOKE_ROOT/caches/torchinductor"
export VLLM_CACHE_ROOT="$SMOKE_ROOT/caches/vllm" TRITON_CACHE_DIR="$SMOKE_ROOT/caches/triton"
# Ray's socket paths must fit AF_UNIX's 107-byte limit. This allocation owns
# its node-local scratch; durable logs/checkpoints are under SMOKE_ROOT.
export RAY_TMPDIR="/tmp/omega-buffer-${SLURM_JOB_ID}"
export TMPDIR="$RAY_TMPDIR/tmp"
mkdir -p "$TMPDIR" "$HF_HOME"
# Reuse the already-prepared Lustre environment; no setup runs on the GPUs.
command=(uv run --frozen --no-sync --extra mcore python -B -u examples/run_grpo.py --config smoke_tests/polaris_buffered_rollouts/config.yaml)
# The same five-update horizon is used in both processes. The native deadline
# requests a save/exit after update1; it does not shrink the collected buffer.
"${command[@]}" checkpointing.checkpoint_must_save_by=00:00:00:01 2>&1 | tee "$SMOKE_ROOT/fresh.log"
"${command[@]}" 2>&1 | tee "$SMOKE_ROOT/resume.log"
CONTAINER
