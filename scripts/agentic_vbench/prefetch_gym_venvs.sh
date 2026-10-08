#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build the Gym server venvs on a CPU node inside the training container, into the
# same NEMO_GYM_VENV_DIR the launcher uses, so GPU jobs start rollouts sooner.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
container=${CONTAINER:-/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh}
export AVB_DATA_ROOT=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
# Same directory run_sc.sh hands to the job (GYM_VENV_DIR -> NEMO_GYM_VENV_DIR).
gym_venv_dir=${GYM_VENV_DIR:-$AVB_DATA_ROOT/gym_venvs}
mkdir -p "$gym_venv_dir"
mounts="/lustre:/lustre,/scratch:/scratch"
for sub in nemo_rl examples tools scripts 3rdparty/Gym-workspace/Gym 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge; do
  mounts+=",$code_dir/$sub:/opt/nemo-rl/$sub"
done
# The dry run instantiates the agent server, which validates its inputs: the remote
# backend must be reachable (client.env + docker CLI), so point it at a live backend.
export AGENTIC_VBENCH_HARBOR_PYTHON=${AGENTIC_VBENCH_HARBOR_PYTHON:-$AVB_DATA_ROOT/harbor-venv/bin/python}
export AGENTIC_VBENCH_OUTPUT_ROOT=${AGENTIC_VBENCH_OUTPUT_ROOT:-$AVB_DATA_ROOT/episodes}
export AGENTIC_VBENCH_DOCKER_CLI_DIR=${AGENTIC_VBENCH_DOCKER_CLI_DIR:-$AVB_DATA_ROOT/docker-client}
export AGENTIC_VBENCH_BACKEND=${AGENTIC_VBENCH_BACKEND:-remote}
export AGENTIC_VBENCH_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-$AVB_DATA_ROOT/backend-cc-smoke}
[[ -s $AGENTIC_VBENCH_BACKEND_DIR/client.env ]] || { echo "No task backend: $AGENTIC_VBENCH_BACKEND_DIR/client.env (submit_backend.sh)" >&2; exit 1; }
export TRAIN_PATH=${TRAIN_PATH:-/dev/null}
export GYM_VENV_DIR_IN_JOB=$gym_venv_dir
srun --account="${SLURM_ACCOUNT:-nemotron_n4_omni}" --partition="${PREFETCH_PARTITION:-cpu}" --nodes=1 --ntasks=1 \
  --cpus-per-task=16 --mem=64G --time=01:00:00 --job-name=nrlcc-gym-venvs \
  --container-image="$container" --container-mounts="$mounts" --container-workdir=/opt/nemo-rl \
  --export=ALL,NRL_CONTAINER=1,NRL_IGNORE_VERSION_MISMATCH=1,UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv,RAY_USAGE_STATS_ENABLED=0,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0,NEMO_RL_VENV_DIR=/opt/ray_venvs,NEMO_GYM_VENV_DIR="$gym_venv_dir" \
  bash -c 'set -euo pipefail; cd /opt/nemo-rl; export NEMO_GYM_VENV_DIR="$GYM_VENV_DIR_IN_JOB"; export PYTHONPATH=/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:${PYTHONPATH:-}; export RAY_TMPDIR=/tmp/ray-gym-prefetch-$$; mkdir -p "$RAY_TMPDIR"; uv run --no-sync python examples/nemo_gym/prefetch_venvs.py scripts/agentic_vbench/prefetch_gym_venvs.yaml; ls -la "$NEMO_GYM_VENV_DIR"; echo GYM_VENVS_OK'
