#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Re-sync every venv of an existing image to THIS checkout's uv.lock (no dependency
# rebuild), then export it. Use after a lock-only change such as aligning Ray.
#   CODE_DIR=$PWD sbatch --export=ALL,CODE_DIR=$PWD --account=nemotron_n4_omni --partition=cpu \
#     --cpus-per-task=32 --mem=400G --time=03:00:00 scripts/agentic_vbench/container/patch_container.sbatch
set -euo pipefail
code_dir=$(cd "${CODE_DIR:-$SLURM_SUBMIT_DIR}" && pwd)
[[ -f $code_dir/pyproject.toml && -d $code_dir/nemo_rl ]] || { echo "CODE_DIR is not a NeMo-RL checkout: $code_dir" >&2; exit 2; }
input=${INPUT_SQSH:-/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh}
output=${OUTPUT_SQSH:-$input}
[[ -f $input ]] || { echo "input image not found: $input" >&2; exit 1; }
name=nrlcc-patch-$$
scratch=${BUILD_SCRATCH:-/tmp/$USER/nrlcc-patch-$$}
export ENROOT_DATA_PATH=$scratch/data ENROOT_CACHE_PATH=$scratch/cache ENROOT_TEMP_PATH=$scratch/tmp ENROOT_RUNTIME_PATH=$scratch/runtime
export ENROOT_MOUNT_HOME=no
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_TEMP_PATH" "$ENROOT_RUNTIME_PATH"
trap 'enroot remove -f "$name" 2>/dev/null || true; rm -rf "$scratch"' EXIT
echo "host=$(hostname) code_dir=$code_dir input=$input output=$output"
echo "rl_commit=$(git -C "$code_dir" rev-parse HEAD) gym_commit=$(git -C "$code_dir/3rdparty/Gym-workspace/Gym" rev-parse HEAD)"
enroot create --name "$name" "$input"
enroot start --root --rw --mount "$code_dir:/mnt/nemo-rl-src:rbind,ro,x-create=dir" "$name" bash -exu -o pipefail -c '
export HOME=/root UV_CACHE_DIR=/root/.cache/uv UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv UV_PYTHON_INSTALL_DIR=/opt/uv/python
export NEMO_RL_VENV_DIR=/opt/ray_venvs NRL_CONTAINER=1 RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export TORCH_CUDA_ARCH_LIST="9.0 10.0" CUDA_HOME=/usr/local/cuda CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
cd /opt/nemo-rl
# Refresh source + lock from the checkout (same excludes as build_inside.sh).
rsync -a --delete --exclude=.venv --exclude=__pycache__ --exclude=results --exclude=logs --exclude=data/agentic_vbench --exclude="*.sqsh" /mnt/nemo-rl-src/ /opt/nemo-rl/
git config --global --add safe.directory "*"
uv lock --check
UV_LINK_MODE=hardlink uv sync --frozen --all-groups
test -s /opt/actor_venvs.tsv
while IFS=$'"'"'\t'"'"' read -r venv_name stage extras; do
  venv_path="$NEMO_RL_VENV_DIR/$venv_name"
  UV_PROJECT_ENVIRONMENT="$venv_path" UV_LINK_MODE=hardlink uv sync --frozen $extras
done < /opt/actor_venvs.tsv
UV_LINK_MODE=hardlink uv run --no-sync nemo_rl/utils/prefetch_venvs.py $(cut -f1 /opt/actor_venvs.tsv)
/opt/nemo_rl_venv/bin/python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
echo "== ray versions =="
for v in /opt/nemo_rl_venv /opt/ray_venvs/*; do printf "%-80s %s\n" "$v" "$($v/bin/python -c "import ray; print(ray.__version__)")"; done
git -C /opt/nemo-rl status --short uv.lock pyproject.toml
echo PATCH_INSIDE_OK'
tmp_output="$output.partial"
rm -f "$tmp_output"
ENROOT_SQUASH_OPTIONS='-noI -noD -noF -noX -no-duplicates -mem 2G' ENROOT_MAX_PROCESSORS=16 enroot export --output "$tmp_output" "$name"
mv "$tmp_output" "$output"
ls -la "$output"
echo "CONTAINER_PATCH_OK $output"
