#!/bin/bash
#SBATCH --job-name=prefetch-gym-venvs
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=prefetch-gym-venvs-%j.out

set -euo pipefail

# ---- Paths ----
REPO=/lustre/fs1/portfolios/coreai/users/youngeunk/mount/nemo-rl-backup
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/sqsh/nemo_rl.super_v3.sqsh
VENV_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/youngeunk/gym_venvs
CONFIG=examples/configs/recipes/llm/performance/grpo-nemotron-nano-16n8g-async-1off-swe.yaml

mkdir -p "$VENV_DIR"

echo "===== prefetch-gym-venvs ====="
echo "Repo:       $REPO"
echo "Container:  $CONTAINER"
echo "Venv dir:   $VENV_DIR  (bound to /opt/gym_venvs inside container)"
echo "Config:     $CONFIG"
echo "=============================="

# Bind the /lustre venv dir to /opt/gym_venvs inside the container, so writes
# from prefetch_venvs.py land on /lustre. Also mount /lustre/fs1 and /lustre/fsw
# (separate Lustre filesystems) and the repo path.
srun --container-image="$CONTAINER" \
     --container-mounts="/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw,${VENV_DIR}:/opt/gym_venvs" \
     --container-workdir="$REPO" \
     --no-container-mount-home \
     bash -xc '
       set -euo pipefail
       export NEMO_GYM_VENV_DIR=/opt/gym_venvs
       export UV_LINK_MODE=hardlink

       # Pick the right torch backend (cuda12x or whatever the pyproject pins).
       UV_TORCH_BACKEND=$(uv run python -c "
import tomllib, pathlib
indexes = tomllib.loads(pathlib.Path(\"pyproject.toml\").read_text())[\"tool\"][\"uv\"][\"index\"]
print(next(i[\"name\"].removeprefix(\"pytorch-\") for i in indexes if i[\"name\"].startswith(\"pytorch-\")))
")
       export UV_TORCH_BACKEND
       echo "UV_TORCH_BACKEND=$UV_TORCH_BACKEND"

       uv run python examples/nemo_gym/prefetch_venvs.py '"$CONFIG"'

       echo "===== Resulting venv tree (top level) ====="
       ls -la /opt/gym_venvs/ || true
       du -sh /opt/gym_venvs/* 2>/dev/null || true
     '

echo "Done. Venvs at: $VENV_DIR"
