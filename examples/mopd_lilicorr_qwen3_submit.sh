#!/bin/bash
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --job-name=mopd_lilicorr_qwen3
#SBATCH --output=/lustre/fsw/portfolios/cosmos/users/yeyu/logs/mopd_lilicorr_qwen3_%j.log
# 1-node MOPD + LiLiCorr pipeclean on a Megatron-supported target.
# Generates the synthetic drafter first, then runs ray.sub with the lilicorr recipe.
set -x
RL_DIR=${RL_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/RL}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/cosmos/users/yeyu/containers/nemo-rl-v0.8.0.sqsh}
DRAFTER=${DRAFTER:-/lustre/fsw/portfolios/cosmos/users/yeyu/lilicorr_qwen3_8b_synth}

cat > "${RL_DIR}/.mk_drafter.sh" <<'INNER'
set -eux
cd "$RL_DIR"
export PYTHONPATH="${RL_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR=/lustre/fsw/portfolios/cosmos/users/yeyu/.uv-cache
export UV_LOCK_TIMEOUT=5400
python -c "from nemo.lens import span_attributes" 2>/dev/null \
  || uv pip install --python python \
       "nemo-lens[sdk,aiohttp,fastapi] @ git+https://github.com/NVIDIA-NeMo/Lens.git@4d045c03186b90afdc815db46b671f6a21bc7e98"
if [ ! -f "$DRAFTER/model.safetensors" ]; then
  uv run --locked --extra mcore python examples/make_lilicorr_checkpoint.py --out "$DRAFTER"
else
  echo "drafter already present at $DRAFTER"
fi
INNER
chmod +x "${RL_DIR}/.mk_drafter.sh"

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre,/scratch:/scratch \
     --container-workdir="$RL_DIR" \
     --export=ALL,RL_DIR="$RL_DIR",DRAFTER="$DRAFTER" \
     --nodes=1 --ntasks=1 \
     bash "${RL_DIR}/.mk_drafter.sh"
