#!/bin/bash
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=convert_super35_megatron
#SBATCH --output=/lustre/fsw/portfolios/cosmos/users/yeyu/logs/convert_super35_%j.log
# Convert the Super 3.5 TextTower from HF to a Megatron checkpoint.
#
# The 16-node MOPD run fails in Megatron's ON-THE-FLY HF conversion, on expert
# _extra_state shards. NeMo-RL supports loading a pre-converted checkpoint instead
# (policy.pretrained_checkpoint, format megatron_bridge), and Megatron-Bridge has a
# nemotron_h bridge that handles this architecture's MTP flattening and expert indices.
#
# Topology: the converter requires nodes*gpus % (TP*PP) == 0 and % (ETP*EP*PP) == 0.
# With the recipe's TP=2 PP=1 EP=32 ETP=1 that is 32 ranks, i.e. 8 nodes x 4 GPUs here.
set -x
RL_DIR=${RL_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/RL}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/cosmos/users/yeyu/containers/nemo-rl-v0.8.0.sqsh}
HF_MODEL=${HF_MODEL:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_modelopt/hf-local/nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-TextTower}
OUT=${OUT:-/lustre/fsw/portfolios/cosmos/users/yeyu/super35_texttower_megatron}
CONV_DIR="${RL_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/scripts/conversion"

cat > "${RL_DIR}/.convert_inner.sh" <<'INNER'
set -eux
# Run from the NeMo-RL project root so `uv run` resolves its environment, but keep the
# conversion scripts importable: run_conversion.py imports sibling modules (cpu_backend,
# gpu_backend, arguments) by bare name.
cd "$RL_DIR"
export PYTHONPATH="${CONV_DIR}:${PYTHONPATH:-}"
export UV_CACHE_DIR=/lustre/fsw/portfolios/cosmos/users/yeyu/.uv-cache
export UV_LOCK_TIMEOUT=${UV_LOCK_TIMEOUT:-5400}
# megatron lives behind the `mcore` extra; the image's plain python has no megatron at
# all, which is what the first attempt died on.
python -c "from nemo.lens import span_attributes" 2>/dev/null \
  || uv pip install --python python \
       "nemo-lens[sdk,aiohttp,fastapi] @ git+https://github.com/NVIDIA-NeMo/Lens.git@4d045c03186b90afdc815db46b671f6a21bc7e98"
# run_conversion.py reads WORLD_SIZE/SLURM_NTASKS and the usual rendezvous vars.
uv run --locked --extra mcore python "${CONV_DIR}/run_conversion.py" import \
  --device gpu \
  --hf-model "$HF_MODEL" \
  --megatron-path "$OUT" \
  --tp 2 --pp 1 --ep 32 --etp 1 \
  --torch-dtype bfloat16 \
  --trust-remote-code \
  --low-memory-save \
  --overwrite
INNER
chmod +x "${RL_DIR}/.convert_inner.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29591

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre,/scratch:/scratch \
     --container-workdir="$CONV_DIR" \
     --export=ALL,CONV_DIR="$CONV_DIR",HF_MODEL="$HF_MODEL",OUT="$OUT",MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT" \
     bash "${RL_DIR}/.convert_inner.sh"
