#!/bin/bash
#SBATCH --account=coreai_dlalgo_modelopt
#SBATCH --partition=batch
#SBATCH --nodes=1
# Whole node: this partition allocates all 4 GPUs per node and rejects a request for
# one as a size-limit violation. The smoke itself uses a single GPU.
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --job-name=lilicorr_smoke
#SBATCH --output=/lustre/fsw/portfolios/cosmos/users/yeyu/logs/lilicorr_smoke_%j.log
# One GPU, no Ray: exercise the drafter path while the 16-node pipeclean queues.
set -x
RL_DIR=${RL_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/RL}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/cosmos/users/yeyu/containers/nemo-rl-v0.8.0.sqsh}
WORKDIR=/lustre/fsw/portfolios/cosmos/users/yeyu/lilicorr_smoke

cat > "${RL_DIR}/.smoke_inner.sh" <<'INNER'
set -eux
cd "$RL_DIR"
export PYTHONPATH="${RL_DIR}${PYTHONPATH:+:$PYTHONPATH}"
# The image venv ships no megatron: it is installed by resolving this branch's lock,
# which builds Megatron-LM from a git dependency. Hence uv run here, as in the pipeclean.
# Persist uv's cache on shared storage. The image's cache lives inside the container,
# so every attempt re-downloads and rebuilds torch, ray and the CUDA wheels -- ~25
# minutes per run, which dominates the iteration loop when debugging a pipeclean.
# A shared cache means concurrent uv processes contend on per-package locks, and the
# default timeout is short enough that a job dies while another is mid-build:
#   "Could not acquire lock for .../flash-attn/2.8.1/.lock, is another uv process
#    running? You can set UV_LOCK_TIMEOUT to increase the timeout."
# Waiting is the right behaviour -- the waiter then reuses the wheel the holder built,
# which is the whole point of sharing the cache. flash-attn alone can take ~1h.
export UV_LOCK_TIMEOUT=${UV_LOCK_TIMEOUT:-5400}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/.uv-cache}
mkdir -p "$UV_CACHE_DIR"
UV=uv
if ! $UV lock --check >/dev/null 2>&1 && ! $UV tree --quiet >/dev/null 2>&1; then
  UVH=/lustre/fsw/portfolios/cosmos/users/yeyu/.smoke-uv
  mkdir -p "$UVH"
  curl -LsSf https://astral.sh/uv/0.11.28/install.sh | env UV_UNMANAGED_INSTALL="$UVH" sh
  UV="$UVH/uv"
fi
"$UV" --version
# megatron-core lives behind the `mcore` extra; a bare `uv run` has no megatron.
# The smoke imports megatron in-process, so it DOES need mcore. It must not
# also ask for vllm: pyproject declares those extras mutually conflicting.
EXTRAS="--locked --extra mcore"
"$UV" run $EXTRAS python -c "import megatron, nemo_rl; print('[smoke] megatron ok')"
"$UV" run $EXTRAS python examples/lilicorr_smoke.py --workdir "$WORKDIR"
INNER
chmod +x "${RL_DIR}/.smoke_inner.sh"

srun --container-image="$CONTAINER" \
     --container-mounts=/lustre:/lustre,/scratch:/scratch \
     --container-workdir="$RL_DIR" \
     --export=ALL,RL_DIR="$RL_DIR",WORKDIR="$WORKDIR" \
     bash "${RL_DIR}/.smoke_inner.sh"
