#!/bin/bash
# Submit the QA-MOPD + LiLiCorr pipeclean on OCI-JHB.
#
# Deliberately a short run: the point is to prove the path end to end -- drafter loads,
# capture feeds it the layers its checkpoint names, draft loss joins the GRPO loss, a
# step completes -- not to produce an RL result. Raise max_num_steps once it is green.
#
# CP is pinned to 1 in the recipe because DFlashBody raises on context_parallel_size > 1
# and LiLiCorr reuses that backbone. The parent runs CP=2, so this roughly doubles
# per-rank activation memory on a 120B; if it OOMs, raise TP/PP rather than re-enabling
# CP, which would just raise at startup.
set -eux

# 16 nodes x 4 GPUs: the recipe's cluster block is 16n4g and JHB nodes carry 4 GPUs.
# Asking sbatch for 8 would either not allocate or leave Ray's view disagreeing with
# cluster.gpus_per_node, which fails after the whole cluster is up.

RL_DIR=${RL_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/RL}
# v0.8.0, not v0.7.0: this branch merges #3945 and #3757, both newer than the v0.7.0
# release, and the v0.7.0 image's uv cannot parse the resulting pyproject.toml. Note
# nemo-rl:latest is referenced in the docs but returns 404 on nvcr.io -- the published
# tags are v0.4.0 through v0.8.0.
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/cosmos/users/yeyu/containers/nemo-rl-v0.8.0.sqsh}
STEPS=${STEPS:-3}

[ -f "$CONTAINER" ] || { echo "container missing: $CONTAINER" >&2; exit 2; }

cd "$RL_DIR"

export CONTAINER
# Per-actor venvs, set HERE rather than in the COMMAND script.
#
# docker/Dockerfile.ngc_pytorch bakes NEMO_RL_PY_EXECUTABLES_SYSTEM=1, which puts every
# Ray actor on the driver's interpreter with no venv of its own. That is only valid when
# the image already satisfies every actor -- this branch needs a newer nemo-lens than it
# ships, so the Megatron worker dies on "cannot import name 'span_attributes'".
#
# It has to be exported before sbatch: ray.sub starts the head and workers BEFORE the
# COMMAND script runs, so unsetting it inside COMMAND only affects the driver and the
# actors still come up on the container interpreter.
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
# Actor venvs are cached; the workers' own fingerprint warning recommends this when the
# code is newer than the container, which is exactly the case here.
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/.nrl-venvs}
mkdir -p "$NEMO_RL_VENV_DIR"
# ray.sub reads GPUS_PER_NODE from the environment and defaults it to 8; the sbatch
# flag alone does not set it. JHB nodes advertise gpu:4, and ray.sub refuses the
# mismatch rather than bringing up a Ray cluster that half-claims each node.
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# /lustre carries the target, the drafter and the results dir. /scratch has to be here
# too: the /lustre paths are symlinks whose real target is /scratch/fsw/..., and pyxis
# chdirs to the RESOLVED submit dir inside the container. Mounting only /lustre leaves
# that directory nonexistent in the container and the Ray head dies at startup with
# "couldn't chdir to /scratch/fsw/...", which reads like a Ray problem and is not one.
export MOUNTS="/lustre:/lustre,/scratch:/scratch"
# Repair nemo-lens on EVERY node, before Ray starts.
#
# pyproject pins nemo-lens to a rev of lens main past v0.3.0 (no release carries the
# scoped span attributes this branch uses) and the v0.8.0 image ships an older one.
# Doing this from the COMMAND script only fixed the head: /opt/nemo_rl_venv lives inside
# the container, which is per-node, so the 15 worker nodes kept their stale copy and the
# Megatron actor still died on "cannot import name 'span_attributes'".
#
# ray.sub runs SETUP_COMMAND on all nodes before starting Ray, which is exactly the hook
# for this. Idempotent: it checks for the symbol first.
export NEMO_LENS_REV=${NEMO_LENS_REV:-4d045c03186b90afdc815db46b671f6a21bc7e98}
export SETUP_COMMAND='python -c "from nemo.lens import span_attributes" 2>/dev/null \
  || uv pip install --python python \
       "nemo-lens[sdk,aiohttp,fastapi] @ git+https://github.com/NVIDIA-NeMo/Lens.git@'"${NEMO_LENS_REV}"'"'

# EXTRA_OVERRIDES lets a caller append config overrides without editing the recipe,
# which is how competing hypotheses get tested without forking it.
export COMMAND="examples/mopd_lilicorr_pipeclean.sh grpo.max_num_steps=${STEPS} logger.wandb_enabled=True ${EXTRA_OVERRIDES:-}"
# CONFIG is read by the COMMAND script; export so a caller can swap recipes.
export CONFIG=${CONFIG:-examples/configs/recipes/llm/mopd-lilicorr-super35-nvfp4-realquant.yaml}
# wandb: the run logs to it (logger.wandb_enabled=True), and without a key the driver
# dies ~3 minutes in, after Ray and the actors are already up. The key lives in
# ~/.wandb_env on this cluster; read it from there rather than committing it.
if [[ -z "${WANDB_API_KEY:-}" && -r "$HOME/.wandb_env" ]]; then
  # shellcheck disable=SC1091
  . "$HOME/.wandb_env"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[pipeclean] WANDB_API_KEY unset and ~/.wandb_env unreadable. Export it, or" \
       "pass logger.wandb_enabled=False to run without it." >&2
  exit 4
fi
export WANDB_API_KEY

export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/cosmos/users/yeyu/hf_home}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HF_HOME"

sbatch \
  --account=coreai_dlalgo_modelopt \
  --partition=batch \
  --job-name=${PIPECLEAN_JOB_NAME:-mopd_lilicorr_pipeclean} \
  --nodes=${PIPECLEAN_NODES:-16} \
  --gpus-per-node=4 \
  --time=2:00:00 \
  --output=/lustre/fsw/portfolios/cosmos/users/yeyu/logs/mopd_lilicorr_%j.log \
  ray.sub
