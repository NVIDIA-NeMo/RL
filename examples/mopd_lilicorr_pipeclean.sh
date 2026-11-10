#!/bin/bash
# COMMAND script for ray.sub: QA-MOPD with a co-trained LiLiCorr drafter.
#
# ray.sub runs this inside the container with the Ray cluster already up, from
# SLURM_SUBMIT_DIR. Extra args are forwarded as config overrides, which is how the
# pipeclean shortens the run without editing the recipe.
set -eux

CONFIG=${CONFIG:-examples/configs/recipes/llm/mopd-lilicorr-super35-nvfp4-realquant.yaml}

# Resolve the full branch environment with `uv run`, using a container whose Ray
# already matches this branch's lock (v0.8.0 ships Ray 2.58.0).
#
# History, because both directions are wrong with the wrong image:
#   - On v0.7.0 (Ray 2.55.1) `uv run` rebuilt the venv to Ray 2.58.0 and the driver was
#     refused by the cluster ray.sub had started from the container venv.
#   - Falling back to the container's interpreter then failed differently: the merged
#     branch needs deps v0.7.0 predates (nemo.lens), and on v0.8.0 it needs a NEWER
#     nemo.lens than the image ships -- `span_attributes` is missing.
# Patching single gaps one at a time does not converge; the branch's lock is the only
# consistent description of what it needs. With v0.8.0 the cluster's Ray and the lock's
# Ray agree, so rebuilding the venv is safe and resolves every gap at once.
#
# Set PIPECLEAN_CONTAINER_PYTHON=1 to use the image's interpreter instead, which only
# works if the image already satisfies every dependency the branch imports.
PYTHON_BIN=${PYTHON_BIN:-python}

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
# Let each Ray actor build its own venv.
#
# docker/Dockerfile.ngc_pytorch sets NEMO_RL_PY_EXECUTABLES_SYSTEM=1, which rewrites
# every uv command in PY_EXECUTABLES to SYSTEM: all actors then launch on the DRIVER's
# interpreter and no per-actor venv is created. The design doc is explicit that this is
# only valid "in an environment where every actor's dependencies are already installed,
# such as a single-environment container image".
#
# That is exactly what this branch is not. It needs a newer nemo-lens than the image
# ships, so the Megatron worker inherited the driver's env and died with
# "cannot import name 'span_attributes' from 'nemo.lens'" -- after Ray, the actors and
# the policy had all started. It also cannot be fixed by giving the driver more extras,
# because mcore and vllm are mutually exclusive and the actors need different ones.
#
# Unsetting it restores the per-actor model: MCORE actors get --extra mcore, VLLM actors
# get --extra vllm, each in its own venv.
# Set the venv switches HERE as well as in the submit script.
#
# The image bakes NEMO_RL_PY_EXECUTABLES_SYSTEM=1 as a Dockerfile ENV, and ray.sub adds
# no --export, so the baked value wins inside the container and the submit-side export
# does not reach it. But the DRIVER is what resolves each actor's py_executable, so
# setting it in this process is what actually decides whether actors get their own venvs.
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/cosmos/users/yeyu/.nrl-venvs}
mkdir -p "$NEMO_RL_VENV_DIR"
echo "[pipeclean] NEMO_RL_PY_EXECUTABLES_SYSTEM=$NEMO_RL_PY_EXECUTABLES_SYSTEM" \
     "NEMO_RL_VENV_DIR=$NEMO_RL_VENV_DIR" >&2

# Keep those venvs on shared storage so they are built once rather than per job.
UV_BIN=${UV_BIN:-uv}
PINNED_UV_VERSION=${PINNED_UV_VERSION:-0.11.28}   # docker/Dockerfile's pin

if [[ "${PIPECLEAN_CONTAINER_PYTHON:-0}" != "1" ]]; then
  # An image older than the branch cannot parse `override-dependencies`' map form and
  # dies on a TOML line number rather than the real problem -- after Ray is already up
  # on every node. Repair it here instead of burning the allocation.
  if ! "$UV_BIN" lock --check >/dev/null 2>&1 && ! "$UV_BIN" tree --quiet >/dev/null 2>&1; then
    echo "[pipeclean] container uv ($("$UV_BIN" --version 2>&1|head -1)) cannot parse this" \
         "branch's pyproject.toml; installing uv ${PINNED_UV_VERSION}" >&2
    UV_HOME=${UV_HOME:-${SLURM_SUBMIT_DIR:-$PWD}/.pipeclean-uv}
    mkdir -p "$UV_HOME"
    curl -LsSf "https://astral.sh/uv/${PINNED_UV_VERSION}/install.sh" \
      | env UV_UNMANAGED_INSTALL="$UV_HOME" sh >&2
    UV_BIN="$UV_HOME/uv"
  fi
  "$UV_BIN" --version >&2
fi

# NO extras on the driver, and --locked.
#
# nemo_rl launches each Ray actor under its OWN interpreter: virtual_cluster.py defines
# MCORE as `uv run --locked --extra mcore` and VLLM as `uv run --locked --extra vllm`.
# The driver only has to import nemo_rl and start Ray; megatron and vllm are imported
# inside those actors.
#
# Passing extras here is not merely unnecessary, it is rejected: pyproject declares
# mcore, vllm, automodel, sglang and trtllm as mutually CONFLICTING, so
# `--extra mcore --extra vllm` fails with "Extras ... are incompatible with the declared
# conflicts". They cannot share one environment by design.
#
# --locked makes the driver use the lockfile exactly, which is also what the actors use,
# so driver and actors agree on the shared base packages.
UV_EXTRAS=${UV_EXTRAS:-}
read -r -a _uv_extras <<< "$UV_EXTRAS"
RUNNER=("$UV_BIN" run --locked "${_uv_extras[@]}" python)
if [[ "${PIPECLEAN_CONTAINER_PYTHON:-0}" == "1" ]]; then
  RUNNER=("$PYTHON_BIN")
fi

# The image ships its own nemo_rl at /opt/nemo-rl, which shadows this checkout. Running
# the branch's examples/run_grpo.py against the image's older package fails with
# ImportError on names the branch added (make_policy_factory). Put the checkout first.
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}${PYTHONPATH:+:$PYTHONPATH}"

# Repair the one package the image is too old for.
#
# pyproject pins nemo-lens to a rev of lens main past v0.3.0 -- explicitly, because no
# release carries the scoped span attributes this branch uses. The v0.8.0 image ships an
# older lens, so both the driver and every Ray actor (which share /opt/nemo_rl_venv) die
# with "cannot import name 'span_attributes' from 'nemo.lens'".
#
# Patching the shared venv is what actually works here. Per-actor venvs are the designed
# answer, but the image bakes NEMO_RL_PY_EXECUTABLES_SYSTEM=1 and uv's project
# environment points back at /opt/nemo_rl_venv, so the actors keep landing in the same
# place; installing the pinned rev there fixes driver and actors together.
#
# Idempotent: skipped entirely when the running lens already has the symbol.
NEMO_LENS_REV=${NEMO_LENS_REV:-4d045c03186b90afdc815db46b671f6a21bc7e98}
if ! "$PYTHON_BIN" -c "from nemo.lens import span_attributes" >/dev/null 2>&1; then
  echo "[pipeclean] image nemo-lens lacks span_attributes; installing pinned rev" >&2
  "$UV_BIN" pip install --python "$PYTHON_BIN" \
    "nemo-lens[sdk,aiohttp,fastapi] @ git+https://github.com/NVIDIA-NeMo/Lens.git@${NEMO_LENS_REV}" >&2
  "$PYTHON_BIN" -c "from nemo.lens import span_attributes; print('[pipeclean] nemo-lens repaired')" >&2
else
  echo "[pipeclean] nemo-lens already provides span_attributes" >&2
fi

# Verify the BRANCH's nemo_rl is what imports -- not merely that some nemo_rl does.
# An earlier version of this check only did `import nemo_rl`, which the image satisfies
# from /opt/nemo-rl, so it passed and the run still died on a missing symbol.
"${RUNNER[@]}" - <<'PREFLIGHT' >&2
import os, sys
import ray, nemo_rl
repo = os.path.realpath(os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd())
pkg = os.path.realpath(os.path.dirname(nemo_rl.__file__))
print(f"[pipeclean] ray {ray.__version__}")
print(f"[pipeclean] nemo_rl {pkg}")
if not pkg.startswith(repo):
    sys.exit(
        f"[pipeclean] nemo_rl resolves to {pkg}, not the checkout at {repo}. "
        "The image's copy would shadow this branch and fail on symbols it adds."
    )
from nemo_rl.data_plane.factory import make_policy_factory  # noqa: F401
from nemo_rl.models.megatron.draft.training import LiLiCorrSpeculator  # noqa: F401
print("[pipeclean] branch nemo_rl imports, lilicorr speculator present")
PREFLIGHT

exec "${RUNNER[@]}" examples/run_grpo.py --config "$CONFIG" "$@"
