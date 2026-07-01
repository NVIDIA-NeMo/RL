#!/bin/bash
# ============================================================================
# Combined bake for the SWE-bench training image, in ONE rootfs / ONE export:
#   (a) apptainer/singularity  — the SWE sandbox (harbor_agent,
#       environment_type="singularity") needs the runtime; the upstream
#       6de99f772 image does NOT ship it (only a stray broken /usr/local/lib
#       /singularity symlink).
#   (b) driver mcore env       — /opt/nemo_rl_venv + `--extra mcore`
#       (transformer-engine + megatron + mamba-ssm + ...). The image bakes the
#       per-worker venvs but NOT the driver's, so `uv run --frozen --extra
#       mcore` recompiles TE (~hr) every launch. Baking it makes that a no-op.
#
# Merges build_apptainer_image.sh + build_mcore_env_image.sh so the 29GB
# unpack/export only happens once (instead of twice, sequentially).
#
# REUSE_ROOTFS=1 reuses an already-unpacked enroot rootfs (CONTAINER_NAME) to
# skip the ~20-min re-unpack. Locks/partial-build dirs must already be cleaned.
#
# Run on a node WITH internet (PPA + git deps) + CUDA toolkit in the image
# (TE cross-compiles via nvcc -> NO GPU needed). Heavy: ~1-2h on few cores.
#
# Target arch: GB200 / Blackwell only (sm_100). See TORCH_CUDA_ARCH_LIST below.
#
# Usage:
#   REUSE_ROOTFS=1 CONTAINER_NAME=nrl_swebench_mcore_build \
#   BASE_IMAGE=... OUT_IMAGE=... REPO=... \
#   bash test_assets/SWE/build_swe_bench_combined.sh
# ============================================================================
set -euo pipefail

REPO="${REPO:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/ruit/evolution_rl}"
# GB200 (aarch64) base image, prebuilt by the user.
BASE_IMAGE="${BASE_IMAGE:-/lustre/fsw/portfolios/nemotron/users/ruit/enroot-images/ruit-swe_bench-6dc8fabea-aarch64-060426.squashfs}"
OUT_IMAGE="${OUT_IMAGE:-/lustre/fsw/portfolios/nemotron/users/ruit/enroot-images/ruit-swe_bench-6dc8fabea-aarch64-060426-mcore-apptainer.squashfs}"
CONTAINER_NAME="${CONTAINER_NAME:-nrl_swebench_combined_build}"
REUSE_ROOTFS="${REUSE_ROOTFS:-0}"
APPTAINER_VERSION="${APPTAINER_VERSION:-1.5.0-2-1}"

export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/tmp/enroot-data-${USER}}"
mkdir -p "${ENROOT_DATA_PATH}"
# Public deps, but pass a token through if present to dodge GitHub rate limits.
source "${HOME}/script/export_env_vars.sh" 2>/dev/null || source "${REPO}/../script/export_env_vars.sh" 2>/dev/null || true
# Cross-compile TE via nvcc -> no GPU; disable enroot's NVIDIA hook so it does
# not require nvidia-container-cli (absent on a login node).
export NVIDIA_VISIBLE_DEVICES=void
# Do NOT bind-mount host home over the container (would shadow /root and hide
# /root/.local/bin/uv) — enroot equivalent of pyxis --no-container-mount-home.
export ENROOT_MOUNT_HOME=n

echo "=========================================="
echo "Base   : ${BASE_IMAGE}"
echo "Out    : ${OUT_IMAGE}"
echo "Repo   : ${REPO} (mounted; editable mcore installs)"
echo "Cont   : ${CONTAINER_NAME}  (reuse=${REUSE_ROOTFS}, data=${ENROOT_DATA_PATH})"
echo "=========================================="

[ -f "${BASE_IMAGE}" ] || { echo "ERROR: base image not found: ${BASE_IMAGE}" >&2; exit 1; }

# ---- [1/5] obtain a writable rootfs ----------------------------------------
if [ "${REUSE_ROOTFS}" = "1" ] && enroot list 2>/dev/null | grep -qx "${CONTAINER_NAME}"; then
  echo "[1/5] Reusing already-unpacked rootfs: ${CONTAINER_NAME}"
else
  echo "[1/5] Unpacking base image into a fresh rootfs..."
  enroot remove -f "${CONTAINER_NAME}" 2>/dev/null || true
  enroot create --name "${CONTAINER_NAME}" "${BASE_IMAGE}"
fi

# ---- [2/5] install apptainer/singularity -----------------------------------
if [ "${SKIP_APPTAINER:-0}" = "1" ] || [ -x "${ENROOT_DATA_PATH}/${CONTAINER_NAME}/usr/bin/apptainer" ]; then
  echo "[2/5] apptainer already present in rootfs — skipping install"
else
echo "[2/5] Installing apptainer (PPA) + singularity symlink..."
enroot start --root --rw "${CONTAINER_NAME}" bash -euxo pipefail -c '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends software-properties-common ca-certificates
  add-apt-repository -y ppa:apptainer/ppa
  apt-get update
  CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
  apt-get install -y --no-install-recommends "apptainer='"${APPTAINER_VERSION}"'~${CODENAME}" \
    || { echo "pinned apptainer '"${APPTAINER_VERSION}"'~${CODENAME} unavailable; installing latest from PPA"; \
         apt-get install -y --no-install-recommends apptainer; }
  ln -sf /usr/bin/apptainer /usr/bin/singularity
  apt-get clean && rm -rf /var/lib/apt/lists/*
  apptainer --version
  singularity --version
'
fi

# ---- [3/5] bake driver mcore env (compiles TE/megatron/mamba into venv) -----
echo "[3/5] Syncing /opt/nemo_rl_venv WITH --extra mcore (builds TE etc.)..."
enroot start --root --rw \
  --mount "${REPO}:${REPO}" \
  --env GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  "${CONTAINER_NAME}" bash -exo pipefail -c "
    export PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:\$PATH
    cd '${REPO}'
    export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    export UV_CACHE_DIR=/tmp/uv_cache
    export UV_HTTP_TIMEOUT=3600
    export UV_LOCK_TIMEOUT=900
    # GB200-only build (Blackwell sm_100). TE v2.14.1 derives its CUDA archs from
    # TORCH_CUDA_ARCH_LIST when NVTE_CUDA_ARCHS is unset (matches docker/Dockerfile,
    # which builds TE for GB200 with TORCH_CUDA_ARCH_LIST of 9.0 10.0). Do NOT set
    # NVTE_CUDA_ARCHS=100 -- TE CMake rejects the raw 100 token (CUDA_ARCHITECTURES
    # empty); the TORCH_CUDA_ARCH_LIST=10.0 derivation emits the correct sm_100.
    # NOTE: keep this block free of double-quote chars -- it lives inside bash -c (double-quoted).
    export TORCH_CUDA_ARCH_LIST='10.0'
    uv run --frozen --extra mcore python -c 'import nemo_rl.algorithms.grpo; print(\"mcore driver env OK\")'
  "

# ---- [4/5] verify BOTH baked artifacts BEFORE exporting --------------------
echo "[4/5] Verifying apptainer + mcore inside the image (abort export if bad)..."
# Verify via 'uv run --frozen --extra mcore' (exactly how the runtime invokes
# training). NOTE: import nemo_rl FIRST -- nemo_rl/__init__.py appends
# 3rdparty/Megatron-LM-workspace/Megatron-LM to sys.path, which is what makes
# 'import megatron.core' resolve (bare import megatron.core fails otherwise).
# Mount the repo so the editable source is present (run-time mounts /lustre likewise).
enroot start --root --rw --mount "${REPO}:${REPO}" "${CONTAINER_NAME}" bash -exo pipefail -c "
  /usr/bin/apptainer --version
  /usr/bin/singularity --version
  export PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:\$PATH
  cd '${REPO}'
  export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
  export UV_CACHE_DIR=/tmp/uv_cache
  uv run --frozen --extra mcore python -c 'import nemo_rl, megatron.core, transformer_engine; print(\"apptainer + megatron + TE OK via uv run --extra mcore\")'
"

# ---- [5/5] export + cleanup -------------------------------------------------
echo "[5/5] Exporting combined squashfs -> ${OUT_IMAGE} ..."
rm -f "${OUT_IMAGE}"
enroot export --output "${OUT_IMAGE}" "${CONTAINER_NAME}"
echo "Cleaning up unpacked rootfs..."
enroot remove -f "${CONTAINER_NAME}"

echo "=========================================="
echo "Done. Combined image (apptainer + mcore):"
ls -lh "${OUT_IMAGE}"
echo
echo "Point the training script at it:"
echo "  CONTAINER=${OUT_IMAGE} bash test_assets/SWE/run_grpo_qwen3_30b_thinking_swe2.sh"
echo "=========================================="