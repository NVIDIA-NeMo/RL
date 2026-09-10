#!/bin/bash
# Install audio/video dependencies that are NOT shipped in the NeMo-RL container.
#
# Run this script before using audio/video features or running audio/VLM tests:
#
#   bash tools/install_audio_deps.sh
#
# Safe to call multiple times.
set -euo pipefail

if ! python -c "import torchcodec" 2>/dev/null; then
    # Install system FFmpeg — torchcodec dlopens libavcodec.so.* at runtime.
    echo "[audio-deps] Installing system FFmpeg..."
    apt-get update && apt-get install -y --no-install-recommends ffmpeg

    # torchaudio 2.11+ routes torchaudio.load through torchcodec, so both are needed.
    # --no-config prevents the project's [tool.uv] overrides from interfering.
    echo "[audio-deps] Installing torchaudio==2.11.0 and torchcodec..."
    uv pip install --no-config \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        --reinstall-package torchaudio \
        "torchaudio==2.11.0" \
        "torchcodec==0.11.1"
fi

# PyAV is intentionally absent from the base image (pyproject excludes it via
# `av; sys_platform == 'never'` because it bundles CVE-carrying codec libs), so it
# must be installed after the fact into the isolated Megatron policy worker
# environment that imports it. `--no-config` bypasses that exclusion; the version
# floor is therefore restated here to keep pyproject's CVE-2026-40962 constraint.
#
# The worker venv is created lazily at worker start, so run this AFTER the
# Megatron worker has been created at least once (or point RAY_MEGATRON_PYTHON at
# an existing venv).
RAY_MEGATRON_PYTHON="${RAY_MEGATRON_PYTHON:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
if [[ ! -x "$RAY_MEGATRON_PYTHON" ]]; then
    echo "[audio-deps] ERROR: Megatron worker environment not found: $RAY_MEGATRON_PYTHON" >&2
    echo "[audio-deps] It is created on first worker start. Run this script after that," >&2
    echo "[audio-deps] or set RAY_MEGATRON_PYTHON to an existing worker interpreter." >&2
    exit 1
fi
if ! "$RAY_MEGATRON_PYTHON" -c "import av" 2>/dev/null; then
    # `uv pip install --python` targets the venv directly; these venvs are built
    # by `uv venv` without `--seed`, so they have no pip to invoke.
    echo "[audio-deps] Installing PyAV in the Megatron worker environment..."
    uv pip install --no-config --python "$RAY_MEGATRON_PYTHON" "av>=17.1.0"
fi

echo "[audio-deps] Done."
