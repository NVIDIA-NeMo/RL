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

# PyAV is intentionally absent from the base image and must be installed into
# the isolated Megatron policy worker environment that imports it.
RAY_MEGATRON_PYTHON="${RAY_MEGATRON_PYTHON:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
if [[ -x "$RAY_MEGATRON_PYTHON" ]]; then
    echo "[audio-deps] Force-reinstalling PyAV in the Megatron worker environment..."
    "$RAY_MEGATRON_PYTHON" -m pip install --no-cache-dir --force-reinstall av
else
    echo "[audio-deps] Megatron worker environment not found; skipping PyAV: $RAY_MEGATRON_PYTHON"
fi

echo "[audio-deps] Done."
