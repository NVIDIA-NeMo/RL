#!/bin/bash
# Install audio/video dependencies that are NOT shipped in the NeMo-RL container.
#
# Run this script before using audio/video features or running audio/VLM tests:
#
#   bash tools/install_audio_deps.sh
#
# Safe to call multiple times — exits immediately if already installed.
set -euo pipefail

# Fast exit: if torchcodec imports cleanly it already has FFmpeg available.
if python -c "import torchcodec" 2>/dev/null; then
    echo "[audio-deps] Already installed and functional, skipping."
    exit 0
fi

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
    "torchcodec>=0.3.0"

echo "[audio-deps] Done."
