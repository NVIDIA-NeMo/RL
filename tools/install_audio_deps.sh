#!/bin/bash
# Install audio/video dependencies that are NOT shipped in the NeMo-RL container.
#
# Run this script before using audio/video features or running audio/VLM tests:
#
#   bash tools/install_audio_deps.sh
#
# Safe to call multiple times — exits immediately if already installed.
set -euo pipefail

# Fast exit: both torchcodec and cv2 must be importable.
if python -c "import torchcodec; import cv2" 2>/dev/null; then
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

# opencv-python-headless is excluded from the shipped container (bundled FFmpeg codec
# libs incur royalties). vllm defaults to opencv for video IO; install >=5.0.0 which
# also patches CVE-2025-9951, CVE-2025-1594, and CVE-2024-31582.
# --no-config prevents the sys_platform=='never' override in pyproject.toml from
# suppressing this install.
echo "[audio-deps] Installing opencv-python-headless..."
uv pip install --no-config "opencv-python-headless>=5.0.0"
python -c "import cv2" || { echo "[audio-deps] ERROR: cv2 not importable after install" >&2; exit 1; }
echo "[audio-deps] Done."
