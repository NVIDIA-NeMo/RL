#!/bin/bash
# Install audio dependencies that are NOT shipped in the NeMo-RL container.
#
# torchaudio and torchcodec bundle an FFmpeg build containing H.264/AAC decoder
# implementations that are subject to patent licensing and require org3 approval
# to distribute. They are therefore excluded from the container image.
#
# Run this script before using audio features or running audio tests:
#
#   bash scripts/install_audio_deps.sh
#
# Safe to call multiple times — exits immediately if already installed.
set -euo pipefail

# The container already has the PyPI torchaudio stub (installed via sglang/vllm),
# but it does NOT bundle FFmpeg — audio decoding requires the pytorch-cu130 build.
# Check specifically for bundled FFmpeg libs (libavcodec.so.*) in torchaudio's lib dir.
HAS_FFMPEG=$(python -c "
import torchaudio, os
lib_dir = os.path.join(os.path.dirname(torchaudio.__file__), 'lib')
has = os.path.isdir(lib_dir) and any(
    f.startswith(('libav', 'libswscale', 'libswresample'))
    for f in (os.listdir(lib_dir) if os.path.isdir(lib_dir) else [])
)
print('yes' if has else 'no')
" 2>/dev/null || echo "no")

if [[ "$HAS_FFMPEG" == "yes" ]]; then
    echo "[audio-deps] torchaudio with bundled FFmpeg already installed, skipping."
    exit 0
fi

echo "[audio-deps] Installing torchaudio==2.11.0 (pytorch-cu130, with FFmpeg) and torchcodec..."
uv pip install \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple \
    --reinstall-package torchaudio \
    "torchaudio==2.11.0" \
    "torchcodec>=0.3.0"
echo "[audio-deps] Done."
