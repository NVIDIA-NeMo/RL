#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Download the Qwen3.5-397B-A17B HF snapshot into an HF-hub cache layout;
# point HF_CKPT_PATH at <HF_HOME>/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/<sha>.
# Usage: HF_HOME=<hub cache root> [HF_TOKEN=...] bash download_base_ckpt.sh
set -euo pipefail

: "${HF_HOME:?HF_HOME not set}"
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-397B-A17B}

hf download "${MODEL_ID}"
echo "Snapshot downloaded under ${HF_HOME}/hub — set HF_CKPT_PATH to the snapshots/<sha> directory."
