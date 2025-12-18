#!/bin/bash
# update_container.sh - Download latest nemo-rl nightly container

# Use script's directory as container directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_DIR="${SCRIPT_DIR}"
DATE=$(date +%Y%m%d)

cd "$CONTAINER_DIR"

# 최신 nightly 다운로드
enroot import -o nemo_rl_nightly_${DATE}.sqsh docker://nvcr.io#nvidian/nemo-rl:nightly

# 날짜별 백업 + 최신 버전으로 설정
mv nvidian+nemo-rl+nightly.sqsh nemo_rl_nightly_${DATE}.sqsh
ln -sf nemo_rl_nightly_${DATE}.sqsh nemo_rl_nightly.sqsh

echo "✅ Updated to nightly ${DATE}"