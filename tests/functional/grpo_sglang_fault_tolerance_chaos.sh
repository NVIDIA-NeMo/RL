#!/bin/bash
# Exercise real SGLang generation, offloaded liveness, and engine recovery on
# 2 GPUs. This focused harness does not launch GRPO training or transfer updated
# trainer weights; it requires survival of a batch interrupted by server SIGKILL.
# Usage: EXPECT=survival bash tests/functional/grpo_sglang_fault_tolerance_chaos.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
EXPECT=${EXPECT:-survival}
if [[ "$EXPECT" != survival ]]; then
    echo "[chaos] EXPECT must be survival; got '$EXPECT'" >&2
    exit 2
fi

ARTIFACT_DIR=${ARTIFACT_DIR:-$(mktemp -d /tmp/sglang-ft-chaos.XXXXXX)}
mkdir -p "$ARTIFACT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

echo "[chaos] Checking generation survival and recovery; artifacts: $ARTIFACT_DIR"
uv run --no-sync python -m pytest \
    tests/unit/models/generation/sglang/test_fault_tolerance_real.py \
    -m sglang --sglang-only --timeout=900 \
    --junitxml="$ARTIFACT_DIR/junit.xml" "$@" 2>&1 | tee "$ARTIFACT_DIR/run.log"
echo "[chaos] PASS: offloaded liveness, in-flight generation, and recovery survived"
