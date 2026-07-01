#!/bin/bash
# Manually force a venv's CuTe-DSL (nvidia-cutlass-dsl) install to single-provenance
# -libs-cu13. Workaround for the packaging bug where -libs-base and -libs-cu13 write
# divergent content to the same paths under nvidia_cutlass_dsl/ (racy under uv), which
# breaks cuDNN-frontend 1.25.0's cutedsl DSAttention path when -libs-base files win.
# The automated path is the hook in nemo_rl/utils/venvs.py (opt-in NRL_FORCE_CUTLASS_CU13);
# this script is for manual repair / non-NeMo-RL launchers.
#
# Upstream (still OPEN):
#   https://github.com/NVIDIA/cutlass/issues/3170
#   https://github.com/NVIDIA/cutlass/issues/3259
#
# Usage: reconcile_cutlass_cu13.sh <mcore-venv-dir> [uv-cache-dir]
set -euo pipefail
VENV="${1:?usage: reconcile_cutlass_cu13.sh <mcore-venv-dir> [uv-cache-dir]}"
UV_CACHE="${2:-${UV_CACHE_DIR:-$HOME/.cache/uv}}"

DST=$(ls -d "$VENV"/lib/python*/site-packages/nvidia_cutlass_dsl/python_packages/cutlass 2>/dev/null | head -1)
[ -n "${DST:-}" ] && [ -d "$DST" ] || { echo "no CuTe-DSL tree in $VENV"; exit 1; }

is_cu13() {  # $1 = cutlass dir
  grep -q "def normalize_field_to_ir_name" "$1/cute/nvgpu/common.py" 2>/dev/null \
    && grep -q "def atomicrmw(op, ptr, a" "$1/_mlir/dialects/_nvvm_ops_gen.py" 2>/dev/null
}

if is_cu13 "$DST"; then echo "already single-provenance cu13; nothing to do"; exit 0; fi

SRC=""
for di in "$UV_CACHE"/archive-v0/*/nvidia_cutlass_dsl_libs_cu13*.dist-info; do
  [ -e "$di" ] || continue
  cand="$(dirname "$di")/nvidia_cutlass_dsl/python_packages/cutlass"
  if [ -d "$cand" ] && is_cu13 "$cand"; then SRC="$cand"; break; fi
done
[ -n "$SRC" ] || { echo "ERROR: no pristine -libs-cu13 tree under $UV_CACHE (see cutlass#3170)"; exit 1; }

rsync -a "$SRC/" "$DST/"
echo "reconciled $DST -> -libs-cu13 (from $SRC)"
echo "  atomicrmw: $(grep -oE 'def atomicrmw\([a-z, ]+' "$DST/_mlir/dialects/_nvvm_ops_gen.py" | head -1)"
echo "  normalize_field_to_ir_name def: $(grep -c 'def normalize_field_to_ir_name' "$DST/cute/nvgpu/common.py")"
