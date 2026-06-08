#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Single entrypoint that chains the four projection-prep CLIs under
# tools/x_token/ to produce a runtime projection matrix from a
# (student, teacher) tokenizer pair. See docs/guides/xtoken-off-policy-distillation.md
# for the underlying steps.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Move to repo root so `uv run python -m tools.x_token.*` resolves.
cd "${SCRIPT_DIR}/../.."

STUDENT=""
TEACHER=""
DATA_DIR="cross_tokenizer_data"
PREP_TOP_K=32
RUNTIME_TOP_K=4
FINAL_OUTPUT=""
USE_CANONICALIZATION=false
ENABLE_SCALE_TRICK=true
ENABLE_REVERSE_PASS=true
ENABLE_SPECIAL_TOKEN_MAPPING=true
SKIP_EXACT_MAP=false

usage() {
  cat <<EOF
Usage: $(basename "$0") --student-model <id> --teacher-model <id> [options]

Chains the projection-prep steps (seed pass + Steps 1-3) into a
single runtime projection matrix.

Required:
  --student-model <id>          HuggingFace student model id
  --teacher-model <id>          HuggingFace teacher model id

Common:
  --data-dir <dir>              Staging dir for intermediate artifacts
                                (default: ${DATA_DIR})
  --prep-top-k <N>              top_k used during prep (seed + Step 1)
                                (default: ${PREP_TOP_K})
  --runtime-top-k <N>           Final runtime top_k, Step 3
                                (default: ${RUNTIME_TOP_K})
  --final-output <path>         Final .pt path
                                (default: <data-dir>/projection_matrix_<S>_<T>_top<N>.pt)
  --skip-exact-map              Skip Step 2 (reapply_exact_map.py)
  --use-canonicalization        Forward to seed + Step 1
  --no-scale-trick              Disable scale trick (Step 1)
  --no-reverse-pass             Disable reverse pass (Step 1)
  --no-special-token-mapping    Disable special-token mapping (Step 1)
  -h, --help                    Show this help and exit

Example:
  $(basename "$0") \\
      --student-model meta-llama/Llama-3.2-1B \\
      --teacher-model Qwen/Qwen3-4B \\
      --runtime-top-k 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model)             STUDENT="$2"; shift 2 ;;
    --teacher-model)             TEACHER="$2"; shift 2 ;;
    --data-dir)                  DATA_DIR="$2"; shift 2 ;;
    --prep-top-k)                PREP_TOP_K="$2"; shift 2 ;;
    --runtime-top-k)             RUNTIME_TOP_K="$2"; shift 2 ;;
    --final-output)              FINAL_OUTPUT="$2"; shift 2 ;;
    --skip-exact-map)            SKIP_EXACT_MAP=true; shift ;;
    --use-canonicalization)      USE_CANONICALIZATION=true; shift ;;
    --no-scale-trick)            ENABLE_SCALE_TRICK=false; shift ;;
    --no-reverse-pass)           ENABLE_REVERSE_PASS=false; shift ;;
    --no-special-token-mapping)  ENABLE_SPECIAL_TOKEN_MAPPING=false; shift ;;
    -h|--help)                   usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$STUDENT" || -z "$TEACHER" ]]; then
  echo "--student-model and --teacher-model are required." >&2
  usage >&2
  exit 2
fi

# Mirror tools/x_token/utils.py::clean_model_name_for_filename so the
# wrapper can predict intermediate filenames without spawning Python.
clean_name() {
  local name="$1" cleaned
  cleaned="$(echo "$name" | sed -E 's/-?[0-9.]+[bBmM]//g; s/-(Base|it|Instruct)//g; s/^[-_]+|[-_]+$//g')"
  if [[ "$name" == *mini* ]]; then
    cleaned="${cleaned}_mini"
  fi
  echo "$cleaned"
}

S_LAST="${STUDENT##*/}"
T_LAST="${TEACHER##*/}"
S_CLEAN="$(clean_name "$S_LAST")"
T_CLEAN="$(clean_name "$T_LAST")"

STEP1_OUT="${DATA_DIR}/temp_projection_map_${S_CLEAN}_to_${T_CLEAN}_top_${PREP_TOP_K}.pt"
STEP2_FILENAME="projection_map_${S_CLEAN}_to_${T_CLEAN}_multitoken_top_${PREP_TOP_K}_double"
if $ENABLE_SPECIAL_TOKEN_MAPPING; then
  STEP2_FILENAME="${STEP2_FILENAME}_special"
fi
STEP2_OUT="${DATA_DIR}/${STEP2_FILENAME}.pt"
STEP3_OUT="${STEP2_OUT%.pt}_exact_map_remapped.pt"
if $SKIP_EXACT_MAP; then
  STEP4_IN="$STEP2_OUT"
else
  STEP4_IN="$STEP3_OUT"
fi
if [[ -z "$FINAL_OUTPUT" ]]; then
  FINAL_OUTPUT="${DATA_DIR}/projection_matrix_${S_CLEAN}_${T_CLEAN}_top${RUNTIME_TOP_K}.pt"
fi

mkdir -p "$DATA_DIR"

echo "[seed] minimal_projection_generator (-> ${STEP1_OUT}) ..."
step1_args=(
  --student-model "$STUDENT"
  --teacher-model "$TEACHER"
  --top_k "$PREP_TOP_K"
  --data_dir "$DATA_DIR"
)
if $USE_CANONICALIZATION; then
  step1_args+=(--use_canonicalization)
fi
uv run python -m tools.x_token.minimal_projection_generator "${step1_args[@]}"

echo "[1/3] minimal_projection_via_multitoken (-> ${STEP2_OUT}) ..."
step2_args=(
  --student-model "$STUDENT"
  --teacher-model "$TEACHER"
  --initial-projection-path "$STEP1_OUT"
  --top-k "$PREP_TOP_K"
  --output-dir "$DATA_DIR"
)
if $ENABLE_SCALE_TRICK; then
  step2_args+=(--enable-scale-trick)
else
  step2_args+=(--disable-scale-trick)
fi
if $ENABLE_REVERSE_PASS; then
  step2_args+=(--enable-reverse-pass)
else
  step2_args+=(--disable-reverse-pass)
fi
if $ENABLE_SPECIAL_TOKEN_MAPPING; then
  step2_args+=(--enable-special-token-mapping)
else
  step2_args+=(--disable-special-token-mapping)
fi
if $USE_CANONICALIZATION; then
  step2_args+=(--use-canonicalization)
fi
uv run python -m tools.x_token.minimal_projection_via_multitoken "${step2_args[@]}"

if ! $SKIP_EXACT_MAP; then
  echo "[2/3] reapply_exact_map (-> ${STEP3_OUT}) ..."
  uv run python -m tools.x_token.reapply_exact_map \
    --student-model "$STUDENT" \
    --teacher-model "$TEACHER" \
    --initial-projection-path "$STEP2_OUT"
else
  echo "[2/3] reapply_exact_map skipped (--skip-exact-map)."
fi

echo "[3/3] sort_and_cut_projection_matrix (-> ${FINAL_OUTPUT}) ..."
uv run python -m tools.x_token.sort_and_cut_projection_matrix \
  --initial-projection-path "$STEP4_IN" \
  --top_k "$RUNTIME_TOP_K" \
  --output_path "$FINAL_OUTPUT"

echo "Done. Final projection matrix: ${FINAL_OUTPUT}"
