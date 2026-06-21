#!/usr/bin/env bash
set -euo pipefail

SOURCE_VLLM_SITE="${SOURCE_VLLM_SITE:?SOURCE_VLLM_SITE is required}"
PARD2_OFFICIAL_VLLM_PATCH_DIR="${PARD2_OFFICIAL_VLLM_PATCH_DIR:?PARD2_OFFICIAL_VLLM_PATCH_DIR is required}"
PYTHON_BIN="${PARD2_VLLM_PATCH_PYTHON:-python3}"
PARD2_VLLM_SOURCE_SITE="${PARD2_VLLM_SOURCE_SITE:-}"

PATCH_FILE="${PARD2_OFFICIAL_VLLM_PATCH_DIR}/vllm_pard2_official_target_feat.patch"
ALIAS_HELPER="${PARD2_OFFICIAL_VLLM_PATCH_DIR}/apply_pard2_alias_idempotent.py"
CHECK_HELPER="${PARD2_OFFICIAL_VLLM_PATCH_DIR}/check_pard2_official_patch.py"

for required in "${PATCH_FILE}" "${ALIAS_HELPER}" "${CHECK_HELPER}"; do
  if [[ ! -s "${required}" ]]; then
    echo "ERROR: missing PARD-2 vLLM patch asset: ${required}" >&2
    exit 2
  fi
done

python_candidates=()
if [[ -n "${PARD2_VLLM_PATCH_PYTHON:-}" ]]; then
  python_candidates+=("${PARD2_VLLM_PATCH_PYTHON}")
fi
python_candidates+=("python3" "/opt/nemo_rl_venv/bin/python")

select_python_with_vllm() {
  local import_site="$1"
  local candidate
  local seen=":"
  for candidate in "${python_candidates[@]}"; do
    if [[ "${seen}" == *":${candidate}:"* ]]; then
      continue
    fi
    seen="${seen}${candidate}:"
    if [[ "${candidate}" == /* && ! -x "${candidate}" ]]; then
      continue
    fi
    if PYTHONPATH="${import_site:+${import_site}:}${PYTHONPATH:-}" "${candidate}" - <<'PY' >/dev/null 2>&1; then
import vllm
PY
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

CHECK_PYTHON="${PYTHON_BIN}"
if selected_python="$(select_python_with_vllm "${SOURCE_VLLM_SITE}")"; then
  CHECK_PYTHON="${selected_python}"
fi

if [[ -d "${SOURCE_VLLM_SITE}/vllm" ]]; then
  if PYTHONPATH="${SOURCE_VLLM_SITE}:${PYTHONPATH:-}" "${CHECK_PYTHON}" "${CHECK_HELPER}" >/tmp/pard2_vllm_check.$$ 2>&1; then
    cat /tmp/pard2_vllm_check.$$
    rm -f /tmp/pard2_vllm_check.$$
    echo "INFO: existing official PARD-2 vLLM site is valid: ${SOURCE_VLLM_SITE}"
    exit 0
  fi
  cat /tmp/pard2_vllm_check.$$ || true
  rm -f /tmp/pard2_vllm_check.$$
  echo "INFO: rebuilding invalid/incomplete official PARD-2 vLLM site: ${SOURCE_VLLM_SITE}"
fi

tmp_site="${SOURCE_VLLM_SITE}.tmp.$$"
rm -rf "${tmp_site}"
mkdir -p "${tmp_site}" "$(dirname "${SOURCE_VLLM_SITE}")"

if [[ -n "${PARD2_VLLM_SOURCE_SITE}" ]]; then
  if [[ ! -d "${PARD2_VLLM_SOURCE_SITE}/vllm" ]]; then
    echo "ERROR: PARD2_VLLM_SOURCE_SITE does not contain vllm/: ${PARD2_VLLM_SOURCE_SITE}" >&2
    exit 2
  fi
  vllm_parent="${PARD2_VLLM_SOURCE_SITE}"
  if selected_python="$(select_python_with_vllm "${PARD2_VLLM_SOURCE_SITE}")"; then
    PYTHON_BIN="${selected_python}"
  fi
else
  if ! selected_python="$(select_python_with_vllm "")"; then
    echo "ERROR: could not find a Python executable that can import vLLM." >&2
    echo "Tried: ${python_candidates[*]}" >&2
    echo "Set PARD2_VLLM_PATCH_PYTHON or PARD2_VLLM_SOURCE_SITE to a valid vLLM install." >&2
    exit 2
  fi
  PYTHON_BIN="${selected_python}"
  vllm_parent="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import vllm

print(pathlib.Path(vllm.__path__[0]).parent)
PY
)"
fi

echo "INFO: building official PARD-2 vLLM site from ${vllm_parent} using ${PYTHON_BIN}"

cp -aL "${vllm_parent}/vllm" "${tmp_site}/vllm"
(
  cd "${tmp_site}"
  patch -p1 < "${PATCH_FILE}"
  "${PYTHON_BIN}" "${ALIAS_HELPER}"
)

"${PYTHON_BIN}" -m py_compile \
  "${tmp_site}/vllm/v1/spec_decode/draft_model.py" \
  "${tmp_site}/vllm/v1/spec_decode/llm_base_proposer.py" \
  "${tmp_site}/vllm/v1/worker/gpu_model_runner.py" \
  "${tmp_site}/vllm/model_executor/models/qwen3.py" \
  "${tmp_site}/vllm/config/speculative.py"

PYTHONPATH="${tmp_site}:${PARD2_VLLM_SOURCE_SITE:+${PARD2_VLLM_SOURCE_SITE}:}${PYTHONPATH:-}" "${PYTHON_BIN}" "${CHECK_HELPER}"

rm -rf "${SOURCE_VLLM_SITE}"
mv "${tmp_site}" "${SOURCE_VLLM_SITE}"
echo "INFO: built official PARD-2 vLLM site: ${SOURCE_VLLM_SITE}"
