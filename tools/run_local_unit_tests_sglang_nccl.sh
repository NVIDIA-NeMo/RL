#!/usr/bin/env bash
# Run the SGLang non-colocated NCCL bridge unit tests on the LOCAL node,
# bypassing ``uv run`` entirely.
#
# Why bypass ``uv``:
# We've repeatedly hit a chain of bring-up problems on this NFS-shared
# checkout (see [issue 2110 local-test bring-up notes](agent-transcripts):
# 8cbd9973-53cf-4f3e-af51-fe76c26abe8e):
#
#   1. ``pyproject.toml`` pins ``requires-python = ">=3.13.13"`` and
#      ``.python-version`` files mirror it; that interpreter doesn't
#      exist on PyPI/uv-managed builds.
#   2. ``Megatron-LM-workspace/setup.py``'s cached VCS deps drift from
#      the submodule's ``[tool.uv.sources]`` (``emerging_optimizers``,
#      ``nvidia-resiliency-ext``).
#   3. Stale Ray clusters + worker-venv version mismatches from past
#      Python downgrades.
#   4. ``TORCH_CUDA_ARCH_LIST`` not set on H100 nodes => failed builds.
#
# All four blow up unit-test invocations even though the unit tests
# themselves don't need GPUs, Ray, or Megatron. This helper invokes
# ``pytest`` directly via the pre-built ``.venv`` interpreter (Python
# 3.12), which already has torch/pytest/ray/aiohttp/requests installed.
# Zero env mutation, zero dep-resolve, zero rebuild.
#
# Usage:
#   bash tools/run_local_unit_tests_sglang_nccl.sh [extra pytest args]
#
# Examples:
#   bash tools/run_local_unit_tests_sglang_nccl.sh             # all bridge tests
#   bash tools/run_local_unit_tests_sglang_nccl.sh -k init     # single test
#   bash tools/run_local_unit_tests_sglang_nccl.sh -x -v       # stop on first fail
#
# If the pre-built .venv is missing or its imports are broken, the
# script falls back to printing the bring-up workarounds.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${REPO_DIR}/.venv/bin/python3"
TEST_FILE="tests/unit/models/generation/test_sglang_nccl_bridge.py"

# Extra pytest args, e.g. "-v -x" or "-k init_collective".
PYTEST_ARGS=("$@")
if [[ ${#PYTEST_ARGS[@]} -eq 0 ]]; then
  PYTEST_ARGS=("-v")
fi

echo "[run-local-unit-tests] repo=${REPO_DIR}"
echo "[run-local-unit-tests] venv python=${VENV_PY}"

# PYTHON_BIN explicitly set wins over auto-detection.
if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
  VENV_PY="${PYTHON_BIN}"
elif [[ ! -x "${VENV_PY}" ]]; then
  cat <<EOF >&2
[run-local-unit-tests] ERROR: ${VENV_PY} not found or not executable.

Override with one of:
  PYTHON_BIN=/path/to/python3 bash tools/run_local_unit_tests_sglang_nccl.sh
  uv sync --python 3.12   # then re-run

EOF
  exit 1
fi

cd "${REPO_DIR}"

# pytest may not be installed into the project .venv (it's a build/dev
# dep that uv only adds when explicitly requested). Probe for it; if
# missing, try a few sensible fallbacks before giving up.
if ! "${VENV_PY}" -c "import pytest" >/dev/null 2>&1; then
  echo "[run-local-unit-tests] pytest missing in ${VENV_PY}, looking for fallbacks..."
  candidates=()
  # 1. Conda base — most users on these nodes have pytest pre-installed
  #    in their (base) environment because they use it for general dev.
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python3")
  fi
  if [[ -n "${CONDA_PYTHON_EXE:-}" ]]; then
    candidates+=("${CONDA_PYTHON_EXE}")
  fi
  # 2. Any pytest-bearing Megatron worker venv from a previous run.
  for venv_py in "${REPO_DIR}/venvs"/*/bin/python3; do
    [[ -x "${venv_py}" ]] && candidates+=("${venv_py}")
  done
  # 3. System Python (last resort).
  candidates+=("/usr/bin/python3")

  fallback_found=""
  for cand in "${candidates[@]}"; do
    [[ -x "${cand}" ]] || continue
    if "${cand}" -c "import pytest, torch, ray, aiohttp, requests" >/dev/null 2>&1; then
      fallback_found="${cand}"
      break
    fi
  done

  if [[ -n "${fallback_found}" ]]; then
    echo "[run-local-unit-tests] using fallback python: ${fallback_found}"
    VENV_PY="${fallback_found}"
  else
    cat <<EOF >&2
[run-local-unit-tests] ERROR: pytest is not installed in ${VENV_PY} and
no fallback Python with all of (pytest, torch, ray, aiohttp, requests)
was found.

Pick the option that best matches your env:

  (a) Install pytest into the project .venv (fastest; one-shot):
        ${REPO_DIR}/.venv/bin/pip install pytest pytest-asyncio
      Then re-run this script.

  (b) Use your conda (base) env if it has the deps:
        PYTHON_BIN="\$(which python3)" bash tools/run_local_unit_tests_sglang_nccl.sh

  (c) Use any other Python 3.12+ with the deps installed:
        PYTHON_BIN=/path/to/python3 bash tools/run_local_unit_tests_sglang_nccl.sh

Tried (no luck): ${candidates[*]:-<none>}
EOF
    exit 1
  fi
fi

# Make `nemo_rl` and submodule paths importable. The .venv has
# nemo_rl installed editably (-e) so this is usually a no-op, but we
# add the repo root defensively for the case where the user uses an
# external Python via PYTHON_BIN.
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Avoid pulling Ray's dashboard / GCS during pytest collection. The
# tests don't need them and Ray sometimes spawns subprocesses on
# import.
export RAY_DEDUP_LOGS=0
export RAY_DISABLE_IMPORT_WARNING=1

# These tests are pure Python; no GPU, no NCCL, no CUDA compilation.
# Suppress the (otherwise noisy) HF Transformers TF/Flax probes.
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export USE_TF=0
export USE_FLAX=0

echo "[run-local-unit-tests] running: ${VENV_PY} -m pytest ${TEST_FILE} ${PYTEST_ARGS[*]}"
exec "${VENV_PY}" -m pytest "${TEST_FILE}" "${PYTEST_ARGS[@]}"
