#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:?set REPO_DIR to the NeMo-RL checkout}"
VENV_DIR="${VENV_DIR:?set VENV_DIR to the vLLM smoke environment}"
UV_CACHE_DIR="${UV_CACHE_DIR:?set UV_CACHE_DIR}"
HF_HOME="${HF_HOME:?set HF_HOME}"

cd "${REPO_DIR}"
export HF_HOME NRL_FORCE_REBUILD_VENVS=true UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

uv venv --allow-existing "${VENV_DIR}"
uv sync --locked --extra vllm --no-dev
"${VENV_DIR}/bin/python" scripts/vllm_024_compat_smoke.py
if [[ -n "${ENGINE_SMOKE_MODEL:-}" ]]; then
  "${VENV_DIR}/bin/python" scripts/vllm_024_engine_smoke.py
fi
if [[ -n "${PYTEST_TARGET:-}" ]]; then
  uv sync --locked --extra vllm
  "${VENV_DIR}/bin/python" -m pytest -q "${PYTEST_TARGET}" -x
fi
