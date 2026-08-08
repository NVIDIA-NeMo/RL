#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

AI_SEARCH_PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SEARCH_REPO_DIR="$(cd "${AI_SEARCH_PLUGIN_DIR}/../../.." && pwd)"
AI_SEARCH_SERVER_DIR="${AI_SEARCH_PLUGIN_DIR}/resources_servers/ai_search"
AI_SEARCH_RUNTIME_DIR="${AI_SEARCH_RUNTIME_DIR:-/tmp/nemo-rl-ai-search}"

if [[ -z "${UV_BIN:-}" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv was not found. Install uv or set UV_BIN=/absolute/path/to/uv." >&2
  return 1 2>/dev/null || exit 1
fi

export UV_BIN
# NeMo Gym starts child services with the command name `uv`, so make the
# directory containing an explicitly supplied UV_BIN visible to those children.
export PATH="$(dirname "${UV_BIN}"):${PATH}"
export UV_CACHE_DIR="${AI_SEARCH_UV_CACHE_DIR:-${AI_SEARCH_RUNTIME_DIR}/uv-cache}"
export HF_HOME="${AI_SEARCH_HF_HOME:-${AI_SEARCH_RUNTIME_DIR}/hf-cache}"
# Ray's Unix-domain socket path is limited to 107 bytes. Cluster-wide TMPDIR
# defaults are often much longer, so use a deliberately short example-local path.
export TMPDIR="${AI_SEARCH_TMPDIR:-${AI_SEARCH_RUNTIME_DIR}/tmp}"
export XDG_CACHE_HOME="${AI_SEARCH_XDG_CACHE_HOME:-${AI_SEARCH_RUNTIME_DIR}/cache}"
export FLASHINFER_WORKSPACE_BASE="${AI_SEARCH_FLASHINFER_HOME:-${AI_SEARCH_RUNTIME_DIR}/flashinfer}"
export TORCH_HOME="${AI_SEARCH_TORCH_HOME:-${AI_SEARCH_RUNTIME_DIR}/torch-cache}"
export CUDA_CACHE_PATH="${AI_SEARCH_CUDA_CACHE_PATH:-${AI_SEARCH_RUNTIME_DIR}/cuda-cache}"
export VLLM_CACHE_ROOT="${AI_SEARCH_VLLM_CACHE_ROOT:-${AI_SEARCH_RUNTIME_DIR}/vllm-cache}"
export TORCHINDUCTOR_CACHE_DIR="${AI_SEARCH_TORCHINDUCTOR_CACHE_DIR:-${AI_SEARCH_RUNTIME_DIR}/torchinductor-cache}"
export TRITON_CACHE_DIR="${AI_SEARCH_TRITON_CACHE_DIR:-${AI_SEARCH_RUNTIME_DIR}/triton-cache}"
export NUMBA_CACHE_DIR="${AI_SEARCH_NUMBA_CACHE_DIR:-${AI_SEARCH_RUNTIME_DIR}/numba-cache}"
export CUPY_CACHE_DIR="${AI_SEARCH_CUPY_CACHE_DIR:-${AI_SEARCH_RUNTIME_DIR}/cupy-cache}"
export UV_LINK_MODE="${AI_SEARCH_UV_LINK_MODE:-copy}"
export NEMO_GYM_VENV_DIR="${AI_SEARCH_GYM_VENV_DIR:-${AI_SEARCH_RUNTIME_DIR}/gym-venvs}"
export NEMO_RL_VENV_DIR="${AI_SEARCH_WORKER_VENV_DIR:-${AI_SEARCH_RUNTIME_DIR}/worker-venvs}"
export UV_PROJECT_ENVIRONMENT="${AI_SEARCH_TRAINING_VENV:-${AI_SEARCH_RUNTIME_DIR}/nemo-rl-venv}"

if [[ -n "${NEMO_GYM_EXTRA_ROOTS:-}" ]]; then
  export NEMO_GYM_EXTRA_ROOTS="${AI_SEARCH_PLUGIN_DIR}:${NEMO_GYM_EXTRA_ROOTS}"
else
  export NEMO_GYM_EXTRA_ROOTS="${AI_SEARCH_PLUGIN_DIR}"
fi

mkdir -p \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TORCH_HOME}" \
  "${CUDA_CACHE_PATH}" \
  "${VLLM_CACHE_ROOT}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${CUPY_CACHE_DIR}" \
  "${NEMO_GYM_VENV_DIR}/resources_servers/ai_search" \
  "${NEMO_RL_VENV_DIR}"

if [[ "${AI_SEARCH_PREPARE_TRAINING_ENV:-1}" == "1" ]]; then
  AI_SEARCH_ARCH="$(uname -m)"
  # FlashInfer does not publish this CUDA 13 JIT-cache build on the default
  # package index. Pin the official release wheel just like vLLM and FlashAttention.
  case "${AI_SEARCH_ARCH}" in
    x86_64)
      AI_SEARCH_VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl"
      AI_SEARCH_FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
      AI_SEARCH_FLASHINFER_JIT_WHEEL="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.13/flashinfer_jit_cache-0.6.13+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"
      ;;
    aarch64)
      AI_SEARCH_VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_aarch64.whl"
      AI_SEARCH_FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_aarch64.whl"
      AI_SEARCH_FLASHINFER_JIT_WHEEL="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.13/flashinfer_jit_cache-0.6.13+cu130-cp39-abi3-manylinux_2_28_aarch64.whl"
      ;;
    *)
      echo "AI-search training supports x86_64 and aarch64 CUDA hosts, not ${AI_SEARCH_ARCH}." >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac

  AI_SEARCH_TRAINING_ENV_READY=0
  if [[ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]] && \
    "${UV_PROJECT_ENVIRONMENT}/bin/python" -c \
      'import flash_attn, nemo_gym, nemo_rl, torch, vllm; assert torch.cuda.is_available()' \
      >/dev/null 2>&1; then
    AI_SEARCH_TRAINING_ENV_READY=1
  fi
  if [[ "${AI_SEARCH_FORCE_INSTALL:-0}" == "1" || "${AI_SEARCH_TRAINING_ENV_READY}" == "0" ]]; then
    "${UV_BIN}" venv --seed --allow-existing --python 3.13.14 "${UV_PROJECT_ENVIRONMENT}"
    "${UV_BIN}" pip install \
      --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
      --torch-backend cu130 \
      -e "${AI_SEARCH_REPO_DIR}" \
      -e "${AI_SEARCH_REPO_DIR}/3rdparty/Gym-workspace/Gym" \
      "${AI_SEARCH_VLLM_WHEEL}" \
      "${AI_SEARCH_FLASH_ATTN_WHEEL}" \
      "${AI_SEARCH_FLASHINFER_JIT_WHEEL}" \
      "cuda-python" \
      "flashinfer-python==0.6.13"
  fi
fi

AI_SEARCH_SERVER_VENV="${NEMO_GYM_VENV_DIR}/resources_servers/ai_search/.venv"
AI_SEARCH_SERVER_ENV_READY=0
if [[ -x "${AI_SEARCH_SERVER_VENV}/bin/python" ]] && \
  PYTHONPATH="${AI_SEARCH_PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${AI_SEARCH_SERVER_VENV}/bin/python" -c \
    'import cupy, cuvs, nemo_gym, retrieval, torch, transformers' \
    >/dev/null 2>&1; then
  AI_SEARCH_SERVER_ENV_READY=1
fi
if [[ "${AI_SEARCH_FORCE_INSTALL:-0}" == "1" || "${AI_SEARCH_SERVER_ENV_READY}" == "0" ]]; then
  "${UV_BIN}" venv --seed --allow-existing --python 3.13.14 "${AI_SEARCH_SERVER_VENV}"
  "${UV_BIN}" pip install \
    --python "${AI_SEARCH_SERVER_VENV}/bin/python" \
    -e "${AI_SEARCH_SERVER_DIR}"
fi

AI_SEARCH_EMBEDDINGS="${AI_SEARCH_SERVER_DIR}/data/index/e5-small-v2.embeddings.npy"
AI_SEARCH_MANIFEST="${AI_SEARCH_SERVER_DIR}/data/index/e5-small-v2.embeddings.manifest.json"
if [[ ! -f "${AI_SEARCH_EMBEDDINGS}" || ! -f "${AI_SEARCH_MANIFEST}" ]]; then
  PYTHONPATH="${AI_SEARCH_PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${UV_BIN}" run --no-project \
    --python "${AI_SEARCH_SERVER_VENV}/bin/python" \
    python -m resources_servers.ai_search.prepare_index \
    --config "${AI_SEARCH_SERVER_DIR}/configs/ai_search.yaml" \
    --force
fi

echo "AI-search environment is ready."
echo "  plugin: ${AI_SEARCH_PLUGIN_DIR}"
if [[ "${AI_SEARCH_PREPARE_TRAINING_ENV:-1}" == "1" ]]; then
  echo "  training venv: ${UV_PROJECT_ENVIRONMENT}"
fi
echo "  server venv: ${AI_SEARCH_SERVER_VENV}"
echo "  embeddings: ${AI_SEARCH_EMBEDDINGS}"
