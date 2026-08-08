#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

AI_SEARCH_PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AI_SEARCH_PREPARE_TRAINING_ENV=0
source "${AI_SEARCH_PLUGIN_DIR}/prepare_ai_search.sh"

AI_SEARCH_SERVER_DIR="${AI_SEARCH_PLUGIN_DIR}/resources_servers/ai_search"
AI_SEARCH_SERVER_VENV="${NEMO_GYM_VENV_DIR}/resources_servers/ai_search/.venv"
AI_SEARCH_PROFILE_DIR="${AI_SEARCH_PROFILE_DIR:-/tmp/nemo-rl-ai-search/profiles}"
AI_SEARCH_PROFILE_DOCUMENTS="${AI_SEARCH_PROFILE_DOCUMENTS:-100000}"
AI_SEARCH_PROFILE_REPEATS="${AI_SEARCH_PROFILE_REPEATS:-20}"

"${UV_BIN}" pip install \
  --python "${AI_SEARCH_SERVER_VENV}/bin/python" \
  -e "${AI_SEARCH_SERVER_DIR}[profile]"

mkdir -p "${AI_SEARCH_PROFILE_DIR}"
cd "${AI_SEARCH_PLUGIN_DIR}"
PYTHONPATH="${AI_SEARCH_PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${AI_SEARCH_SERVER_VENV}/bin/python" -m profiling.profile_retrieval \
  --documents "${AI_SEARCH_PROFILE_DOCUMENTS}" \
  --repeats "${AI_SEARCH_PROFILE_REPEATS}" \
  --json-out "${AI_SEARCH_PROFILE_DIR}/retrieval.json" \
  --markdown-out "${AI_SEARCH_PROFILE_DIR}/retrieval.md"

PYTHONPATH="${AI_SEARCH_PLUGIN_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${AI_SEARCH_SERVER_VENV}/bin/python" -m profiling.profile_pipeline \
  --repeats "${AI_SEARCH_PROFILE_REPEATS}" \
  --json-out "${AI_SEARCH_PROFILE_DIR}/pipeline.json"

echo "Profiles written to ${AI_SEARCH_PROFILE_DIR}"
