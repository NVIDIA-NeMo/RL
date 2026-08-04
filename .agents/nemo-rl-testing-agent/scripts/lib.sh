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

# Shared bootstrap for the LOCAL-side nemo-rl-testing-agent scripts: locates
# config.env, exports every key, and provides small helpers.
#
# The in-container scripts (prep_container.sh, run_suite_remote.sh) must NOT
# source this. They run inside the image where this repo may not be a git
# checkout, so they take everything through plain arguments and environment.

NRLTA_SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NRLTA_HOME="$(dirname "${NRLTA_SCRIPT_DIR}")"
NRLTA_CONFIG="${NRLTA_CONFIG:-${NRLTA_HOME}/config.env}"

if [[ ! -f "${NRLTA_CONFIG}" ]]; then
  echo "nemo-rl-testing-agent: config not found: ${NRLTA_CONFIG}" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "${NRLTA_CONFIG}"
set +a

nrlta_log() {
  echo "[nrlta] $*" >&2
}

nrlta_die() {
  echo "[nrlta] ERROR: $*" >&2
  exit 1
}

# Fails unless every named config key has a non-empty value.
nrlta_require() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      nrlta_die "${name} is not set in ${NRLTA_CONFIG}"
    fi
  done
}

# Loads secrets (HF_TOKEN, WANDB_API_KEY, ...) from the tokens file. Values are
# never echoed; callers expand them into remote commands themselves.
nrlta_load_tokens() {
  if [[ -f "${TOKENS_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${TOKENS_FILE}"
    set +a
  else
    nrlta_log "tokens file not found: ${TOKENS_FILE} (HF-gated tests will fail)"
  fi
}

# Per-PR ledger directory, e.g. ~/.nemo-rl-testing-agent/pr-5700.
nrlta_pr_dir() {
  local pr="${1:?nrlta_pr_dir <pr-number>}"
  echo "${STATE_DIR}/pr-${pr}"
}
