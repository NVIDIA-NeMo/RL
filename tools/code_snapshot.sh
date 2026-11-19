#!/bin/bash
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

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=${SCRIPT_DIR}/..
cd ${PROJECT_ROOT}

echo2() {
    echo "$@" >&2
}

if [[ ! -e "$PROJECT_ROOT/.git" ]]; then
  echo2 "[Error]: This script was not run from the root of NeMo RL git repo. Please clone it first."
  exit 1
elif [[ $# -lt 1 ]]; then
  echo2 "[Error]: This script requires one argument: the name of the experiment to be used as the snapshot directory name"
  echo2 "Usage: bash tools/code_snapshot.sh <experiment_name>"
  echo2 "Usage: CODE_SNAPSHOT_DIRNAME=code_snapshots_dbg bash tools/code_snapshot.sh <experiment_name>"
  exit 1
fi

EXP_NAME=$1
CODE_SNAPSHOT_DIRNAME=${CODE_SNAPSHOT_DIRNAME:-code_snapshots}

# Nested names like team/run1 are allowed, but the name must stay inside
# $CODE_SNAPSHOT_DIRNAME and be a single line, since callers `cd` into the echoed path.
is_valid_exp_name() {
  local name=$1 part
  local -a parts
  [[ -n "$name" && "$name" != /* && "$name" != *$'\n'* ]] || return 1
  IFS=/ read -ra parts <<< "$name"
  for part in "${parts[@]}"; do
    [[ "$part" != . && "$part" != .. ]] || return 1
  done
}

if ! is_valid_exp_name "$EXP_NAME"; then
  echo2 "[Error]: Invalid experiment name: '$EXP_NAME'"
  exit 1
fi

SNAPSHOT_DIR="$PROJECT_ROOT/${CODE_SNAPSHOT_DIRNAME}/${EXP_NAME}"
# Written last: an unmarked dir is an interrupted or pre-marker copy.
SNAPSHOT_MARKER="$SNAPSHOT_DIR/.snapshot_complete"

if [[ -f "$SNAPSHOT_MARKER" ]]; then
  echo2 "Using existing code snapshot in $SNAPSHOT_DIR"
  # Echo the snapshot directory so the caller can use it to `cd` into it
  echo ${SNAPSHOT_DIR}
  exit
elif [[ -d "$SNAPSHOT_DIR" ]]; then
  echo2 "Existing code snapshot in $SNAPSHOT_DIR is incomplete; filling in missing files"
else
  echo2 "Creating new code snapshot in $SNAPSHOT_DIR"
fi
mkdir -p "$SNAPSHOT_DIR"

echo2 "Copying git-tracked files and submodules..."
# Materialized so a git failure trips set -e instead of marking an empty snapshot complete.
TRACKED_FILES=$(git ls-files --recurse-submodules --cached --full-name)
# The dir may hold run outputs (logs, ckpts, continue.sh), so never delete it; --ignore-existing
# keeps already-copied code frozen, and rsync's temp-file+rename leaves no truncated files.
rsync -a --ignore-existing --files-from=<(printf '%s\n' "$TRACKED_FILES") ./ "$SNAPSHOT_DIR"/
touch "$SNAPSHOT_MARKER"

# Echo the snapshot directory so the caller can use it to `cd` into it
echo ${SNAPSHOT_DIR}
