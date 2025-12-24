#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
PROJECT_ROOT=$(realpath $SCRIPT_DIR/..)

# Accept exactly one argument: TEST_CASE path
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <TEST_CASE>" >&2
  exit 2
fi
TEST_CASE="$1"
if [[ ! -f "$TEST_CASE" ]]; then
  echo "[ERROR]: TEST_CASE not found: $TEST_CASE" >&2
  exit 2
fi

# If any of these env vars exit 128 (right outside valid git bisect exit code to abort the bisect process)
for i in HF_HOME HF_DATASETS_CACHE CONTAINER MOUNTS ACCOUNT PARTITION; do
    if [[ -z "${!i:-}" ]]; then
        echo "[ERROR]: $i environment variable is not set."
        exit 128
    fi
done

# If SED_CLAUSES is provided, apply them to the TEST_CASE before launching
if [[ -n "${SED_CLAUSES:-}" ]]; then
    # If the test is tracked in git, set up a restore trap to revert our edits on exit
    if git ls-files --error-unmatch "$TEST_CASE" >/dev/null 2>&1; then
        trap 'git restore --source=HEAD -- "$TEST_CASE" >/dev/null 2>&1 || true' EXIT
    else
        echo "[WARN]: $TEST_CASE is not tracked by git; modifications will not be auto-restored." >&2
    fi

    echo "[bisect-helper] Applying SED_CLAUSES to $TEST_CASE..." >&2
    TMP_SED_SCRIPT=$(mktemp)
    printf "%s\n" "$SED_CLAUSES" > "$TMP_SED_SCRIPT"
    # Use sed -i with a script file; do not force -E to allow generic sed syntax
    sed -i -f "$TMP_SED_SCRIPT" "$TEST_CASE"
    rm -f "$TMP_SED_SCRIPT"
fi

# This makes it so ./launch will block until the test is done and check if the metrics pass or fail and exit appropriately
export WATCH=1
# Always rebuild venvs because we are using a static container, but we need the environment to match the commit
# Use a different megatron checkpoint directory since there may be failures related to the checkpoint conversion itself
export EXTRA_ENV="${EXTRA_ENV:-} NRL_FORCE_REBUILD_VENVS=true NRL_MEGATRON_CHECKPOINT_DIR=$PROJECT_ROOT/code_snapshots_bisect/$(basename $TEST_CASE .sh)/mcore_ckpt_dir_$(git log -1 --format='%h-%f' HEAD)"
# Use a different code snapshot directory name for each commit otherwise the same named test will run
export CODE_SNAPSHOT_DIRNAME=code_snapshots_bisect/$(git log -1 --format='%h-%f' HEAD)

set +e
bash $SCRIPT_DIR/launch "$TEST_CASE"
ret_code=$?
set -e

# We clean out each submodule since if you do not clean it out you can sometimes get this error:
# error: Entry '3rdparty/Automodel-workspace/Automodel/.github/CODEOWNERS' not uptodate. Cannot merge.
# error: Submodule '3rdparty/Automodel-workspace/Automodel' could not be updated.
# error: Entry '3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/examples/models/generate_from_hf.py' not uptodate. Cannot merge.
# error: Submodule '3rdparty/Megatron-Bridge-workspace/Megatron-Bridge' could not be updated.
# error: Entry '3rdparty/Megatron-LM-workspace/Megatron-LM/.github/copy-pr-bot.yaml' not uptodate. Cannot merge.
# error: Submodule '3rdparty/Megatron-LM-workspace/Megatron-LM' could not be updated.
# error: Cannot update submodule:
#         3rdparty/Automodel-workspace/Automodel
#         3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
#         3rdparty/Megatron-LM-workspace/Megatron-LM
git submodule foreach --recursive 'git reset --hard >/dev/null 2>&1 && git clean -fdx >/dev/null 2>&1'

exit $ret_code
