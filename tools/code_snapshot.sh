#!/bin/bash

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

SNAPSHOT_DIR="$PROJECT_ROOT/${CODE_SNAPSHOT_DIRNAME}/${EXP_NAME}"
if [[ ! -d "$SNAPSHOT_DIR" ]]; then
  echo2 "Creating new code snapshot in $SNAPSHOT_DIR"
  mkdir -p $SNAPSHOT_DIR
else
  echo2 "Using existing code snapshot in $SNAPSHOT_DIR"
  # Echo the snapshot directory so the caller can use it to `cd` into it
  echo ${SNAPSHOT_DIR}
  exit
fi

echo2 "Copying git-tracked files and submodules..."
rsync -a --files-from=<(
  {
    git ls-files
    echo .gitmodules
    git submodule foreach --recursive --quiet 'git ls-files | sed "s|^|$path/|"'
  }
) ./ $SNAPSHOT_DIR/


# Echo the snapshot directory so the caller can use it to `cd` into it
echo ${SNAPSHOT_DIR}
