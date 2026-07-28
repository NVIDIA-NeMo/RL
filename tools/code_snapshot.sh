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
  echo2 "Refreshing existing code snapshot in $SNAPSHOT_DIR"
fi

# Always re-copy. Returning early on an existing directory made a rerun execute
# whatever that directory happened to hold, while the caller recorded the
# current source commit as provenance -- so evidence could describe code that
# never ran. rsync is incremental, so refreshing an up-to-date snapshot is
# cheap. Deliberately no --delete: run outputs live under this directory.
echo2 "Copying git-tracked files and submodules..."
rsync -a --files-from=<(
  git ls-files --recurse-submodules --cached --full-name
) ./ $SNAPSHOT_DIR/

# Record what this snapshot was built from, so a stale or hand-edited tree is
# identifiable after the fact.
git rev-parse HEAD > "$SNAPSHOT_DIR/.code_snapshot_source_commit"


# Echo the snapshot directory so the caller can use it to `cd` into it
echo ${SNAPSHOT_DIR}
