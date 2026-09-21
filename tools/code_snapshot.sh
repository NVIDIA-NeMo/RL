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
# Written last, so its presence means the copy below ran to completion. The
# directory used to be created up front and treated as reusable from then on,
# which meant a run that died between the mkdir and the end of the rsync left a
# directory that exists but is missing files. Every later run for the same
# experiment name then reused that partial tree -- for instance failing at
# `sbatch` because ray.sub had never been copied. Checking for the marker rather
# than for the directory also repairs snapshots left behind before this change.
SNAPSHOT_MARKER="$SNAPSHOT_DIR/.snapshot_complete"

if [[ -f "$SNAPSHOT_MARKER" ]]; then
  echo2 "Using existing code snapshot in $SNAPSHOT_DIR"
  # Echo the snapshot directory so the caller can use it to `cd` into it
  echo ${SNAPSHOT_DIR}
  exit
elif [[ -d "$SNAPSHOT_DIR" ]]; then
  echo2 "Existing code snapshot in $SNAPSHOT_DIR is unmarked or incomplete; rebuilding it"
else
  echo2 "Creating new code snapshot in $SNAPSHOT_DIR"
fi

# Build in a staging directory and put it in place with a rename, which is
# atomic, so the final path never appears in a half-copied state.
STAGING_DIR="${SNAPSHOT_DIR}.partial.$$"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
trap 'rm -rf "$STAGING_DIR"' EXIT

echo2 "Copying git-tracked files and submodules..."
rsync -a --files-from=<(
  git ls-files --recurse-submodules --cached --full-name
) ./ "$STAGING_DIR"/
touch "$STAGING_DIR/.snapshot_complete"

# A rename cannot replace a non-empty directory, so clear any unmarked remains
# first. Two jobs staging the same experiment concurrently are both complete by
# this point, so whichever lands second simply keeps what it finds.
rm -rf "$SNAPSHOT_DIR"
if mv -T "$STAGING_DIR" "$SNAPSHOT_DIR" 2>/dev/null; then
  trap - EXIT
elif [[ -f "$SNAPSHOT_MARKER" ]]; then
  echo2 "Another complete snapshot of $EXP_NAME appeared while copying; using it"
else
  echo2 "[Error]: Failed to move the staged snapshot into $SNAPSHOT_DIR"
  exit 1
fi

# Echo the snapshot directory so the caller can use it to `cd` into it
echo ${SNAPSHOT_DIR}
