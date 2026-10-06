#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --time=01:00:00
#SBATCH --mem=0
set -euo pipefail
: "${REPO:?Set source checkout}"
: "${SOURCE_COMMIT:?Set expected source commit}"
: "${SOURCE_ARCHIVE:?Set shared immutable archive filename}"
cd "$REPO"
git -c fetch.recurseSubmodules=false pull --ff-only
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
git submodule update --init --recursive --checkout --jobs 4
state=$(git submodule status --recursive)
if grep -qE '^[+U-]' <<< "$state"; then
  printf '%s\n' "$state" >&2
  exit 1
fi
test -z "$(git status --porcelain --untracked-files=no --ignore-submodules=none)"
mkdir -p "$(dirname "$SOURCE_ARCHIVE")"
export TMPDIR="/raid/scratch/${USER}/nr-prepare-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
manifest=$(mktemp)
trap 'rm -f "$manifest" "${SOURCE_ARCHIVE}.partial"' EXIT
git ls-files -z --recurse-submodules --cached --full-name > "$manifest"
test ! -e "$SOURCE_ARCHIVE"
tar --null -cf "${SOURCE_ARCHIVE}.partial" -T "$manifest"
mv "${SOURCE_ARCHIVE}.partial" "$SOURCE_ARCHIVE"
printf 'source_commit=%s\n%s\n' "$SOURCE_COMMIT" "$state" > "${SOURCE_ARCHIVE}.metadata.txt"
sha256sum "$SOURCE_ARCHIVE" >> "${SOURCE_ARCHIVE}.metadata.txt"
python3 tools/generate_fingerprint.py > "${SOURCE_ARCHIVE}.fingerprint.json"
