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

# When we bisect, we need to ensure that the venvs are refreshed b/c the commit could
# habe changed the uv.lock or 3rdparty submoduels, so we need to force a rebuild to be safe
export NRL_FORCE_REBUILD_VENVS=true
print_usage() {
  cat <<EOF
Usage: GOOD=<good_ref> BAD=<bad_ref> tools/bisect-script.sh [command ...]

Runs a git bisect session between GOOD and BAD to find the first bad commit.
Sets NRL_FORCE_REBUILD_VENVS=true to ensure test environments are rebuilt to match commit's uv.lock.

Examples:
  GOOD=56a6225 BAD=32faafa tools/bisect-script.sh uv run --group dev pre-commit run --all-files
  GOOD=464ed38 BAD=c843f1b tools/bisect-script.sh uv run --group test pytest tests/unit/test_foobar.py

  # Example ouptut:
  #    1. Will run until hits the first bad commit.
  #    2. Will show the bisect log (what was run) and visualize the bisect.
  #    3. Reset git bisect state to return you to the git state you were originally.
  #
  #    25e05a3d557dfe59a14df43048e16b6eea04436e is the first bad commit
  #    commit 25e05a3d557dfe59a14df43048e16b6eea04436e
  #    Author: Terry Kong <terryk@nvidia.com>
  #    Date:   Fri Sep 26 17:24:45 2025 +0000
  #
  #        3==4
  #
  #        Signed-off-by: Terry Kong <terryk@nvidia.com>
  #
  #     tests/unit/test_foobar.py | 2 +-
  #     1 file changed, 1 insertion(+), 1 deletion(-)
  #    bisect found first bad commit
  #    + RUN_STATUS=0
  #    + set +x
  #    [bisect] --- bisect log ---
  #    # bad: [c843f1b994cb7e331aa8bc41c3206a6e76e453ef] try echo
  #    # good: [464ed38e68dcd23f0c1951784561dc8c78410ffe] add passing foobar
  #    git bisect start 'c843f1b' '464ed38'
  #    # good: [8b8b3961e9cdbc1b4a9b6a912f7d36d117952f62] try visualize
  #    git bisect good 8b8b3961e9cdbc1b4a9b6a912f7d36d117952f62
  #    # bad: [25e05a3d557dfe59a14df43048e16b6eea04436e] 3==4
  #    git bisect bad 25e05a3d557dfe59a14df43048e16b6eea04436e
  #    # good: [c82e0b69d52b8e1641226c022cb487afebe8ba99] 2==2
  #    git bisect good c82e0b69d52b8e1641226c022cb487afebe8ba99
  #    # first bad commit: [25e05a3d557dfe59a14df43048e16b6eea04436e] 3==4
  #    [bisect] --- bisect visualize (oneline) ---
  #    25e05a3d (HEAD) 3==4

Exit codes inside the command determine good/bad:
  0 -> good commit
  non-zero -> bad commit
  125 -> skip this commit (per git-bisect convention)

Environment variables:
  GOOD    Commit-ish known to be good (required)
  BAD     Commit-ish suspected bad (required)
  (The script will automatically restore the repo state with 'git bisect reset' on exit.)

Notes:
  - The working tree will be reset by git bisect. Ensure you have no uncommitted changes.
  - If GOOD is an ancestor of BAD with 0 or 1 commits in between, git can
    conclude immediately; the script will show the result and exit without
    running your command.
EOF
}

# Minimal color helpers: blue for info, red for errors (TTY-only; NO_COLOR disables)
BLUE=""; RED=""; NC=""
if [[ -z "${NO_COLOR:-}" ]] && { [[ -t 1 ]] || [[ -t 2 ]]; }; then
  BLUE=$'\033[34m'
  RED=$'\033[31m'
  NC=$'\033[0m'
fi

iecho() { printf "%b%s%b\n" "$BLUE" "$*" "$NC"; }
fecho() { printf "%b%s%b\n" "$RED" "$*" "$NC" >&2; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if [[ -z "${GOOD:-}" || -z "${BAD:-}" ]]; then
  fecho "ERROR: GOOD and BAD environment variables are required."
  echo >&2
  print_usage >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  fecho "ERROR: Not inside a git repository."
  exit 2
fi

# Ensure there is a command to run
if [[ $# -lt 1 ]]; then
  fecho "ERROR: Missing command to evaluate during bisect."
  echo >&2
  print_usage >&2
  exit 2
fi

USER_CMD=("$@")

# Require a clean working tree
git update-index -q --refresh || true
if ! git diff --quiet; then
  fecho "ERROR: Unstaged changes present. Commit or stash before bisect."
  exit 2
fi
if ! git diff --cached --quiet; then
  fecho "ERROR: Staged changes present. Commit or stash before bisect."
  exit 2
fi

# On interruption or script error, print helpful message
on_interrupt_or_error() {
  local status=$?
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git bisect log >/dev/null 2>&1; then
      iecho "[bisect] Script interrupted or failed (exit ${status})."
      iecho "[bisect] Restoring original state with 'git bisect reset' on exit."
    fi
  fi
}
trap on_interrupt_or_error INT TERM ERR

# Always reset bisect on exit to restore original state
cleanup_reset() {
  if [[ -n "${BISECT_NO_RESET:-}" ]]; then
    # Respect user's request to not reset the bisect
    return
  fi
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git bisect log >/dev/null 2>&1; then
      git bisect reset >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup_reset EXIT

# Check if we are already in a bisect session
if git bisect log >/dev/null 2>&1; then
  fecho "[bisect] We are already in a bisect session. Please reset the bisect manually if you want to start a new one."
  exit 1
fi

set -x
git bisect start "$BAD" "$GOOD"
set +x

# Detect immediate conclusion (no midpoints to test)
if git bisect log >/dev/null 2>&1; then
  if git bisect log | grep -q "first bad commit:"; then
    iecho "[bisect] Immediate conclusion from endpoints; no midpoints to test."
    iecho "[bisect] --- bisect log ---"
    git bisect log | cat
    exit 0
  fi
fi

set -x
set +e  # Temporarily allow the command to fail to capture the exit status
git bisect run "${USER_CMD[@]}"
RUN_STATUS=$?
set -e
set +x

# Show bisect details before cleanup
if git bisect log >/dev/null 2>&1; then
  iecho "[bisect] --- bisect log ---"
  git bisect log | cat
fi

exit $RUN_STATUS


