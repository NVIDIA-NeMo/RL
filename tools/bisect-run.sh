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
# have changed the uv.lock or 3rdparty submodules, so we need to force a rebuild to be safe
export NRL_FORCE_REBUILD_VENVS=true
print_usage() {
  cat <<'EOF'
Usage: GOOD=<good_ref> BAD=<bad_ref> tools/bisect-run.sh [command ...]

Runs a git bisect session between GOOD and BAD to find the first bad commit.
Sets NRL_FORCE_REBUILD_VENVS=true to ensure test environments are rebuilt to match commit's uv.lock.

Additionally, this script will first check out and run your command on the GOOD
commit to verify it actually passes. If it does not, the script aborts early so
you can pick a truly good baseline.

Examples:
  GOOD=56a6225 BAD=32faafa tools/bisect-run.sh uv run --group dev pre-commit run --all-files
  GOOD=464ed38 BAD=c843f1b tools/bisect-run.sh uv run --group test pytest tests/unit/test_foobar.py

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

Example nightly bisect:

rsync -ahP --delete tools/ tools.bisect/  # This copies bisect utilities outside of VCS so we always run the latest copy
TEST_CASE=tests/test_suites/llm/sft-llama3.2-1b-1n8g-fsdp2tp1.v3.sh

HF_HOME=... \
HF_DATASETS_CACHE=... \
CONTAINER=... \
MOUNTS=... \
ACCOUNT=... \
PARTITION=... \
\
SED_CLAUSES=$(cat <<'SED'
s#mean(data\["timing/train/total_step_time"\], -6, -1) < 0\.6#mean(data["timing/train/total_step_time"], -6, -1) < 0.63#
/ray\/node\.0\.gpu\.0\.mem_gb/d
SED
) \
GOOD=$(git log --format="%h" --diff-filter=A -- $TEST_CASE) \
BAD=5b9ab15799c35428c557ab6f8687ec461b69383e \
  tools.bisect/bisect-run.sh tools.bisect/launch-bisect.sh $TEST_CASE

Requirements (ensure submodules update when switching commits):
  Per-repo (recommended inside this repo):
    git config submodule.recurse true
    git config fetch.recurseSubmodules on-demand

  Or set globally:
    git config --global submodule.recurse true
    git config --global fetch.recurseSubmodules on-demand

Exit codes inside the command determine good/bad:
  0 -> good commit
  non-zero -> bad commit
  125 -> skip this commit (per git-bisect convention)

Environment variables:
  GOOD    Commit-ish known to be good (required)
  BAD     Commit-ish suspected bad (required)
  SKIP_GOOD_CHECK  If set to any non-empty value, skip pre-checking the GOOD commit
  (The script will automatically restore the repo state with 'git bisect reset' on exit.)

Additional features:
  - Automatically saves a timestamped git bisect log on failure or interruption
    to '<repo_root>/bisect-logs/'. Override with BISECT_SAVE_DIR.
  - Resume from a prior bisect log via replay:
        BISECT_REPLAY_LOG=/path/to/bisect-YYYYmmdd-HHMMSS-<sha>.log \
          tools.bisect/bisect-run.sh [command ...]
    This will 'git bisect replay' the provided log, then continue with 'git bisect run'.
  - Set BISECT_NO_RESET=1 to keep the bisect state after the script exits.
    By default, the script resets the bisect on exit.

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

# Require GOOD/BAD unless resuming from a bisect replay log
if [[ -z "${BISECT_REPLAY_LOG:-}" ]]; then
  if [[ -z "${GOOD:-}" || -z "${BAD:-}" ]]; then
    fecho "ERROR: GOOD and BAD environment variables are required."
    echo >&2
    print_usage >&2
    exit 2
  fi
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

# Helper to save bisect log to a timestamped file and print its path
save_bisect_log() {
  local reason="${1:-}"
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi
  if ! git bisect log >/dev/null 2>&1; then
    return 0
  fi
  local repo_root
  repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
  local save_dir
  save_dir="${BISECT_SAVE_DIR:-$repo_root/bisect-logs}"
  mkdir -p "$save_dir" || true
  local ts
  ts=$(date '+%Y%m%d-%H%M%S')
  local head
  head=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
  local log_path
  if [[ -n "$reason" ]]; then
    log_path="$save_dir/bisect-${ts}-${head}-${reason}.log"
  else
    log_path="$save_dir/bisect-${ts}-${head}.log"
  fi
  git bisect log >"$log_path" 2>/dev/null || true
  iecho "[bisect] Saved bisect log to: $log_path"
  echo "$log_path"
}

# On interruption or script error, save log and print helpful message
on_interrupt_or_error() {
  local status=$?
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git bisect log >/dev/null 2>&1; then
      iecho "[bisect] Script interrupted or failed (exit ${status})."
      local saved
      saved=$(save_bisect_log "interrupt") || true
      if [[ -n "$saved" ]]; then
        iecho "[bisect] To resume later: BISECT_REPLAY_LOG=$saved <other_env_vars>... tools.bisect/bisect-run.sh ${USER_CMD[@]}"
      fi
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

#############################################
# Ensure submodules are initialized and clean
#############################################
# Unshallow all submodules so we can jump to any submodule commit during bisect
iecho "[bisect] Unshallowing submodules (required to checkout any submodule commit)..."
git submodule foreach 'if [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then echo "Unshallowing..." && git fetch --unshallow; else echo "Already unshallow, skipping"; fi'
# Fetch all branches from all submodule remotes to ensure we have complete history
iecho "[bisect] Fetching all branches from submodule remotes (to ensure all commits are available)..."
git submodule foreach 'git fetch origin +refs/heads/*:refs/remotes/origin/* 2>/dev/null || true'
# Fetch all GitHub PR refs to capture commits that may have been on feature branches
iecho "[bisect] Fetching GitHub PR refs from submodule remotes (to capture PR commits)..."
git submodule foreach 'git fetch origin +refs/pull/*/head:refs/remotes/origin/pr/* 2>/dev/null || true'

# Require submodules to be clean before we begin
if ! git submodule foreach --recursive 'git update-index -q --refresh || true; if ! git diff --quiet || ! git diff --cached --quiet; then echo "$path has local changes"; exit 1; fi'; then
  fecho "ERROR: One or more submodules have local changes."
  iecho "Please commit/stash, or clean them, e.g.:"
  iecho "  git submodule foreach --recursive 'git reset --hard && git clean -fdx'"
  exit 2
fi

#############################################
# Verify required git config for submodules
#############################################
# We need submodules to follow along when switching commits and fetch on-demand
SUBMOD_RECURSE=$(git config --get submodule.recurse || true)
FETCH_RECURSE=$(git config --get fetch.recurseSubmodules || true)

# Normalize to lowercase for comparison
SUBMOD_RECURSE_LC=${SUBMOD_RECURSE,,}
FETCH_RECURSE_LC=${FETCH_RECURSE,,}

SUBMOD_OK=false
case "$SUBMOD_RECURSE_LC" in
  true|1|yes|on)
    SUBMOD_OK=true
    ;;
esac

FETCH_OK=false
if [[ "$FETCH_RECURSE_LC" == "on-demand" ]]; then
  FETCH_OK=true
fi

if [[ "$SUBMOD_OK" != true || "$FETCH_OK" != true ]]; then
  fecho "ERROR: Required git config not set for submodules handling during bisect."
  iecho "Set these (per-repo preferred) before running this script:"
  iecho "  git config submodule.recurse true"
  iecho "  git config fetch.recurseSubmodules on-demand"
  iecho "Or set globally:"
  iecho "  git config --global submodule.recurse true"
  iecho "  git config --global fetch.recurseSubmodules on-demand"
  iecho "Current values: submodule.recurse='${SUBMOD_RECURSE:-<unset>}', fetch.recurseSubmodules='${FETCH_RECURSE:-<unset>}'"
  exit 2
fi

if [[ -n "${BISECT_REPLAY_LOG:-}" ]]; then
  #############################################
  # Resume via git bisect replay
  #############################################
  iecho "[bisect] Resuming from bisect log: '${BISECT_REPLAY_LOG}'"
  git bisect reset >/dev/null 2>&1 || true
  set -x
  git bisect replay "${BISECT_REPLAY_LOG}"
  set +x
  if git bisect log >/dev/null 2>&1; then
    if git bisect log | grep -q "first bad commit:"; then
      iecho "[bisect] Immediate conclusion after replay; no midpoints to test."
      iecho "[bisect] --- bisect log ---"
      git bisect log | cat
      exit 0
    fi
  fi
else
  #############################################
  # Pre-check: verify GOOD commit is actually good
  #############################################
  if [[ -z "${SKIP_GOOD_CHECK:-}" ]]; then
    iecho "[bisect] Verifying GOOD commit '${GOOD}' returns exit code 0 before starting bisect..."

    git checkout "$GOOD"

    set -x
    set +e
    "${USER_CMD[@]}"
    GOOD_STATUS=$?
    set -e
    set +x

    # Restore original ref regardless of outcome (manually clean )
    git submodule foreach --recursive 'git reset --hard >/dev/null 2>&1 && git clean -fdx >/dev/null 2>&1'
    git checkout -

    if [[ $GOOD_STATUS -ne 0 ]]; then
      fecho "ERROR: Command failed on GOOD commit ($GOOD) with exit code $GOOD_STATUS."
      fecho "Please choose a GOOD commit where the command succeeds and retry."
      exit 2
    fi
  else
    iecho "[bisect] Skipping GOOD commit verification (SKIP_GOOD_CHECK is set)"
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

# On non-zero status, save a resumable log and print a hint
if [[ $RUN_STATUS -ne 0 ]]; then
  saved_after_run=$(save_bisect_log "run-exit-${RUN_STATUS}") || true
  if [[ -n "$saved_after_run" ]]; then
    iecho "[bisect] To resume later: BISECT_REPLAY_LOG=$saved_after_run tools.bisect/bisect-run.sh ${USER_CMD[@]}"
  fi
fi

exit $RUN_STATUS


