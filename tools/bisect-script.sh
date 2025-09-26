#!/usr/bin/env bash

set -euo pipefail

print_usage() {
  cat <<EOF
Usage: GOOD=<good_ref> BAD=<bad_ref> tools/bisect-script.sh [command ...]

Runs a git bisect session between GOOD and BAD to find the first bad commit.

Examples:
  GOOD=v1.2.3 BAD=HEAD tools/bisect-script.sh uv run --group test pytest tests/unit/test_foobar.py
  GOOD=56a6225 BAD=32faafa tools/bisect-script.sh uv run --group dev pre-commit run --all-files

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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if [[ -z "${GOOD:-}" || -z "${BAD:-}" ]]; then
  echo "ERROR: GOOD and BAD environment variables are required." >&2
  echo >&2
  print_usage >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: Not inside a git repository." >&2
  exit 2
fi

# Ensure there is a command to run
if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing command to evaluate during bisect." >&2
  echo >&2
  print_usage >&2
  exit 2
fi

USER_CMD=("$@")

# Require a clean working tree
git update-index -q --refresh || true
if ! git diff --quiet; then
  echo "ERROR: Unstaged changes present. Commit or stash before bisect." >&2
  exit 2
fi
if ! git diff --cached --quiet; then
  echo "ERROR: Staged changes present. Commit or stash before bisect." >&2
  exit 2
fi

# On interruption or script error, print helpful message
on_interrupt_or_error() {
  local status=$?
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git bisect log >/dev/null 2>&1; then
      echo "[bisect] Script interrupted or failed (exit ${status})." >&2
      echo "[bisect] Restoring original state with 'git bisect reset' on exit." >&2
    fi
  fi
}
trap on_interrupt_or_error INT TERM ERR

# Always reset bisect on exit to restore original state
cleanup_reset() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git bisect log >/dev/null 2>&1; then
      git bisect reset >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup_reset EXIT

# Reset any ongoing bisect unless user asked to keep it
if git bisect log >/dev/null 2>&1; then
  # We are in a bisect session
  if [[ -z "${BISECT_NO_RESET:-}" ]]; then
    git bisect reset >/dev/null 2>&1 || true
  fi
fi

set -x
git bisect start "$BAD" "$GOOD"
set +x

# Detect immediate conclusion (no midpoints to test)
if git bisect log >/dev/null 2>&1; then
  if git bisect log | grep -q "first bad commit:"; then
    echo "[bisect] Immediate conclusion from endpoints; no midpoints to test."
    echo "[bisect] --- bisect log ---"
    git bisect log | cat
    echo "[bisect] --- bisect visualize (oneline) ---"
    GIT_PAGER=cat git bisect visualize --oneline --decorate -n 20 | cat || true
    exit 0
  fi
fi

set -x
git bisect run "${USER_CMD[@]}"
RUN_STATUS=$?
set +x

# Show bisect details before cleanup
if git bisect log >/dev/null 2>&1; then
  echo "[bisect] --- bisect log ---"
  git bisect log | cat
  echo "[bisect] --- bisect visualize (oneline) ---"
  GIT_PAGER=cat git bisect visualize --oneline --decorate -n 20 | cat || true
fi

exit $RUN_STATUS


