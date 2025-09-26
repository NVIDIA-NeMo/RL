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
  - The script will attempt to reset bisect state on exit unless BISECT_NO_RESET is set.
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
git bisect run "${USER_CMD[@]}"
RUN_STATUS=$?
set +x

exit $RUN_STATUS


