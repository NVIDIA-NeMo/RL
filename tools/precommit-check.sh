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

# Run the GPU-free CI gates (and optionally unit tests) locally, from inside the
# container, before pushing a commit to a PR.
#
# Each stage mirrors a job that GitHub Actions will run anyway:
#   ruff      -> the ruff hooks in .pre-commit-config.yaml
#   lint      -> cicd-main.yml "Lint check" (pre-commit run --all-files)
#   pyrefly   -> cicd-main.yml "Check if any files with zero errors not in whitelist"
#   copyright -> copyright-check.yml
#   lockfile  -> lockfile-check.yml (off by default; slow on a cold uv cache)
#   unit      -> cicd-unit-tests (needs 2 GPUs or a reachable ray cluster)

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")

ALL_STAGES=(ruff lint pyrefly copyright lockfile unit)
DEFAULT_STAGES=(ruff lint pyrefly copyright unit)

declare -A STAGE_DESC=(
    [ruff]="ruff check + ruff format on all tracked Python files"
    [lint]="pre-commit run --all-files (the CI lint gate)"
    [pyrefly]="pyrefly check + pyrefly.toml whitelist audit"
    [copyright]="NVIDIA copyright headers on *.py changed vs the base ref"
    [lockfile]="uv lock --check (root + docker/dynamo)"
    [unit]="pytest unit tests (needs 2 GPUs or a ray cluster)"
)

STAGES=("${DEFAULT_STAGES[@]}")
SKIP=()
FIX=0
BAIL=0
FAST=${FAST:-0}
COPYRIGHT_ALL=0
BASE_REF=${BASE_REF:-origin/main}
UV_EXTRA=""
TEST_TARGETS=()
PYTEST_ARGS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [-- <extra pytest args>]

Stages (default: ${DEFAULT_STAGES[*]}):
  ${ALL_STAGES[*]}

Options:
  --only <a,b,c>    Run only these stages.
  --skip <a,b,c>    Skip these stages.
  --fix             Let the ruff stage rewrite files instead of only reporting.
                    (The lint stage always rewrites: its ruff hooks carry --fix.)
  --bail            Stop at the first failing stage (default: run all, report at end).
  --fast            Use the Lfast exclusion list for the unit stage.
  --extra <name>    uv extra for the unit stage (e.g. mcore, vllm). Default: none.
  --base <ref>      Base ref the copyright stage diffs against. Default: origin/main.
  --copyright-all   Scan the whole tree for copyright headers (has known
                    pre-existing failures) instead of just this branch's diff.
  --tests <target>  pytest target for the unit stage, relative to tests/
                    (repeatable; default: unit/).
  -h, --help        Show this message.

Examples:
  $(basename "$0")                                   # everything but lockfile
  $(basename "$0") --only ruff,lint,pyrefly          # static checks only, no GPUs needed
  $(basename "$0") --fix --only ruff                 # autoformat before committing
  $(basename "$0") --only unit --tests unit/algorithms/ -- -x -k grpo
  $(basename "$0") --only unit --extra mcore --tests unit/models/policy/
EOF
}

split_csv() { echo "${1//,/ }"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) read -r -a STAGES <<<"$(split_csv "$2")"; shift 2 ;;
        --skip) read -r -a SKIP <<<"$(split_csv "$2")"; shift 2 ;;
        --fix) FIX=1; shift ;;
        --bail) BAIL=1; shift ;;
        --fast) FAST=1; shift ;;
        --extra) UV_EXTRA="$2"; shift 2 ;;
        --base) BASE_REF="$2"; shift 2 ;;
        --copyright-all) COPYRIGHT_ALL=1; shift ;;
        --tests) TEST_TARGETS+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; PYTEST_ARGS=("$@"); break ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for stage in "${STAGES[@]}"; do
    if [[ ! " ${ALL_STAGES[*]} " == *" ${stage} "* ]]; then
        echo "[ERROR] Unknown stage '${stage}'. Valid stages: ${ALL_STAGES[*]}" >&2
        exit 2
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found. This script is meant to run inside the NeMo-RL container." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

FAILED=()
PASSED=()
SKIPPED=()

# Runs one stage, keeping going after a failure unless --bail was passed.
run_stage() {
    local name=$1
    local desc=$2
    shift 2
    if [[ " ${SKIP[*]:-} " == *" ${name} "* ]]; then
        SKIPPED+=("${name}")
        echo "==> [${name}] skipped (--skip)"
        return 0
    fi
    echo ""
    echo "======================================================================"
    echo "==> [${name}] ${desc}"
    echo "======================================================================"
    if "$@"; then
        PASSED+=("${name}")
    else
        FAILED+=("${name}")
        echo "==> [${name}] FAILED"
        if [[ ${BAIL} -eq 1 ]]; then
            summarize
            exit 1
        fi
    fi
}

summarize() {
    echo ""
    echo "======================================================================"
    echo "Summary"
    echo "======================================================================"
    [[ ${#PASSED[@]} -gt 0 ]] && echo "  passed:  ${PASSED[*]}"
    [[ ${#SKIPPED[@]} -gt 0 ]] && echo "  skipped: ${SKIPPED[*]}"
    [[ ${#FAILED[@]} -gt 0 ]] && echo "  FAILED:  ${FAILED[*]}"
    # uv.lock goes stale when pyproject or a submodule pointer moves without a relock.
    if ! git diff --quiet HEAD -- pyproject.toml 3rdparty 2>/dev/null; then
        echo ""
        echo "  note: pyproject.toml/3rdparty are modified. lockfile-check.yml will run"
        echo "        on this PR — verify with: $(basename "$0") --only lockfile"
    fi
}

stage_ruff() {
    if [[ ${FIX} -eq 1 ]]; then
        uv run --group dev ruff check --fix $(git ls-files '*.py' '*.pyi') \
            && uv run --group dev ruff format $(git ls-files '*.py' '*.pyi')
    else
        uv run --group dev ruff check $(git ls-files '*.py' '*.pyi') \
            && uv run --group dev ruff format --check $(git ls-files '*.py' '*.pyi')
    fi
}

stage_lint() {
    # Superset of stage_ruff: also runs taplo, pyrefly, the markdown-name check
    # and the recipe config minimize-check. This is the exact CI lint gate.
    # Note its ruff hooks are configured with --fix, so a failure here usually
    # means files were rewritten in place -- re-run to confirm they are clean.
    uv run --group dev pre-commit run --all-files --show-diff-on-failure --color=always
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[hint] pre-commit may have rewritten files in place; review 'git diff' and re-run." >&2
    fi
    return ${rc}
}

# CI requires that every file pyrefly reports zero errors for is listed in
# pyrefly.toml's project-includes, so newly-clean files don't silently escape
# type-checking.
stage_pyrefly() {
    uv run --group dev pyrefly check || return 1

    if ! command -v jq >/dev/null 2>&1; then
        echo "[WARN] jq not found; skipping the pyrefly whitelist audit (CI still runs it)." >&2
        return 0
    fi

    local tracked
    tracked=$(git ls-files 'nemo_rl/**/*.py' 'examples/**/*.py' 'docs/*.py' 'tools/**/*.py')
    local missing=0
    for file in $(uv run --group dev pyrefly check ${tracked} --output-format json \
        | jq -r --slurpfile all_files <(echo "${tracked}" | jq -R -s 'split("\n")[:-1]') \
               --arg pwd "$(pwd)/" \
               '(.errors | group_by(.path) | map({(.[0].path | sub($pwd; "")): length}) | add // {}) as $error_counts | $all_files[0][] | . as $file | if ($error_counts[$file] // 0) == 0 then $file else empty end'); do
        if ! grep -qF "${file}" pyrefly.toml; then
            echo "File ${file} has zero errors but is not in pyrefly.toml in the 'project-includes' list. Please add it to this whitelist."
            missing=$((missing + 1))
        fi
    done
    return $((missing > 0))
}

# tools/copyright.sh scans the whole tree and currently fails on files that
# predate the check, so it is useless as a gate. CI (copyright-check.yml) only
# gates files the PR touches, so scope this stage to the branch diff the same
# way. Use --copyright-all to get the full-tree scan instead.
stage_copyright() {
    if [[ ${COPYRIGHT_ALL} -eq 1 ]]; then
        ./tools/copyright.sh
        return $?
    fi

    local base
    if ! base=$(git merge-base HEAD "${BASE_REF}" 2>/dev/null); then
        echo "[WARN] cannot resolve '${BASE_REF}'; falling back to the full-tree scan." >&2
        echo "[WARN] pass --base <ref> to point at your PR's target branch." >&2
        ./tools/copyright.sh
        return $?
    fi

    # Committed-on-branch, unstaged/staged, and untracked .py files.
    local candidates
    candidates=$( { git diff --name-only --diff-filter=d "${base}" HEAD -- '*.py'
                    git diff --name-only --diff-filter=d HEAD -- '*.py'
                    git ls-files --others --exclude-standard -- '*.py'
                  } | sort -u )

    local missing=()
    local file first_line
    while IFS= read -r file; do
        [[ -n "${file}" ]] || continue
        # Only the directories tools/copyright.sh covers.
        case "${file}" in
            nemo_rl/*|examples/*|tests/*|tools/*|research/*|docs/*.py) ;;
            *) continue ;;
        esac
        [[ -s "${file}" ]] || continue  # empty files need no header
        first_line=$(head -2 "${file}" | grep -iv 'coding=' | head -1)
        if ! echo "${first_line}" | grep -Eiq \
            'Copyright.*NVIDIA CORPORATION.*All rights reserved.|BSD 3-Clause License|Copyright.*Microsoft|Copyright.*The Open AI Team|Copyright.*The Google AI|Copyright.*Facebook'; then
            missing+=("${file}")
        fi
    done <<<"${candidates}"

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Error: files changed on this branch are missing a copyright header:"
        printf '  %s\n' "${missing[@]}"
        echo ""
        echo "Expected first line:"
        echo "# Copyright (c) $(date +%Y), NVIDIA CORPORATION.  All rights reserved."
        return 1
    fi
    echo "Ok: all .py files changed vs ${BASE_REF} start with a copyright notice."
}

stage_lockfile() {
    uv lock --check && uv lock --check --directory docker/dynamo
}

stage_unit() {
    local excluded=()
    if [[ "${FAST}" == "1" ]]; then
        source tests/unit/excluded_unit_tests.sh
        excluded=("${EXCLUDED_UNIT_TESTS[@]}")
    fi

    local uv_args=(run --group test)
    [[ -n "${UV_EXTRA}" ]] && uv_args+=(--extra "${UV_EXTRA}")

    uv "${uv_args[@]}" tests/unit/prepare_unit_test_assets.py || return 1

    local targets=("unit/")
    [[ ${#TEST_TARGETS[@]} -gt 0 ]] && targets=("${TEST_TARGETS[@]}")

    uv "${uv_args[@]}" bash -x ./tests/run_unit.sh \
        "${targets[@]}" ${excluded[@]+"${excluded[@]}"} ${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}
}

for stage in "${STAGES[@]}"; do
    run_stage "${stage}" "${STAGE_DESC[${stage}]}" "stage_${stage}"
done

summarize
[[ ${#FAILED[@]} -eq 0 ]] || exit 1
echo ""
echo "All requested checks passed. Safe to push."
