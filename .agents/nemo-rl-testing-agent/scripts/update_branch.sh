#!/usr/bin/env bash
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

# Brings a labeled Megatron-LM PR up to date with its base branch using
# GitHub's "Update branch" button semantics (merge, no history rewrite), then
# reports the resulting head revision.
#
# Usage: update_branch.sh <pr-number>
# Output: KEY=value lines (PR_NUMBER, HEAD_SHA, BASE_SHA, ...) on stdout.
# Exit codes: 0 updated or already current, 2 update refused (conflict, fork
# without maintainer edits, ...) -- the caller should report and skip the PR.

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

pr="${1:?usage: update_branch.sh <pr-number>}"
nrlta_require MEGATRON_REPO MEGATRON_BASE_BRANCH

base_sha_before="$(gh api "repos/${MEGATRON_REPO}/commits/${MEGATRON_BASE_BRANCH}" --jq '.sha')"
head_sha_before="$(gh pr view "${pr}" --repo "${MEGATRON_REPO}" --json headRefOid --jq '.headRefOid')"
# `gh pr update-branch` reports success even when there is nothing to merge, so
# ask GitHub how far behind the branch actually is before touching the PR.
behind="$(gh api "repos/${MEGATRON_REPO}/compare/${base_sha_before}...${head_sha_before}" --jq '.behind_by')"

update_status="merged"
if [[ "${behind}" == "0" ]]; then
  nrlta_log "PR #${pr}: already contains ${MEGATRON_BASE_BRANCH} ${base_sha_before:0:8}; nothing to merge"
  update_status="already-current"
else
  nrlta_log "PR #${pr}: branch is ${behind} commit(s) behind ${MEGATRON_BASE_BRANCH}; merging"
  update_out=""
  if update_out="$(gh pr update-branch "${pr}" --repo "${MEGATRON_REPO}" 2>&1)"; then
    # GitHub advances the ref asynchronously; give it a moment before reading back.
    sleep 5
  elif [[ "${update_out}" == *"up to date"* || "${update_out}" == *"up-to-date"* ]]; then
    update_status="already-current"
  else
    nrlta_log "PR #${pr}: update-branch refused: ${update_out}"
    echo "PR_NUMBER=${pr}"
    echo "UPDATE_STATUS=refused"
    echo "UPDATE_ERROR=${update_out//$'\n'/ }"
    exit 2
  fi
fi

view_json="$(gh pr view "${pr}" --repo "${MEGATRON_REPO}" \
  --json number,url,headRefName,headRefOid,baseRefName,mergeable,mergeStateStatus)"
base_sha="$(gh api "repos/${MEGATRON_REPO}/commits/${MEGATRON_BASE_BRANCH}" --jq '.sha')"

read_field() {
  gh_json_field="$1"
  printf '%s' "${view_json}" | python3 -c "import json,sys;print(json.load(sys.stdin)['${gh_json_field}'])"
}

echo "PR_NUMBER=${pr}"
echo "UPDATE_STATUS=${update_status}"
echo "HEAD_REF=$(read_field headRefName)"
echo "HEAD_SHA=$(read_field headRefOid)"
echo "BASE_REF=$(read_field baseRefName)"
echo "BASE_SHA=${base_sha}"
echo "MERGEABLE=$(read_field mergeable)"
echo "MERGE_STATE=$(read_field mergeStateStatus)"
echo "URL=$(read_field url)"
