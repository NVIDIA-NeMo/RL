#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# =============================================================================
# Record, at job start, the code that is actually about to run.
#
# provenance.txt is written on the login node at *submit* time, but with
# USE_SNAPSHOT=0 the source is bind-mounted live and is read at *job start*.
# Under a QOS that serialises large jobs those two moments can be hours and
# several commits apart, and the arms of a sweep then do not all run the same
# code. That already happened: the previous four-arm staleness sweep ran a
# mixture, and the low end of the trend was confounded by it.
#
# Two independent records, because either alone can mislead:
#
#   git    -- the host repo is visible through the /lustre mount, so its HEAD
#             is readable from inside the container. It is the useful handle
#             for "which commit", but it describes the checkout as of now, not
#             necessarily what was imported, and it is silent about uncommitted
#             edits and about which paths are actually mounted.
#   digest -- a content hash of the mounted source, which is ground truth for
#             what this job will import regardless of git state. Two arms with
#             the same digest provably ran the same code.
#
# Best effort by construction: this runs inside the training command, so a
# failure here must never take the job with it. Every step is guarded and the
# script always exits 0.
#
# Usage: record_code_at_start.sh <code_root> <run_dir> [repo_root]
# =============================================================================

CODE_ROOT="${1:?code root required}"
RUN_DIR="${2:?run dir required}"
REPO_ROOT="${3:-}"

OUT="${RUN_DIR}/code_at_start.txt"

{
  echo "recorded_at: $(date -Iseconds)"
  echo "hostname: $(hostname 2>/dev/null || echo unknown)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "code_root: ${CODE_ROOT}"

  if [[ -n "${REPO_ROOT}" ]]; then
    echo "repo_root: ${REPO_ROOT}"
    echo "git_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_branch: $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unavailable)"
    # Truncated: a long dirty list would bury the hashes below it.
    echo "git_dirty: $(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | head -20 | tr '\n' ';' || echo unavailable)"
  fi

  # Hash what will be imported. Scoped to the three trees ultra_launch.sh and
  # nano35_dolphin_launch.sh actually overlay onto the image -- hashing all of
  # CODE_ROOT would pull in the container's own site-packages and swamp the
  # signal with bytes no launcher can change.
  #
  # The file count is printed with the digest, and the digest of empty input is
  # rejected outright, because the failure mode here is silent agreement: if
  # the pipeline below dies (it did, on a login node at its PID ceiling) every
  # command in it still exits and sha256sum still prints -- the hash of
  # nothing. That is a well-formed hash that every arm would report
  # identically, so the one record meant to prove the arms matched would be the
  # record asserting it most confidently while measuring nothing at all.
  empty_digest=$(printf '' | sha256sum | cut -d' ' -f1)
  for tree in "${CODE_ROOT}/nemo_rl" "${CODE_ROOT}/examples/nemo_gym" "${CODE_ROOT}/examples/configs"; do
    if [[ ! -d "${tree}" ]]; then
      echo "digest ${tree}: missing"
      continue
    fi
    listing=$(find "${tree}" -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' \) -print 2>/dev/null | LC_ALL=C sort)
    count=$(printf '%s\n' "${listing}" | grep -c . || true)
    digest=$(printf '%s\n' "${listing}" | tr '\n' '\0' | xargs -0 -r sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
    if [[ -z "${digest}" || "${digest}" == "${empty_digest}" || "${count}" -eq 0 ]]; then
      echo "digest ${tree}: UNAVAILABLE (hashing failed; files=${count}) — do not treat arms as matched on this record"
    else
      echo "digest ${tree}: ${digest} (files=${count})"
    fi
  done
} >"${OUT}" 2>&1

echo "code provenance at start recorded to ${OUT}"
cat "${OUT}" 2>/dev/null

exit 0
