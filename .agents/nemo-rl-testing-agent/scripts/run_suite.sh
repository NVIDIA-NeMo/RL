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

# Submits one NeMo-RL functional suite to the GPU cluster through `cog`, with a
# specific Megatron-LM revision checked out inside the container.
#
# L1 is a single-node job; L2 is a 2-node `--launcher ray` job (1 generation +
# 1 training node) and needs the nano-3.5 checkpoint mounts.
#
# Usage:
#   run_suite.sh --suite l1|l2 --mcore-ref <ref> [options]
#
#   --suite l1|l2        Which functional suite to run (required).
#   --mcore-ref <ref>    Fully-qualified ref fetched in the container, e.g.
#                        refs/pull/5700/head or refs/heads/main (required).
#   --mcore-sha <sha>    Expected head SHA; the job aborts on mismatch.
#   --bridge-ref <ref>   Megatron-Bridge revision to pin in the container.
#                        Defaults to the sha this NeMo-RL checkout pins, which is
#                        the Bridge/mcore pairing NeMo-RL actually ships. Pass
#                        `image` to keep whatever Bridge the image happens to
#                        carry (only useful for reproducing an old run: a stale
#                        Bridge fails every test at import).
#   --nemo-rl-ref <ref>  NeMo-RL revision to test. Defaults to the integration
#                        branch on the fork, which is `main` plus the agent fixes
#                        that are raised but not yet merged. Pass `worktree` to
#                        test the local checkout instead (the fix loop does this);
#                        that requires a clean tree unless --allow-dirty is given.
#   --allow-dirty        Permit `--nemo-rl-ref worktree` with uncommitted changes,
#                        recording the diff alongside the run.
#   --run-name <name>    cog run name (default nrlta-<suite>-<utc-stamp>). Must be
#                        unique per submission; re-using one is refused unless
#                        --reuse-run-name is passed.
#   --reuse-run-name     Submit under a run name that has already been used,
#                        knowingly overwriting its artifacts.
#   --tests "a b"        Run only these sub-tests (used by the fix loop).
#   --time HH:MM:SS      Override the suite's Slurm time limit.
#   --dry-run            Print the cog invocation without submitting.
#
# `cog submit` blocks until the Slurm job finishes, so run this backgrounded
# and poll the printed COG_LOG.

set -euo pipefail
# shellcheck source=lib.sh
source "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

suite=""
mcore_ref=""
mcore_sha=""
bridge_ref=""
nemo_rl_ref=""
allow_dirty=0
run_name=""
reuse_run_name=0
only_tests=""
time_limit=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) suite="$2"; shift 2 ;;
    --mcore-ref) mcore_ref="$2"; shift 2 ;;
    --mcore-sha) mcore_sha="$2"; shift 2 ;;
    --bridge-ref) bridge_ref="$2"; shift 2 ;;
    --nemo-rl-ref) nemo_rl_ref="$2"; shift 2 ;;
    --allow-dirty) allow_dirty=1; shift ;;
    --run-name) run_name="$2"; shift 2 ;;
    --reuse-run-name) reuse_run_name=1; shift ;;
    --tests) only_tests="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) nrlta_die "unknown argument: $1" ;;
  esac
done

[[ -n "${suite}" ]] || nrlta_die "--suite is required (l1 or l2)"
[[ -n "${mcore_ref}" ]] || nrlta_die "--mcore-ref is required"
nrlta_require MEGATRON_CLONE_URL COG_CLUSTER COG_PARTITION GPUS_PER_NODE \
  CONTAINER_ROOT CONTAINER_MCORE_DIR CLUSTER_ARTIFACTS_ROOT NEMO_RL_REPO_PATH STATE_DIR

case "${suite}" in
  l1)
    suite_rel="${L1_SUITE}"
    nodes="${L1_NODES}"
    time_limit="${time_limit:-${L1_TIME}}"
    launcher=""
    ;;
  l2)
    suite_rel="${L2_SUITE}"
    nodes="${L2_NODES}"
    time_limit="${time_limit:-${L2_TIME}}"
    launcher="ray"
    # The nano-3.5 weights and HF config live outside cog's auto-mounted scratch.
    export COG_EXTRA_MOUNTS="${L2_EXTRA_MOUNTS}"
    ;;
  *)
    nrlta_die "--suite must be l1 or l2 (got '${suite}')"
    ;;
esac

run_name="${run_name:-nrlta-${suite}-$(date -u +%Y%m%d-%H%M%S)}"
artifact_dir="${CLUSTER_ARTIFACTS_ROOT}/${run_name}"
local_run_dir="${STATE_DIR}/runs/${run_name}"
cog_log="${local_run_dir}/cog.log"

# The run name is the identity of one submission: it names the artifact directory
# the per-test logs are written to and the run directory the slurm .out files land
# in. Re-using it puts two revisions' logs in one place, and nothing downstream can
# separate them -- ensure_baseline.sh parses with `cat <slurm_log_dir>/*.out`, which
# would concatenate both into a single results JSON. The prose said to use a fresh
# name per attempt long before this guard existed and a resumed sweep still re-used
# `-a1`, because the operator picking the name is not the one who knows it was
# already spent. So it is checked rather than documented.
if [[ -e "${cog_log}" && "${reuse_run_name}" -eq 0 && "${dry_run}" -eq 0 ]]; then
  nrlta_die "$(printf '%s\n' \
    "run name '${run_name}' was already used (${cog_log} exists)." \
    "Bump the attempt suffix (-a1 -> -a2) so this submission gets its own artifacts," \
    "or pass --reuse-run-name to overwrite the previous one knowingly.")"
fi
mkdir -p "${local_run_dir}"

# Resolve which NeMo-RL is under test. Default to the integration branch: it is
# `main` plus the fixes the agent has already raised, so a break diagnosed on an
# earlier PR does not resurface on every later one while its review is pending.
nemo_rl_ref="${nemo_rl_ref:-${NEMO_RL_INTEGRATION_BRANCH}}"
nemo_rl_sha=""
nemo_rl_url=""
if [[ "${nemo_rl_ref}" == "worktree" ]]; then
  # Local-checkout mode, for iterating on a candidate fix before it is pushed.
  nemo_rl_sha="$(git -C "${NEMO_RL_REPO_PATH}" rev-parse HEAD)"
  dirty="$(git -C "${NEMO_RL_REPO_PATH}" status --porcelain -- nemo_rl examples tests 2>/dev/null || true)"
  if [[ -n "${dirty}" ]]; then
    if [[ "${allow_dirty}" -eq 0 ]]; then
      nrlta_die "$(printf '%s\n' \
        "${NEMO_RL_REPO_PATH} has uncommitted changes under nemo_rl/, examples/ or tests/." \
        "Those get copied into the container, so the run would test something no ref names" \
        "and nobody could reproduce it. Commit them, or pass --allow-dirty to record the" \
        "diff and proceed:" \
        "${dirty}")"
    fi
    # Proceeding knowingly: keep the exact diff next to the run so the result
    # stays interpretable after the worktree moves on.
    git -C "${NEMO_RL_REPO_PATH}" diff -- nemo_rl examples tests > "${local_run_dir}/worktree.patch"
    nemo_rl_sha="${nemo_rl_sha}-dirty"
    nrlta_log "testing a dirty worktree; diff saved to ${local_run_dir}/worktree.patch"
  fi
else
  # The integration branch lives on the fork; everything else comes from upstream.
  if [[ "${nemo_rl_ref}" == "${NEMO_RL_INTEGRATION_BRANCH}" ]]; then
    nemo_rl_url="${NEMO_RL_FORK_URL}"
  else
    nemo_rl_url="${NEMO_RL_CLONE_URL}"
  fi
  nemo_rl_sha="$(git ls-remote "${nemo_rl_url}" "${nemo_rl_ref}" 2>/dev/null | awk 'NR==1{print $1}')"
  if [[ -z "${nemo_rl_sha}" ]]; then
    nrlta_die "$(printf '%s\n' \
      "'${nemo_rl_ref}' does not exist in ${nemo_rl_url}." \
      "If this is the integration branch, create it with sync_integration.sh first," \
      "or pass --nemo-rl-ref worktree to test the local checkout.")"
  fi
fi

# One thing outranks the pin below: a Megatron-Bridge fix this agent has already
# raised and that is still in review. A NeMo-RL fix reaches later runs on its own
# because sync_integration.sh cherry-picks it onto the integration branch, but a
# Bridge commit cannot ride a NeMo-RL branch, so it would otherwise have to be
# remembered as --bridge-ref on every single submit. Forgetting it does not
# produce a missing result, it produces a confidently wrong one: the break shows
# up against whichever PR happens to be under test, weeks after it was diagnosed
# and fixed. The registry knows the branch, so ask it rather than the operator.
if [[ -z "${bridge_ref}" ]]; then
  pending_bridge_ref="$(uv run --script "${NRLTA_SCRIPT_DIR}/known_issues.py" \
    pending-fix-ref --repo "${MEGATRON_BRIDGE_REPO}")" || nrlta_die \
    "could not decide which Megatron-Bridge fix to carry; pass --bridge-ref explicitly"
  if [[ -n "${pending_bridge_ref}" ]]; then
    bridge_ref="${pending_bridge_ref}"
    nrlta_log "carrying unmerged Megatron-Bridge fix '${bridge_ref}' (known-issues registry)"
  fi
fi

# Default the Bridge pin to the one the NeMo-RL *under test* points at, which is
# the combination NeMo-RL ships. Pairing a current megatron-core with a Bridge
# from anywhere else breaks every test deep inside Bridge and reads as the
# author's bug. This has to run after the NeMo-RL revision is resolved, and has
# to read the pin out of that revision rather than out of the local checkout's
# HEAD: those are different commits whenever the operator is on a branch, so
# reading HEAD silently made the Bridge under test a property of the working tree
# instead of the run.
if [[ -z "${bridge_ref}" ]]; then
  if [[ "${nemo_rl_ref}" == "worktree" ]]; then
    bridge_ref="$(git -C "${NEMO_RL_REPO_PATH}" rev-parse "HEAD:${BRIDGE_SUBMODULE_PATH}" 2>/dev/null || true)"
  else
    git -C "${NEMO_RL_REPO_PATH}" fetch -q "${nemo_rl_url}" "${nemo_rl_ref}" 2>/dev/null || true
    bridge_ref="$(git -C "${NEMO_RL_REPO_PATH}" ls-tree "${nemo_rl_sha}" "${BRIDGE_SUBMODULE_PATH}" 2>/dev/null | awk '{print $3}')"
  fi
  [[ -n "${bridge_ref}" ]] || nrlta_die \
    "could not read the Megatron-Bridge pin at ${BRIDGE_SUBMODULE_PATH} from ${nemo_rl_sha}; pass --bridge-ref explicitly"
fi
if [[ "${bridge_ref}" == "image" ]]; then
  bridge_ref=""
fi

nrlta_load_tokens

# Everything the in-container scripts need. Both scripts are dependency-free and
# read only these variables, so the same block works for --command (head node)
# and --setup-command (every node). The HF token is injected separately so it can
# be masked out of anything this script prints.
# cog points PYTHONPATH at the synced workspace, whose 3rdparty submodules are
# empty. Force the image checkout so the driver and its Ray workers agree.
remote_env="export PYTHONPATH='${CONTAINER_ROOT}';"
remote_env+=" export CONTAINER_ROOT='${CONTAINER_ROOT}';"
remote_env+=" export CONTAINER_MCORE_DIR='${CONTAINER_MCORE_DIR}';"
remote_env+=" export MEGATRON_CLONE_URL='${MEGATRON_CLONE_URL}';"
remote_env+=" export MCORE_FETCH_REF='${mcore_ref}';"
remote_env+=" export MCORE_EXPECTED_SHA='${mcore_sha}';"
remote_env+=" export CONTAINER_BRIDGE_DIR='${CONTAINER_BRIDGE_DIR}';"
remote_env+=" export MEGATRON_BRIDGE_CLONE_URL='${MEGATRON_BRIDGE_CLONE_URL}';"
remote_env+=" export BRIDGE_FETCH_REF='${bridge_ref}';"
remote_env+=" export NEMO_RL_CLONE_URL='${nemo_rl_url}';"
remote_env+=" export NEMO_RL_FETCH_REF='${nemo_rl_ref}';"
remote_env+=" export NEMO_RL_EXPECTED_SHA='${nemo_rl_sha}';"
remote_env+=" export NRLTA_ARTIFACT_DIR='${artifact_dir}';"
remote_env+=" export ONLY_TESTS='${only_tests}';"
remote_env+=" export MODEL_DIR='${NANO35_MODEL_DIR}';"
remote_env+=" export HF_DIR='${NANO35_HF_DIR}';"

scripts_rel=".agents/nemo-rl-testing-agent/scripts"
prep_cmd="bash ${scripts_rel}/prep_container.sh"
run_cmd="bash ${scripts_rel}/run_suite_remote.sh '${CONTAINER_ROOT}/${suite_rel}'"

# cog has no --env flag, so the HF token has to travel inside the command string
# (it is needed by the HF-gated sub-tests). Keep the value out of stdout.
token_env="export HF_TOKEN='${HF_TOKEN:-}';"
token_env_masked="export HF_TOKEN='***';"

cog_args=(
  submit
  --repo "${NEMO_RL_REPO_PATH}"
  --cluster-name "${COG_CLUSTER}"
  --run-name "${run_name}"
  --gpus "${GPUS_PER_NODE}"
  --nodes "${nodes}"
  --ntasks-per-node 1
  --partition "${COG_PARTITION}"
  --time "${time_limit}"
  --job-name "${run_name:0:32}"
)

cog_args_masked=("${cog_args[@]}")

if [[ "${launcher}" == "ray" ]]; then
  # Ray fans the suite out across nodes: prep must run everywhere, the suite
  # only on the head (it is the driver).
  cog_args+=(--launcher ray)
  cog_args+=(--setup-command "${token_env} ${remote_env} ${prep_cmd}")
  cog_args+=(--command "${token_env} ${remote_env} ${run_cmd}")
  cog_args_masked+=(--launcher ray)
  cog_args_masked+=(--setup-command "${token_env_masked} ${remote_env} ${prep_cmd}")
  cog_args_masked+=(--command "${token_env_masked} ${remote_env} ${run_cmd}")
else
  cog_args+=(--command "${token_env} ${remote_env} ${prep_cmd} && ${run_cmd}")
  cog_args_masked+=(--command "${token_env_masked} ${remote_env} ${prep_cmd} && ${run_cmd}")
fi

echo "RUN_NAME=${run_name}"
echo "SUITE=${suite}"
echo "MCORE_REF=${mcore_ref}"
echo "MCORE_SHA=${mcore_sha:-<unpinned>}"
echo "BRIDGE_REF=${bridge_ref:-<image default>}"
echo "NEMO_RL_REF=${nemo_rl_ref}"
echo "NEMO_RL_SHA=${nemo_rl_sha}"
echo "ARTIFACT_DIR=${artifact_dir}"
echo "SLURM_LOG_DIR=${CLUSTER_RUNS_ROOT}/${run_name}/slurm"
echo "COG_LOG=${cog_log}"
# cog picks the QOS up from the environment rather than an argument, so it is
# invisible in the submitted command line. Record it, and say so when it is
# missing: the fallback is the cluster default, which is where a run silently
# turns into a multi-hour queue wait for no visible reason.
echo "QOS=${COG_SLURM_QOS:-<cluster default>}"
if [[ -z "${COG_SLURM_QOS:-}" ]]; then
  nrlta_log "COG_SLURM_QOS is unset; this job inherits the cluster's default QOS and may queue for hours"
fi

if [[ "${dry_run}" -eq 1 ]]; then
  printf 'cog'
  printf ' %q' "${cog_args_masked[@]}"
  printf '\n'
  exit 0
fi

# The token also has to stay out of the tee'd log.
if [[ -n "${HF_TOKEN:-}" ]]; then
  cog "${cog_args[@]}" 2>&1 | sed -e "s/${HF_TOKEN}/***/g" | tee "${cog_log}"
else
  cog "${cog_args[@]}" 2>&1 | tee "${cog_log}"
fi
exit "${PIPESTATUS[0]}"
