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

# Runs INSIDE the container, on every node, before the suite starts.
#
# Overlays the synced worktree onto the image's NeMo-RL checkout and points the
# editable megatron-core submodule at the Megatron-LM revision under test, then
# proves that `import megatron.core` really resolves to that checkout.
#
# Configured entirely through the environment (no config.env, no git metadata in
# the synced workspace to rely on):
#   CONTAINER_ROOT, CONTAINER_MCORE_DIR, MEGATRON_CLONE_URL, MCORE_FETCH_REF,
#   MCORE_EXPECTED_SHA (optional), NRLTA_ARTIFACT_DIR (optional).

set -euo pipefail

: "${CONTAINER_ROOT:=/opt/nemo-rl}"
: "${CONTAINER_MCORE_DIR:?CONTAINER_MCORE_DIR must be set}"
: "${MCORE_FETCH_REF:?MCORE_FETCH_REF must be set (e.g. refs/pull/1234/head)}"
: "${MEGATRON_CLONE_URL:=https://github.com/NVIDIA/Megatron-LM.git}"

workspace="$(pwd)"
git config --global --add safe.directory '*' || true

# Cluster nodes occasionally cannot reach github.com for a minute or two; a
# single failed fetch used to abort an otherwise healthy allocation (and on L2
# the head re-runs prep after setup already succeeded, doubling the exposure).
nrlta_git_fetch() {
  local dir="$1" url="$2" ref="$3"
  local attempt=1 delay
  while [ "${attempt}" -le 4 ]; do
    if git -C "${dir}" fetch --depth=1 --force "${url}" "${ref}"; then
      return 0
    fi
    delay=$(( attempt * 15 ))
    echo "[nrlta-prep] git fetch ${ref} from ${url} failed (attempt ${attempt}/4); retrying in ${delay}s"
    sleep "${delay}"
    attempt=$(( attempt + 1 ))
  done
  echo "NRLTA_PREP_FAIL: git fetch ${ref} from ${url} failed after 4 attempts"
  return 1
}

echo "===NRLTA_PREP_BEGIN node=$(hostname)==="

# 1. Establish which NeMo-RL is under test.
#
#    Two modes. With NEMO_RL_FETCH_REF set, take the named revision's SOURCE so
#    the run is reproducible from a sha and picks up fixes that are raised but
#    not yet merged. Without it, overlay the synced worktree, which is how a
#    candidate fix gets exercised before it is pushed.
#
#    Source only -- deliberately not `git reset --hard`. The image's Ray worker
#    venvs under /opt/ray_venvs were resolved from the image's `uv.lock`, and
#    swapping in another revision's lock file makes uv rebuild them against
#    different pins than the driver venv still has. That is not theoretical: the
#    first attempt did exactly this and every test died in 29 seconds on
#    `AttributeError: Can't get attribute '_get_opentelemetry' on
#    ray.util.tracing.tracing_helper`, a driver/worker Ray mismatch that looks
#    nothing like its cause. Dependency metadata therefore stays at the image's
#    revision; only the Python sources and test scripts move.
#
#    This has to happen BEFORE the Bridge and mcore checkouts: Bridge lives under
#    CONTAINER_ROOT/3rdparty, and touching NeMo-RL afterwards risks their pins.
#    Outermost repo first, innermost last.
NEMO_RL_SOURCE_PATHS="nemo_rl tests examples"
nemo_rl_sha_before="$(git -C "${CONTAINER_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
nemo_rl_sha=""
if [ -n "${NEMO_RL_FETCH_REF:-}" ]; then
  if ! git -C "${CONTAINER_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "NRLTA_PREP_FAIL: ${CONTAINER_ROOT} is not a git checkout, so the NeMo-RL"
    echo "  revision under test cannot be pinned. Re-run with the worktree overlay."
    exit 1
  fi
  nrlta_git_fetch "${CONTAINER_ROOT}" "${NEMO_RL_CLONE_URL}" "${NEMO_RL_FETCH_REF}"
  nemo_rl_sha="$(git -C "${CONTAINER_ROOT}" rev-parse FETCH_HEAD)"

  if [ -n "${NEMO_RL_EXPECTED_SHA:-}" ] && [ "${nemo_rl_sha}" != "${NEMO_RL_EXPECTED_SHA}" ]; then
    echo "NRLTA_PREP_FAIL: NeMo-RL fetched ${nemo_rl_sha} but caller expected"
    echo "  ${NEMO_RL_EXPECTED_SHA} (the ref moved mid-submission; re-run)"
    exit 1
  fi

  # Remove first, so a file the revision under test deleted does not survive from
  # the image and get imported anyway.
  for dir in ${NEMO_RL_SOURCE_PATHS}; do
    rm -rf "${CONTAINER_ROOT:?}/${dir}"
  done
  # shellcheck disable=SC2086
  git -C "${CONTAINER_ROOT}" checkout FETCH_HEAD -- ${NEMO_RL_SOURCE_PATHS}

  # Prove it landed, rather than trusting that checkout did what was asked.
  # shellcheck disable=SC2086
  if ! git -C "${CONTAINER_ROOT}" diff --quiet FETCH_HEAD -- ${NEMO_RL_SOURCE_PATHS}; then
    echo "NRLTA_PREP_FAIL: ${CONTAINER_ROOT} sources still differ from ${nemo_rl_sha}"
    # shellcheck disable=SC2086
    git -C "${CONTAINER_ROOT}" diff --stat FETCH_HEAD -- ${NEMO_RL_SOURCE_PATHS} | tail -20
    exit 1
  fi
  echo "nemo_rl_mode=pinned"
else
  # The nightly image ships an EMPTY tests/functional, and local edits under
  # nemo_rl/ or examples/ must be the ones exercised by the run.
  mkdir -p "${CONTAINER_ROOT}/tests/functional"
  for dir in ${NEMO_RL_SOURCE_PATHS}; do
    if [ -d "${workspace}/${dir}" ]; then
      mkdir -p "${CONTAINER_ROOT}/${dir}"
      cp -rf "${workspace}/${dir}/." "${CONTAINER_ROOT}/${dir}/"
    fi
  done
  echo "nemo_rl_mode=worktree"
fi

# 2. Pin Megatron-Bridge next, because CONTAINER_MCORE_DIR is a submodule INSIDE
#    it -- doing this after the mcore checkout would reset mcore back to Bridge's
#    own pin. The image's Bridge is only as new as the image, and pairing a stale
#    Bridge with a current megatron-core fails every test at import time.
bridge_sha_before=""
bridge_sha=""
if [ -n "${BRIDGE_FETCH_REF:-}" ]; then
  if [ ! -d "${CONTAINER_BRIDGE_DIR:-}" ]; then
    echo "NRLTA_PREP_FAIL: CONTAINER_BRIDGE_DIR='${CONTAINER_BRIDGE_DIR:-}' does not exist in this image"
    exit 1
  fi
  bridge_sha_before="$(git -C "${CONTAINER_BRIDGE_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  nrlta_git_fetch "${CONTAINER_BRIDGE_DIR}" "${MEGATRON_BRIDGE_CLONE_URL}" "${BRIDGE_FETCH_REF}"
  git -C "${CONTAINER_BRIDGE_DIR}" reset --hard FETCH_HEAD
  # -fd, never -ff: -ff deletes nested git repositories, which would wipe the
  # Megatron-LM submodule checkout living under this directory.
  git -C "${CONTAINER_BRIDGE_DIR}" clean -fd
  bridge_sha="$(git -C "${CONTAINER_BRIDGE_DIR}" rev-parse HEAD)"
  echo "bridge_sha_before=${bridge_sha_before}"
  echo "bridge_sha=${bridge_sha}"
else
  echo "[nrlta-prep] BRIDGE_FETCH_REF unset; keeping the image's Megatron-Bridge"
fi

# 3. Check out the Megatron-LM revision under test. megatron-core is installed
#    editable from this path, so the checkout is the install.
if [ ! -d "${CONTAINER_MCORE_DIR}" ]; then
  echo "NRLTA_PREP_FAIL: ${CONTAINER_MCORE_DIR} does not exist in this image"
  exit 1
fi

orig_sha="$(git -C "${CONTAINER_MCORE_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
# Ask git, don't test for a .git directory: in a submodule checkout .git is a
# FILE containing a gitdir: pointer, and clobbering it with `git init` would
# silently detach the checkout from its real object store.
if ! git -C "${CONTAINER_MCORE_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[nrlta-prep] ${CONTAINER_MCORE_DIR} is not a git checkout; initializing one in place"
  git -C "${CONTAINER_MCORE_DIR}" init -q
fi

# L2 Ray runs prep twice on the head (setup-command + driver). If setup already
# landed the expected tip, skip the second github fetch — cluster egress to
# github.com flakes often enough that the re-fetch alone has aborted healthy jobs.
if [ -n "${MCORE_EXPECTED_SHA:-}" ] && [ "${orig_sha}" = "${MCORE_EXPECTED_SHA}" ]; then
  echo "[nrlta-prep] mcore already at expected ${MCORE_EXPECTED_SHA}; skipping fetch"
  new_sha="${orig_sha}"
  new_subject="$(git -C "${CONTAINER_MCORE_DIR}" log -1 --pretty=%s)"
else
  nrlta_git_fetch "${CONTAINER_MCORE_DIR}" "${MEGATRON_CLONE_URL}" "${MCORE_FETCH_REF}"
  git -C "${CONTAINER_MCORE_DIR}" reset --hard FETCH_HEAD
  # -ffd (not -x) removes stale sources from the previous revision while keeping
  # gitignored build outputs such as the compiled datasets helpers.
  git -C "${CONTAINER_MCORE_DIR}" clean -ffd
  find "${CONTAINER_MCORE_DIR}" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  new_sha="$(git -C "${CONTAINER_MCORE_DIR}" rev-parse HEAD)"
  new_subject="$(git -C "${CONTAINER_MCORE_DIR}" log -1 --pretty=%s)"
fi

if [ -n "${MCORE_EXPECTED_SHA:-}" ] && [ "${new_sha}" != "${MCORE_EXPECTED_SHA}" ]; then
  echo "NRLTA_PREP_FAIL: checked out ${new_sha} but caller expected ${MCORE_EXPECTED_SHA}"
  echo "  (the PR branch moved between discovery and submission; re-run discovery)"
  exit 1
fi

# 4. Prove the revision under test is the one that will actually be imported.
#    megatron-core is NOT in the driver venv (/opt/nemo_rl_venv) by design: NeMo-RL
#    runs each worker class in its own uv venv under NEMO_RL_VENV_DIR
#    (/opt/ray_venvs in the image), and only the megatron worker venvs carry
#    megatron-core -- editable-installed from CONTAINER_MCORE_DIR, which is why a
#    plain checkout there is enough to change what the workers run.
cd "${CONTAINER_ROOT}"
venv_root="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
echo "worker_venv_root=${venv_root}"

if [ ! -d "${venv_root}" ]; then
  echo "NRLTA_PREP_FAIL: worker venv root ${venv_root} does not exist, so there is"
  echo "  no environment in which to prove the PR revision is imported."
  exit 1
fi

echo "[nrlta-prep] venvs present under ${venv_root}:"
ls -1 "${venv_root}" 2>&1 | sed 's/^/  /'

# nullglob so a non-matching venv/site-packages glob yields nothing instead of a
# literal unexpanded path (which reads as "not a directory" and, as the last
# command in a loop body, would trip `set -e` with no diagnostic at all).
shopt -s nullglob

checked=0
resolved=""
for venv in "${venv_root}"/*; do
  [ -x "${venv}/bin/python" ] || continue
  venv_name="$(basename "${venv}")"

  site_packages=""
  for candidate in "${venv}"/lib/python*/site-packages "${venv}"/lib64/python*/site-packages; do
    if [ -d "${candidate}" ]; then
      site_packages="${candidate}"
    fi
  done
  if [ -z "${site_packages}" ]; then
    echo "[nrlta-prep] ${venv_name}: no site-packages directory, skipping"
    continue
  fi
  # Cheap prefilter: a venv with no megatron distribution at all cannot import
  # megatron.core. It deliberately also matches megatron-bridge, so the import
  # below -- not the name -- is what decides.
  if ! ls "${site_packages}" | grep -qi megatron; then
    echo "[nrlta-prep] ${venv_name}: no megatron distribution, skipping"
    continue
  fi

  # Keep the assignment off `set -e`: a failing command substitution would
  # otherwise abort the script and discard the traceback it just captured.
  import_rc=0
  import_out="$("${venv}/bin/python" -c 'import megatron.core as m; print(m.__file__)' 2>&1)" || import_rc=$?
  if [ "${import_rc}" -ne 0 ]; then
    # Expected for venvs carrying only megatron-bridge: they never run mcore.
    echo "[nrlta-prep] ${venv_name}: has a megatron distribution but cannot import megatron.core, skipping"
    printf '%s\n' "${import_out}" | tail -3 | sed 's/^/    /'
    continue
  fi

  candidate_path="$(printf '%s\n' "${import_out}" | tail -1)"
  echo "megatron_venv=${venv_name} resolves=${candidate_path}"
  case "${candidate_path}" in
    "${CONTAINER_MCORE_DIR}"/*)
      checked=$((checked + 1))
      resolved="${candidate_path}"
      ;;
    *)
      echo "NRLTA_PREP_FAIL: worker venv ${venv_name} imports megatron.core from"
      echo "  ${candidate_path}"
      echo "  which is not the revision under test at ${CONTAINER_MCORE_DIR}"
      exit 1
      ;;
  esac

  # Integration smoke test: run the import chain a megatron worker actually
  # performs. Every L1 test once failed 13 minutes into a run on
  # `megatron.bridge.training.checkpointing`; doing it here costs seconds and
  # names the incompatible symbol directly.
  smoke_rc=0
  smoke_out="$("${venv}/bin/python" -c '
import importlib, sys
try:
    import megatron.bridge as bridge
except ModuleNotFoundError:
    print("NO_BRIDGE")
    sys.exit(0)
print("BRIDGE_FILE", bridge.__file__)
importlib.import_module("megatron.bridge.training.checkpointing")
print("BRIDGE_IMPORT_OK")
' 2>&1)" || smoke_rc=$?

  if [ "${smoke_rc}" -ne 0 ]; then
    # Distinct marker: mcore-under-test is incompatible with the Bridge NeMo-RL
    # pins. That is a finding about the revision, not a broken harness, so it
    # must still be baselined against mcore main before blaming a PR.
    echo "NRLTA_PREP_FAIL_INTEGRATION: ${venv_name} cannot import megatron.bridge.training.checkpointing"
    echo "  against megatron-core ${new_sha}:"
    printf '%s\n' "${smoke_out}" | tail -20 | sed 's/^/    /'
    exit 1
  fi

  bridge_file="$(printf '%s\n' "${smoke_out}" | awk '/^BRIDGE_FILE /{print $2}')"
  if [ -n "${bridge_file}" ]; then
    echo "bridge_venv=${venv_name} resolves=${bridge_file}"
    if [ -n "${CONTAINER_BRIDGE_DIR:-}" ]; then
      case "${bridge_file}" in
        "${CONTAINER_BRIDGE_DIR}"/*) ;;
        *)
          echo "NRLTA_PREP_FAIL: ${venv_name} imports megatron.bridge from"
          echo "  ${bridge_file}"
          echo "  which is outside the pinned Bridge at ${CONTAINER_BRIDGE_DIR}"
          exit 1
          ;;
      esac
    fi
  fi
done

shopt -u nullglob

if [ "${checked}" -eq 0 ]; then
  echo "NRLTA_PREP_FAIL: no venv under ${venv_root} provides megatron-core, so the"
  echo "  PR revision cannot be proven to be under test."
  exit 1
fi
echo "megatron_worker_venvs_verified=${checked}"

if [ -n "${NRLTA_ARTIFACT_DIR:-}" ]; then
  mkdir -p "${NRLTA_ARTIFACT_DIR}"
fi

# The clone URLs go in the record too, so a reader of the report can follow a sha
# to the commit it names. Which repository each one lives in is not guessable
# from the report: NeMo-RL is fetched from a fork whenever the integration branch
# is in play, and a bare sha with no repository behind it is a string nobody can
# check.
echo "nemo_rl_url=${NEMO_RL_CLONE_URL:-}"
echo "mcore_url=${MEGATRON_CLONE_URL:-}"
echo "bridge_url=${MEGATRON_BRIDGE_CLONE_URL:-}"
echo "nemo_rl_fetch_ref=${NEMO_RL_FETCH_REF:-<worktree overlay>}"
echo "nemo_rl_sha_before=${nemo_rl_sha_before}"
echo "nemo_rl_sha=${nemo_rl_sha:-${nemo_rl_sha_before}}"
# The environment (uv.lock, pyproject.toml, prebuilt venvs) is always the image's,
# never the pinned revision's. Anyone comparing two runs needs to know that.
echo "nemo_rl_env_sha=${nemo_rl_sha_before}"
echo "mcore_fetch_ref=${MCORE_FETCH_REF}"
echo "mcore_sha_before=${orig_sha}"
echo "mcore_sha=${new_sha}"
echo "mcore_subject=${new_subject}"
echo "megatron_core_file=${resolved}"
echo "bridge_fetch_ref=${BRIDGE_FETCH_REF:-<image default>}"
echo "bridge_sha=${bridge_sha:-${bridge_sha_before:-unknown}}"
echo "===NRLTA_PREP_END node=$(hostname)==="
