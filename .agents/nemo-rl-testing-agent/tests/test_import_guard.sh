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

# Exercises the megatron import guard in prep_container.sh against a fake
# /opt/ray_venvs tree.
#
# The guard runs inside a container on an allocated GPU node, so a bug in it
# costs a full cluster round trip to observe -- and two such bugs (a `for` loop
# whose last command was a failing `[ -d ]`, and an assignment from a failing
# command substitution) aborted the run under `set -e` with no diagnostic at
# all. Both shapes are covered below. Run this after touching the guard:
#
#   bash .agents/nemo-rl-testing-agent/tests/test_import_guard.sh

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${here}/../scripts/prep_container.sh"
ROOT=/tmp/nrlta-guard-test
MCORE="${ROOT}/mcore"
BRIDGE="${ROOT}/bridge"
GUARD="${ROOT}.guard.sh"

# The guard is the tail of prep_container.sh, from the venv_root assignment on.
# Extract it rather than duplicating it, so the test always runs the real code.
guard_start="$(grep -n 'venv_root="${NEMO_RL_VENV_DIR' "${SCRIPT}" | head -1 | cut -d: -f1)"
if [ -z "${guard_start}" ]; then
  echo "FAIL: could not locate the guard block in ${SCRIPT}" >&2
  exit 1
fi

failures=0

# The guard calls python twice per venv: once for megatron.core, once for the
# megatron.bridge smoke test. The fake dispatches on the -c payload.
make_venv() { # name, mcore_body, bridge_body, site_packages_entry, create_libdir
  local d="${ROOT}/ray_venvs/$1"
  mkdir -p "${d}/bin"
  {
    echo '#!/bin/bash'
    echo 'case "$2" in'
    echo '  *megatron.bridge*)'
    printf '    %s\n' "$3"
    echo '    ;;'
    echo '  *)'
    printf '    %s\n' "$2"
    echo '    ;;'
    echo 'esac'
  } > "${d}/bin/python"
  chmod +x "${d}/bin/python"
  if [ "$5" = "yes" ]; then
    mkdir -p "${d}/lib/python3.13/site-packages"
    [ -n "$4" ] && touch "${d}/lib/python3.13/site-packages/$4"
  fi
}

reset_fixture() {
  rm -rf "${ROOT}"
  mkdir -p "${MCORE}/megatron/core" "${BRIDGE}/src/megatron/bridge" "${ROOT}/ray_venvs"
  sed -n "${guard_start},\$p" "${SCRIPT}" > "${GUARD}"
}

run_guard() {
  local out rc
  out="$(
    set -euo pipefail
    CONTAINER_ROOT="${ROOT}"
    CONTAINER_MCORE_DIR="${MCORE}"
    CONTAINER_BRIDGE_DIR="${BRIDGE}"
    NEMO_RL_VENV_DIR="${ROOT}/ray_venvs"
    orig_sha=aaa new_sha=bbb new_subject=subj MCORE_FETCH_REF=ref
    bridge_sha=ccc bridge_sha_before=ddd BRIDGE_FETCH_REF=ccc
    nemo_rl_sha=eee nemo_rl_sha_before=fff
    export CONTAINER_ROOT CONTAINER_MCORE_DIR CONTAINER_BRIDGE_DIR NEMO_RL_VENV_DIR
    source "${GUARD}" 2>&1
  )" && rc=0 || rc=$?
  printf '%s\n' "${out}"
  return "${rc}"
}

check() { # description, expected_rc, expected_substring
  local desc="$1" want_rc="$2" want_text="$3" out rc
  out="$(run_guard)" && rc=0 || rc=$?
  if [ "${rc}" -ne "${want_rc}" ]; then
    echo "FAIL: ${desc}: expected exit ${want_rc}, got ${rc}"
    printf '%s\n' "${out}" | sed 's/^/    /'
    failures=$((failures + 1))
    return
  fi
  if ! printf '%s' "${out}" | grep -q "${want_text}"; then
    echo "FAIL: ${desc}: output missing '${want_text}'"
    printf '%s\n' "${out}" | sed 's/^/    /'
    failures=$((failures + 1))
    return
  fi
  echo "ok: ${desc}"
}

good_mcore="echo ${MCORE}/megatron/core/__init__.py"
good_bridge="echo BRIDGE_FILE ${BRIDGE}/src/megatron/bridge/__init__.py; echo BRIDGE_IMPORT_OK"

# A good venv must be verified even when surrounded by venvs that have no lib
# directory, no megatron at all, a bridge-only install whose megatron.core import
# fails, and a directory that is not a venv.
reset_fixture
make_venv MegatronPolicyWorker "${good_mcore}" "${good_bridge}" megatron_core-0.1.dist-info yes
make_venv BridgeOnly "echo 'ModuleNotFoundError: megatron.core' >&2; exit 1" "${good_bridge}" megatron_bridge-0.1.dist-info yes
make_venv VllmWorker "echo /somewhere/torch.py" "echo NO_BRIDGE" torch yes
make_venv NoLibDir "echo hi" "echo NO_BRIDGE" "" no
mkdir -p "${ROOT}/ray_venvs/NotAVenv"
check "verifies a good venv and skips the awkward ones" 0 "megatron_worker_venvs_verified=1"

# Importing mcore from anywhere other than the revision under test is the whole
# reason the guard exists.
reset_fixture
make_venv MegatronPolicyWorker "echo /opt/other/megatron/core/__init__.py" "${good_bridge}" megatron_core-0.1.dist-info yes
check "rejects mcore resolved outside the revision under test" 1 "NRLTA_PREP_FAIL"

reset_fixture
make_venv VllmWorker "echo /somewhere/torch.py" "echo NO_BRIDGE" torch yes
make_venv BridgeOnly "exit 1" "${good_bridge}" megatron_bridge-0.1.dist-info yes
check "fails when no venv provides megatron.core" 1 "NRLTA_PREP_FAIL"

reset_fixture
rm -rf "${ROOT}/ray_venvs"
check "fails when the venv root is missing" 1 "NRLTA_PREP_FAIL"

# The regression that motivated pinning Bridge: mcore-under-test is incompatible
# with the Bridge NeMo-RL pins. This must be reported distinctly from a broken
# harness, because it is a real finding about the revision.
reset_fixture
make_venv MegatronPolicyWorker "${good_mcore}" \
  "echo \"ImportError: cannot import name 'get_default_save_sharded_strategy'\" >&2; exit 1" \
  megatron_core-0.1.dist-info yes
check "flags a bridge/mcore incompatibility as INTEGRATION" 1 "NRLTA_PREP_FAIL_INTEGRATION"

# A Bridge imported from outside the pinned checkout means the pin did not take.
reset_fixture
make_venv MegatronPolicyWorker "${good_mcore}" \
  "echo BRIDGE_FILE /opt/somewhere/else/megatron/bridge/__init__.py; echo BRIDGE_IMPORT_OK" \
  megatron_core-0.1.dist-info yes
check "rejects megatron.bridge resolved outside the pinned checkout" 1 "outside the pinned Bridge"

# A venv with mcore but no bridge at all is legitimate.
reset_fixture
make_venv MegatronValueWorker "${good_mcore}" "echo NO_BRIDGE" megatron_core-0.1.dist-info yes
check "accepts an mcore venv that has no bridge" 0 "megatron_worker_venvs_verified=1"

rm -rf "${ROOT}" "${GUARD}"

echo
if [ "${failures}" -eq 0 ]; then
  echo "import guard: all checks passed"
else
  echo "import guard: ${failures} check(s) failed"
fi
exit $(( failures > 0 ))
