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

# Phase control shared by the prefix-recovery functional tests. Source it after
# setting TEST_DIR; a test starts phase one in its own process group and keeps
# its PID in PHASE1_PID.

PHASE1_PID=""

# Hard-kill phase one and wait until its controller can no longer write
# snapshots.
stop_phase1() {
    if [[ -z "${PHASE1_PID}" ]]; then
        return
    fi
    kill -KILL -- "-${PHASE1_PID}" 2>/dev/null || true
    wait "${PHASE1_PID}" 2>/dev/null || true
    PHASE1_PID=""
    # Ray actors outlive the driver until their raylet notices the crash. Do
    # not start phase two while the old controller can still write snapshots.
    local deadline=$((SECONDS + 120))
    while pgrep -f "^ray::SingleControllerActor" >/dev/null; do
        if ((SECONDS >= deadline)); then
            echo "[ERROR] phase-one SingleControllerActor outlived its driver"
            return 1
        fi
        sleep 1
    done
}

# On exit, stop phase one and delete the given checkpoint artifacts if the test
# passed, unless SC_PREFIX_RECOVERY_KEEP_CHECKPOINTS=1.
install_phase_cleanup() {
    PHASE_CLEANUP_PATHS=("$@")
    trap _phase_cleanup EXIT
}

_phase_cleanup() {
    local status=$?
    stop_phase1
    if [[ "${status}" -eq 0 && "${SC_PREFIX_RECOVERY_KEEP_CHECKPOINTS:-0}" != "1" ]]; then
        rm -rf "${PHASE_CLEANUP_PATHS[@]}"
    else
        echo "Preserving prefix-recovery artifacts for inspection: ${TEST_DIR}"
    fi
    return "${status}"
}
