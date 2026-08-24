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
#
# Decide whether protocol="nvlink" actually moves bytes over NVLink.
#
# Bandwidth could not answer it: at 512MB nvlink reached 16.4 GB/s against
# rdma's 18.3 GB/s -- close enough to be either a slow NVLink path or plain
# RDMA. So take the NIC away instead. Mooncake's device filter is a token list
# matched against discovered HCAs, so a name no device carries leaves topology
# discovery empty, and a transfer that still succeeds cannot have used RDMA.
#
# Arm B is the control: without it, "nvlink fails with no NIC" is ambiguous
# between "RDMA was carrying it" and "the bogus filter broke engine init for
# every protocol".
#
# Both arms invoke the tool as a real file. An earlier version inlined this
# with `python - <<'PY'`, which cannot work: multiprocessing's spawn start
# method re-executes the main module by path, and <stdin> is not a path, so
# every child died in _fixup_main_from_path before running anything.
set -u
export PYTHONPATH=$PWD
export NRL_IGNORE_VERSION_MISMATCH=1

echo "########## A. nvlink, device filter matching no NIC ##########"
MC_LOG_LEVEL=INFO python tools/smoke_nvlink_ipc.py \
  --protocol nvlink --mb 64 --device-name no_such_nic0 2>&1 \
  | grep -viE "FutureWarning|import pynvml" | tail -25

echo
echo "########## B. rdma, same bogus filter (control) ##########"
python tools/smoke_nvlink_ipc.py \
  --protocol rdma --mb 64 --device-name no_such_nic0 2>&1 \
  | grep -viE "FutureWarning|import pynvml" | tail -20

echo
echo "########## C. nvlink, normal discovery (reference) ##########"
python tools/smoke_nvlink_ipc.py --protocol nvlink --mb 64 2>&1 \
  | grep -E "^put:|^get:|^bytes_match|^RESULT:" | tail -6

echo
echo "########## D. is ENABLE_MULTI_PROTOCOL compiled in? ##########"
# The "selectTransport route" literal exists only inside the multi-protocol
# branch, so its absence means the adaptive machinery is not in this build.
SO=/opt/nemo_rl_venv/lib/python3.13/site-packages/mooncake/engine.so
echo -n "  selectTransport route literals: "
strings -a "$SO" 2>/dev/null | grep -c "selectTransport route"
echo -n "  multi-protocol comma-split literals: "
strings -a "$SO" 2>/dev/null | grep -cE "not supported by target segment|No matching buffer for target offset"
