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

set -euo pipefail

: "${IMAGE:?Set IMAGE to the derived Dynamo image tag}"
PLATFORM=${PLATFORM:-linux/amd64}
PUSH=${PUSH:-0}
NEMO_RL_IMAGE=${NEMO_RL_IMAGE:-${IMAGE}-nemo-rl-base}
NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-$(git rev-parse HEAD)}

output=(--load)
if [[ "${PUSH}" == "1" ]]; then
  output=(--push)
fi

docker buildx build \
  --platform "${PLATFORM}" \
  --build-context nemo-rl=. \
  --target release \
  --build-arg SKIP_VLLM_BUILD=1 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_RL_COMMIT="${NEMO_RL_COMMIT}" \
  --tag "${NEMO_RL_IMAGE}" \
  "${output[@]}" \
  --file docker/Dockerfile \
  .

docker buildx build \
  --platform "${PLATFORM}" \
  --build-arg NEMO_RL_BASE_IMAGE="${NEMO_RL_IMAGE}" \
  --tag "${IMAGE}" \
  "${output[@]}" \
  --file docker/dynamo/Dockerfile \
  .
