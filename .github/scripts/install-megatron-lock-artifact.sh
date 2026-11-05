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

if (($# != 4)); then
  echo "Usage: $0 <artifact directory> <NeMo-RL directory> <NeMo-RL commit SHA> <Megatron-LM commit SHA>" >&2
  exit 2
fi

artifact_dir="$1"
nemo_rl_dir="$2"
nemo_rl_commit="$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')"
megatron_lm_commit="$(printf '%s' "$4" | tr '[:upper:]' '[:lower:]')"
megatron_lm_dir="$nemo_rl_dir/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

for commit in "$nemo_rl_commit" "$megatron_lm_commit"; do
  if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Expected full 40-character hexadecimal commit SHAs." >&2
    exit 1
  fi
done

test "$(git -C "$nemo_rl_dir" rev-parse HEAD)" = "$nemo_rl_commit"
test "$(git -C "$megatron_lm_dir" rev-parse HEAD)" = "$megatron_lm_commit"
grep -Fx "nemo-rl-commit=$nemo_rl_commit" "$artifact_dir/manifest.txt"
grep -Fx "megatron-lm-commit=$megatron_lm_commit" "$artifact_dir/manifest.txt"
(
  cd "$artifact_dir"
  sha256sum --check uv.lock.sha256
)
install -m 0644 "$artifact_dir/uv.lock" "$nemo_rl_dir/uv.lock"
