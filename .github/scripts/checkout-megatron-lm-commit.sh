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

if (($# != 2)); then
  echo "Usage: $0 <Megatron-LM directory> <commit SHA>" >&2
  exit 2
fi

megatron_lm_dir="$1"
megatron_lm_commit="${2,,}"
trusted_repository="https://github.com/NVIDIA/Megatron-LM.git"
trusted_ref_namespace="refs/remotes/nvidia-branches"

if [[ ! "$megatron_lm_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Megatron-LM commit must be a full 40-character hexadecimal SHA." >&2
  exit 1
fi

fetch_args=(--no-tags --force --prune --filter=blob:none)
if [[ "$(git -C "$megatron_lm_dir" rev-parse --is-shallow-repository)" == "true" ]]; then
  fetch_args+=(--unshallow)
fi

# Fetch every branch directly from NVIDIA/Megatron-LM. Do not fetch the
# requested SHA: its object must arrive through a branch advertised by the
# trusted repository.
git -C "$megatron_lm_dir" fetch "${fetch_args[@]}" "$trusted_repository" \
  "+refs/heads/*:$trusted_ref_namespace/*"

if ! git -C "$megatron_lm_dir" cat-file -e "$megatron_lm_commit^{commit}" 2>/dev/null; then
  echo "Megatron-LM commit is not reachable from an NVIDIA/Megatron-LM branch." >&2
  exit 1
fi

containing_refs="$(
  git -C "$megatron_lm_dir" for-each-ref \
    --format='%(refname)' \
    --contains="$megatron_lm_commit" \
    "$trusted_ref_namespace/"
)"
if [[ -z "$containing_refs" ]]; then
  echo "Megatron-LM commit is not reachable from an NVIDIA/Megatron-LM branch." >&2
  exit 1
fi

git -C "$megatron_lm_dir" checkout --detach "$megatron_lm_commit"
test "$(git -C "$megatron_lm_dir" rev-parse HEAD)" = "$megatron_lm_commit"
