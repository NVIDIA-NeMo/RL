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
protected_repository="https://github.com/NVIDIA/Megatron-LM.git"
protected_ref_namespace="refs/remotes/nvidia-protected"

if [[ ! "$megatron_lm_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Megatron-LM commit must be a full 40-character hexadecimal SHA." >&2
  exit 1
fi

fetch_args=(--no-tags --force --filter=blob:none)
if [[ "$(git -C "$megatron_lm_dir" rev-parse --is-shallow-repository)" == "true" ]]; then
  fetch_args+=(--unshallow)
fi

# Fetch the protected branches directly from NVIDIA/Megatron-LM. Do not fetch
# the requested SHA: its object must arrive through one of these trusted refs.
git -C "$megatron_lm_dir" fetch "${fetch_args[@]}" "$protected_repository" \
  "+refs/heads/main:$protected_ref_namespace/main" \
  "+refs/heads/dev:$protected_ref_namespace/dev"

if ! git -C "$megatron_lm_dir" merge-base --is-ancestor "$megatron_lm_commit" "$protected_ref_namespace/main" && \
  ! git -C "$megatron_lm_dir" merge-base --is-ancestor "$megatron_lm_commit" "$protected_ref_namespace/dev"; then
  echo "Megatron-LM commit is not reachable from protected main or dev." >&2
  exit 1
fi

git -C "$megatron_lm_dir" checkout --detach "$megatron_lm_commit"
test "$(git -C "$megatron_lm_dir" rev-parse HEAD)" = "$megatron_lm_commit"
