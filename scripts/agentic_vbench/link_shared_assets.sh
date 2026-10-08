#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reuse already prepared gym-native Agentic-VBench inputs instead of fetching them:
# the pinned benchmark checkout (PhiloLabs/agentic-vbench@410a75fe), a Harbor 0.6.6
# venv whose interpreter lives on the shared filesystem, and the static Docker client.
#   bash scripts/agentic_vbench/link_shared_assets.sh \
#     --benchmark /lustre/.../agenticvbench-eval-factory/agentic-vbench \
#     --harbor-venv /lustre/.../pivot-rl-video/data/agentic_vbench/harbor-venv \
#     --docker-client /lustre/.../agenticvbench-eval-factory/docker-client
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
target=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
benchmark=""; harbor=""; docker_client=""
while (( $# )); do
  case $1 in
    --benchmark) benchmark=$2; shift 2 ;;
    --harbor-venv) harbor=$2; shift 2 ;;
    --docker-client) docker_client=$2; shift 2 ;;
    --target) target=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$target"
link() { # name source
  [[ -e $2 ]] || { echo "Missing $2" >&2; exit 1; }
  if [[ -e $target/$1 && ! -L $target/$1 ]]; then echo "Keeping existing $target/$1"; return; fi
  ln -sfn "$(cd "$2" && pwd)" "$target/$1"
}
[[ -z $benchmark ]] || link agentic-vbench "$benchmark"
[[ -z $harbor ]] || link harbor-venv "$harbor"
[[ -z $docker_client ]] || link docker-client "$docker_client"
if [[ -e $target/agentic-vbench ]]; then
  revision=$(git -C "$target/agentic-vbench" rev-parse HEAD)
  expected=$(python3 -c "import sys; sys.path.insert(0, '$code_dir/3rdparty/Gym-workspace/Gym'); from responses_api_agents.agentic_vbench_agent.core import REVISION; print(REVISION)")
  [[ $revision == "$expected" ]] || { echo "Benchmark checkout is at $revision, expected $expected" >&2; exit 1; }
fi
if [[ -e $target/harbor-venv ]]; then
  "$target/harbor-venv/bin/python" -c 'import importlib.metadata as m, dotenv; assert m.version("harbor") == "0.6.6"; print("harbor 0.6.6 ok")'
  case $(readlink -f "$target/harbor-venv/bin/python") in
    /lustre/*|/scratch/*) ;;
    *) echo "WARNING: the Harbor venv's interpreter is not on the shared filesystem; it may be missing inside the training container" >&2 ;;
  esac
fi
if [[ -e $target/docker-client ]]; then
  [[ -x $target/docker-client/docker && -x $target/docker-client/cli-plugins/docker-compose ]] || { echo "docker-client must hold docker and cli-plugins/docker-compose" >&2; exit 1; }
fi
mkdir -p "$target/episodes" "$target/gym_venvs" "$target/train"
echo "Agentic-VBench inputs linked into $target"
