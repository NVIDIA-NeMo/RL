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
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
book=$root/docs/runbooks/supervl3p5
source "$book/nemo-rl/scripts/hsg.env.sh"
# Set a new output directory and W&B ID, or reuse them for an unfinished run.
export SUPER_CONFIG=${SUPER_CONFIG:-$book/nemo-rl/configs/supervl3p5-v2-ready-first.yaml}
exec bash "$book/nemo-rl/scripts/run_v2.sh" "$@" checkpointing.save_period=10
