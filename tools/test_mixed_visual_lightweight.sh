#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${PYTHON:-python3}
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$root:$root/3rdparty/Gym-workspace/Gym${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
"$python_bin" -m pytest --noconftest -q -o addopts='' \
  tests/unit/environments/test_nemo_gym_tool_calls.py \
  tests/unit/tools/test_visual_image_tools_mix.py \
  tests/unit/tools/test_visual_image_tools_metrics.py \
  tests/unit/tools/test_prepare_image_tools_grpo.py \
  tests/unit/tools/test_image_tools_processor_assets.py \
  tests/unit/tools/test_image_tools_recipe.py \
  tests/unit/tools/test_image_tools_launcher.py \
  tests/unit/tools/test_image_tools_dataset_helpers.py \
  tests/unit/tools/test_vllm_engine_input_threading.py
cd "$root/3rdparty/Gym-workspace/Gym"
"$python_bin" -m pytest --noconftest -q -o addopts='' \
  tests/unit_tests/test_compact_rollout_diagnostics.py
