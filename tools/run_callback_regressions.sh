#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /opt/nemo-rl
learner_python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
# Wheels are universal, hash-checked against uv.lock and imported directly.
# This environment applies only to this subprocess, never to training actors.
test_wheels=$("$learner_python" tools/callback_test_runtime.py)
export PYTHONPATH="$test_wheels${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$learner_python" -m pytest --version
timeout 240 "$learner_python" -m pytest -q --noconftest --disable-warnings -p no:cacheprovider -o addopts='' \
  tests/unit/models/megatron/test_megatron_setup.py::TestFinalizeMegatronSetup
(
  cd /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
  timeout 240 "$learner_python" -m pytest -q --noconftest --disable-warnings -p no:cacheprovider -o addopts='' \
    tests/unit_tests/training/test_runtime_callbacks.py
)
echo 'MIXED_CALLBACK_REGRESSIONS_PASS rl_pr=4116 bridge_pr=6067'
