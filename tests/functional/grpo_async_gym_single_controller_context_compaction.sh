#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Real-model CC coverage through the shared SingleController + Gym launcher.
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export SC_TEST_CONTEXT_COMPACTION=1
export SC_TEST_EXP_NAME=grpo_async_gym_single_controller_context_compaction
exec bash "$SCRIPT_DIR/grpo_async_gym_single_controller.sh" "$@"
