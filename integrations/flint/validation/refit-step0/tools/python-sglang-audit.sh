#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export FLINT_STEP0_ENABLE=1
exec /opt/flint/bin/python-sglang "$@"
