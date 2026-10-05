#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /opt/nemo-rl/docker/nano35/environment.sh
if (($# == 0)); then
    set -- bash
fi
exec "$@"
