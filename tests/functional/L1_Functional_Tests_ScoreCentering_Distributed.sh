#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CI runners provide two GPUs: exercise TP=2 and CP=2 on NCCL, plus combined
# TP=2/CP=2 on four CPU/Gloo ranks with the same value/gradient assertions.
# The combined NCCL cases also run when four GPUs are visible.
set -xeuo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
cd "${PROJECT_ROOT}/tests"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
uv run --no-sync python -c 'import torch; assert torch.cuda.device_count() >= 2, "Score-centering TP/CP checks require two visible GPUs"; assert torch.distributed.is_gloo_available(), "Combined score-centering TP/CP checks require Gloo"'
uv run --no-sync python -m pytest unit/distributed/test_model_utils.py \
    -k 'test_score_centering_packed_tp_cp or test_get_policy_logprobs_at_sampler_topk_tp_cp' \
    --hf-gated --cov=nemo_rl --cov-report=term-missing
