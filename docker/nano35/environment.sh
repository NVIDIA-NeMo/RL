#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
export NRL_CONTAINER=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_PYTHON_INSTALL_DIR=/opt/uv/python
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=/opt/gym_venvs
export TRTLLM_WHEEL_CACHE_DIR=/opt/trtllm_wheels
export CUDA_HOME=/usr/local/cuda
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl
export TORCH_CUDA_ARCH_LIST='10.0'
export CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
export CUDNN_PATH="$CUDNN_HOME"
export OPAL_PREFIX=/usr/local/mpi
# Keep MPI libraries and plugins in the same HPC-X installation. Debian's
# libmpi has the same SONAME but an incompatible component search layout.
export LD_LIBRARY_PATH="/usr/local/mpi/lib:/opt/nemo_rl_venv/lib/python3.13/site-packages/z3/lib:$CUDNN_HOME/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/opt/nemo_rl_venv/bin:/usr/local/bin:$PATH"
