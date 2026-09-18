#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stage immutable build inputs, then compile once in a small writable runtime mount.
set -euo pipefail
mode=${1:?Expected stage or build}
case "$mode" in
  stage)
    source_dir=${2:?Expected immutable dataset source directory}
    target_dir=${3:?Expected new run-owned build directory}
    test -f "$source_dir/helpers.cpp"
    test -f "$source_dir/Makefile"
    # Never reuse an existing tree from another source or interrupted launch.
    mkdir "$target_dir"
    cp -p "$source_dir/"*.py "$source_dir/Makefile" "$source_dir/helpers.cpp" "$target_dir/"
    ;;
  build)
    target_dir=${2:?Expected mounted dataset directory}
    : "${IMAGE_TOOLS_BUILD_PYTHON:?Expected bundled learner interpreter}"
    test -f "$target_dir/helpers.cpp"
    test -x "$IMAGE_TOOLS_BUILD_PYTHON"
    # All nodes share this run-local directory. Serialize compilation, not training.
    exec 9>"$target_dir/.build.lock"
    flock -w 180 9
    suffix=$("$IMAGE_TOOLS_BUILD_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
    includes=$("$IMAGE_TOOLS_BUILD_PYTHON" -m pybind11 --includes)
    make -C "$target_dir" "LIBEXT=$suffix" "CPPFLAGS=$includes"
    # Exercise the exact upstream make path used by Bridge, including python3-config.
    "$IMAGE_TOOLS_BUILD_PYTHON" -c 'from megatron.core.datasets.utils import compile_helpers; compile_helpers(); from megatron.core.datasets import helpers_cpp; assert callable(helpers_cpp.build_sample_idx_int32); print("IMAGE_TOOLS_DATASET_HELPERS_OK", helpers_cpp.__file__, flush=True)'
    ;;
  *) echo 'Expected stage or build' >&2; exit 2 ;;
esac
