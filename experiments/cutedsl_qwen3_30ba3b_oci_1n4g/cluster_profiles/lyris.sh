#!/bin/bash
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

export CUTEDSL_PROFILE_NAME=lyris
export CUTEDSL_ACCOUNT=coreai_dlalgo_llm
export CUTEDSL_PARTITION=gb200
export CUTEDSL_GRES=
export CUTEDSL_SEGMENT=1
export CUTEDSL_COMMENT=metrics
export CUTEDSL_IMAGE=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260711_2346595.sqsh
export CUTEDSL_IMAGE_SHA256=bb5beff9ade16a1eeb6badde7601731bb003a95b4cccf85b3bd9b11c84803a2a
export CUTEDSL_FUNCTIONAL_TIME=02:00:00
export CUTEDSL_BENCHMARK_TIME=05:00:00
