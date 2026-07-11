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

export CUTEDSL_PROFILE_NAME=aws_dfw
export CUTEDSL_ACCOUNT=nemotron_sw_post
export CUTEDSL_PARTITION=batch_long
export CUTEDSL_GRES=gpu:4
export CUTEDSL_SEGMENT=
export CUTEDSL_COMMENT=metrics
export CUTEDSL_IMAGE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sna/containers/nemo_rl_nightly_20260711_1873004.sqsh
export CUTEDSL_IMAGE_SHA256=a393e1b8f12e5edafa49a84c0b78b172aa163ad29be04fca6e42855a5f16304a
export CUTEDSL_FUNCTIONAL_TIME=02:00:00
export CUTEDSL_BENCHMARK_TIME=08:00:00
