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
"""Opt-in FP8 per-token-head KV attention for vLLM rollouts on SM100.

``backend``: page-local cache layout and the TRITON_ATTN override (Triton prefill).
``decode``: Gluon/tcgen05 split-KV decode kernel.
``worker``: vLLM ``worker_cls`` that installs the backend; see its docstring for config.

Submodules import vLLM/Triton and are not imported here.
"""
