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
"""No-op Ray fixture overrides for the TRT-LLM direct-server tests.

Tests here use tensorrt_llm.LLM directly, not via Ray actors, so they must not
auto-connect to a running cluster via the session-scoped autouse fixtures in
tests/unit/conftest.py.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor(init_ray_cluster):
    yield
