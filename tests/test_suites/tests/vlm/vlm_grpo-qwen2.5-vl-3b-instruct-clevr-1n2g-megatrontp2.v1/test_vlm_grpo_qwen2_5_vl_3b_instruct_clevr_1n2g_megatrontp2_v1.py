# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from tests.test_suites.base_config import BaseNeMoRLTest, NeMoRLTestConfig


class TestVlmGrpoQwen25Vl3bInstructClevr1n2gMegatrontp2V1(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp2.v1",
        algorithm="grpo",
        model_class="vlm",
        test_suites=["nightly"],
        time_limit_minutes=180,
        overrides={
            "grpo.max_num_steps": 200,
        },
    )
