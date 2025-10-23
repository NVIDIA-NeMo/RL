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


class TestGrpoMathLlamaNemotronSuper49bV54n8gFsdp2tp8(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="grpo-math-llama-nemotron-super-49b-v.5-4n8g-fsdp2tp8",
        algorithm="grpo",
        model_class="llm",
        test_suites=["nightly"],
        time_limit_minutes=30,
        overrides={
            "grpo.max_num_steps": 2,
        },
    )
