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

from tests.test_suites.utils.base_test import BaseNeMoRLTest
from tests.test_suites.utils.types.base_config import NeMoRLTestConfig


class TestSftLlama3170b8n8gTp4pp2LongMegatron(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="sft-llama3.1-70b-8n8g-tp4pp2-long-megatron",
        algorithm="sft",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=240,
        overrides={
            "sft.max_num_steps": 300,
        },
    )
