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

from tests.test_suites.base_config import NeMoRLTestConfig
from tests.test_suites.base_test import BaseNeMoRLTest


class TestDpoLlama318bTulu31n8gFsdp2tp1(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="dpo-llama3.1-8b-tulu3-1n8g-fsdp2tp1",
        algorithm="dpo",
        model_class="llm",
        test_suites=["release"],
        time_limit_minutes=45,
        overrides={
            "dpo.max_num_steps": 150,
        },
    )
