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

import pytest

from tests.test_suites.base_config import NeMoRLTestConfig
from tests.test_suites.base_test import BaseNeMoRLTest


class TestDapoQwen257b(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="dapo-qwen2.5-7b",
        algorithm="dapo",
        model_class="llm",
        test_suites=["release", "akiswani"],  # TODO(ahmadki): test_suites
        time_limit_minutes=240,
        overrides={
            "grpo.max_num_steps": 20,
        },
    )

    # TODO(ahmadki): dummy tests
    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    @pytest.mark.order(1)
    def one(self):
        assert 1 + 1 == 2

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    @pytest.mark.order(2)
    def two(self):
        assert 2 + 2 == 4

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    @pytest.mark.order(3)
    def three(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    def four(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    def five(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    def six(self):
        assert 4 + 4 == 8
