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

from tests.test_suites.utils.base_test import BaseNeMoRLTest
from tests.test_suites.utils.types.base_config import NeMoRLTestConfig


class TestDistillationQwen332bTo17bBase1n8gFsdp2tp1V1(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="distillation-qwen3-32b-to-1.7b-base-1n8g-fsdp2tp1.v1",
        algorithm="distillation",
        model_class="llm",
        test_suites=["nightly", "akiswani"],  # TODO(ahmadki): test_suites
        time_limit_minutes=60,
        steps_per_run=2,
        overrides={
            "distillation.max_num_steps": 10,
            "distillation.val_period": 20,
        },
    )

    # TODO(ahmadki): dummy tests
    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    @pytest.mark.order(1)
    def test_one(self):
        assert 1 + 1 == 2

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_1")
    def test_two(self):
        assert 2 + 2 == 4

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_2")
    def test_three(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    @pytest.mark.job_group("job_2")
    def test_four(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    def test_five(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    def test_six(self):
        assert 4 + 4 == 8
