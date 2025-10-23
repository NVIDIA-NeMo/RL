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
from tests.test_suites.utils.types.job_dependencies import JobDependencies


class TestDistillationQwen332bTo4bBase2n8gFsdp2tp2LongV1(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="distillation-qwen3-32b-to-4b-base-2n8g-fsdp2tp2-long.v1",
        algorithm="distillation",
        model_class="llm",
        test_suites=["release", "long"],
        time_limit_minutes=240,
        overrides={
            "distillation.max_num_steps": 100,
        },
    )

    # TODO(ahmadki): dummy job dependencies that doesn't make sense, but here as an example
    job_dependencies = JobDependencies(
        stages={
            "training": {"depends_on": ["validation"], "needs": []},
            "validation": {"depends_on": [], "needs": []},
        },
        job_groups={},
    )

    # TODO(ahmadki): dummy tests
    @pytest.mark.stage("validation")
    def one(self):
        assert 1 + 1 == 2

    @pytest.mark.stage("validation")
    def two(self):
        assert 2 + 2 == 4

    @pytest.mark.stage("validation")
    def three(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    def four(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    def five(self):
        assert 4 + 4 == 8

    @pytest.mark.stage("validation")
    def six(self):
        assert 4 + 4 == 8
