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

import pytest
from logra.config import LoGRAConfig
from pydantic import ValidationError


def test_unsupported_optimizer_and_bad_rank_fail():
    with pytest.raises(ValidationError):
        LoGRAConfig(optimizer="adam")
    with pytest.raises(ValidationError):
        LoGRAConfig(rank=0)


def test_predicted_kl_is_not_a_project_option():
    with pytest.raises(ValidationError):
        LoGRAConfig(kl_budget=0.01)
