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

from pathlib import Path

import yaml


def test_template_grpo_refit_offload_defaults() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "grpo_math_1B.yaml"
    policy = yaml.safe_load(config_path.read_text())["policy"]

    assert policy["offload_policy_before_refit"] is False
    assert policy["offload_optimizer_for_refit"] is True
