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

from nemo_rl.algorithms.ppo_utils import ground_truth_privilege_prefix as gtp
from nemo_rl.algorithms.ppo_utils.privilege_prefix_utils import TRUNCATION_MARKER
from nemo_rl.models.value.config import PrivilegedValueConfig
from tests.unit.algorithms.test_privileged_value import _CharTokenizer


def _config(max_total_tokens):
    return PrivilegedValueConfig(
        enabled=True,
        max_total_tokens=max_total_tokens,
        prefix_builder_fqn=f"{gtp.__name__}.build_privilege_prefix",
    )


def test_builder_emits_the_ground_truth():
    block, stats = gtp.build_privilege_prefix(
        {"ground_truth": 42}, _CharTokenizer(), _config(64)
    )

    assert block == "<reference>\n<ground_truth>\n42\n</ground_truth>\n</reference>"
    assert stats == {"truncated_fields": [], "wanted_tokens": 2, "kept_tokens": 2}


def test_builder_cuts_the_ground_truth_to_the_budget():
    block, stats = gtp.build_privilege_prefix(
        {"ground_truth": "a" * 30}, _CharTokenizer(), _config(20)
    )

    assert f"<ground_truth>\naaaa{TRUNCATION_MARKER}\n</ground_truth>" in block
    assert stats["truncated_fields"] == ["ground_truth"]
    assert stats["kept_tokens"] == 20


@pytest.mark.parametrize("env_info", [{}, {"ground_truth": ""}])
def test_builder_fails_loudly_without_a_ground_truth(env_info):
    with pytest.raises(ValueError, match="ground_truth"):
        gtp.build_privilege_prefix(env_info, _CharTokenizer(), _config(64))
