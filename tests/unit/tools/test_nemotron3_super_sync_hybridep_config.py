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


def test_super_sync_hybridep_only_changes_dispatcher() -> None:
    project_root = Path(__file__).resolve().parents[3]
    config_dir = (
        project_root / "examples" / "configs" / "recipes" / "llm" / "performance"
    )
    hybridep_path = config_dir / "grpo-nemotron3-super-120BA12B-32n4g-hybridep.yaml"

    assert hybridep_path.read_text() == (
        "defaults: grpo-nemotron3-super-120BA12B-32n4g.yaml\n"
        "\n"
        "policy:\n"
        "  megatron_cfg:\n"
        "    moe_token_dispatcher_type: flex\n"
        "    moe_flex_dispatcher_backend: hybridep\n"
        "    moe_hybridep_num_sms: 32\n"
    )
