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

import pytest

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

RECIPE_ROOT = Path(__file__).parents[2] / "examples/configs/recipes"


def test_gemma4_cp_keeps_local_hybridep_inputs_aligned() -> None:
    register_omegaconf_resolvers()
    config = load_config(
        RECIPE_ROOT / "llm/dapo-gemma4-26ba4b-it-4n8g-fsdp2ep16cp2-automodel.yaml"
    )
    global_alignment = 64 * config.policy.dtensor_cfg.context_parallel_size
    assert config.policy.make_sequence_length_divisible_by % global_alignment == 0
    assert config.policy.dynamic_batching.sequence_length_round % global_alignment == 0


@pytest.mark.parametrize(
    ("recipe_name", "dispatcher", "experts"),
    [
        ("llm/dpo-nanov3-30B3AB-1n4g-fsdp4ep4-automodel.yaml", "torch", "torch_mm"),
        ("llm/grpo-glm47-flash-4n8g-automodel.yaml", "hybridep", "torch_mm"),
        ("llm/grpo-minimax-m27-dapo-8n8g-automodel.yaml", "hybridep", "torch_mm"),
        ("llm/grpo-moonlight-16b-automodel-1n8g-ep8.yaml", "hybridep", "torch_mm"),
        (
            "llm/grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.v2.yaml",
            "hybridep",
            "torch_mm",
        ),
        ("llm/dapo-gemma4-26ba4b-it-4n8g-fsdp2-automodel.yaml", "hybridep", "gmm"),
        ("llm/dapo-nanov3.5-30BA3B-4n8g-automodel.yaml", "hybridep", "gmm"),
        ("llm/grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml", "hybridep", "gmm"),
        ("llm/grpo-qwen3.5-35ba3b-dapo-4n8g-automodel.yaml", "hybridep", "gmm"),
        (
            "vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2.yaml",
            "hybridep",
            "torch_mm",
        ),
        (
            "vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml",
            "hybridep",
            "torch_mm",
        ),
        (
            "vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml",
            "hybridep",
            "gmm",
        ),
        ("vlm/vlm_grpo-gemma4-e4b-geo3k-1n8g-automodel.yaml", "hybridep", "gmm"),
    ],
)
def test_automodel_recipes_preserve_backend_choices(
    recipe_name: str, dispatcher: str, experts: str
) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_ROOT / recipe_name)
    backend = config.policy.dtensor_cfg.automodel_kwargs.backend
    assert backend.dispatcher == dispatcher
    assert backend.experts == experts
