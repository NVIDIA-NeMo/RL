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
"""The lilicorr draft config must survive a real recipe merge.

Recipes deep-merge policy.draft from grpo_math_1B.yaml, so a lilicorr block arrives
carrying Eagle3's knobs (token_chunk_size, ttt_steps, train_embed_and_head...). The
config forbids extra keys, so without the inherited-defaults validator the run dies at
startup config validation rather than at anything informative.
"""

import json

import pytest

from nemo_rl.models.policy.draft_config import (
    _DRAFT_CONFIG_ADAPTER,
    BLOCK_DRAFT_ALGOS_NEEDING_SAMPLE_IDS,
    MEGATRON_BLOCK_DRAFT_ALGOS,
    LiLiCorrDraftConfig,
)

# Exactly what the Super 3.5 QA-MOPD recipe resolves policy.draft to.
MERGED_RECIPE_DRAFT = {
    "enabled": True,
    "speculator_type": "lilicorr",
    "loss_weight": 0.1,
    "anchors_per_sample": 32,
    "learning_rate": 1.0e-4,
    # inherited from grpo_math_1B.yaml's eagle3 defaults, undeletable by a nested override
    "num_layers": None,
    "aux_layer_indices": None,
    "token_chunk_size": 4096,
    "optimizer": None,
    "ttt_steps": 3,
    "ttt_step_loss_decay": 1.0,
    "train_embed_and_head": True,
}


@pytest.fixture
def drafter_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "num_target_layers": 88,
                "block_size": 4,
                "dflash_config": {
                    "target_layer_ids": [1, 22, 43, 64, 85],
                    "mask_token_id": 990,
                    "projector_type": "lilicorr",
                },
            }
        )
    )
    return str(tmp_path)


def test_merged_recipe_block_validates(drafter_checkpoint):
    config = dict(MERGED_RECIPE_DRAFT, model_name=drafter_checkpoint)
    resolved = _DRAFT_CONFIG_ADAPTER.validate_python(config)
    assert isinstance(resolved, LiLiCorrDraftConfig)
    assert resolved.anchors_per_sample == 32


def test_geometry_comes_from_the_checkpoint(drafter_checkpoint):
    config = dict(MERGED_RECIPE_DRAFT, model_name=drafter_checkpoint)
    resolved = _DRAFT_CONFIG_ADAPTER.validate_python(config)
    # gamma counts mask slots: a block is one anchor plus gamma masks, and the
    # reranker scores block_size - 1 of them.
    assert resolved.gamma == 3
    assert resolved.mask_token_id == 990
    assert resolved.target_hidden_state_layer_ids == [1, 22, 43, 64, 85]


def test_lilicorr_needs_sample_ids_without_joining_the_discriminator_set():
    """Both halves matter.

    lilicorr must get draft_sample_ids, or prepare_batch has no anchor schedule. It
    must NOT join MEGATRON_BLOCK_DRAFT_ALGOS, which selects a "megatron_<algo>" config
    variant that does not exist for lilicorr.
    """
    assert "lilicorr" in BLOCK_DRAFT_ALGOS_NEEDING_SAMPLE_IDS
    assert "lilicorr" not in MEGATRON_BLOCK_DRAFT_ALGOS


def test_a_genuine_typo_is_still_rejected(drafter_checkpoint):
    config = dict(MERGED_RECIPE_DRAFT, model_name=drafter_checkpoint, loss_wieght=0.2)
    with pytest.raises(Exception):
        _DRAFT_CONFIG_ADAPTER.validate_python(config)
