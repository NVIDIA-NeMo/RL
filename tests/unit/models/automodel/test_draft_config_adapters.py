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
"""Speculators-format draft config adapters.

Fixtures are trimmed copies of the real published checkpoint configs:
- RedHatAI/Qwen3-8B-speculator.dflash @ 0ac7575
- RedHatAI/Qwen3-8B-speculator.eagle3 @ 08610ff
"""

import pytest

# Skip entire module if nemo_automodel is not available
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.models.automodel.draft.integration import (
    _adapt_speculators_dspark_config,
    _adapt_speculators_eagle3_config,
    default_eagle3_aux_layer_ids_vllm,
    validate_dspark_draft_config,
)

pytestmark = pytest.mark.automodel

_QWEN3_8B_LAYER_FIELDS = {
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "max_position_embeddings": 40960,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "vocab_size": 151936,
}

_REDHAT_DFLASH_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "aux_hidden_state_layer_ids": [2, 10, 18, 26, 34],
    "block_size": 8,
    "draft_vocab_size": 32000,
    "mask_token_id": 151669,
    "speculators_model_type": "dflash",
    "transformer_layer_config": {
        "model_type": "qwen3",
        "num_hidden_layers": 5,
        **_QWEN3_8B_LAYER_FIELDS,
    },
}

_REDHAT_EAGLE3_CONFIG = {
    "architectures": ["Eagle3Speculator"],
    "draft_vocab_size": 32000,
    "norm_before_residual": True,
    "speculators_model_type": "eagle3",
    "transformer_layer_config": {
        # speculators publishes qwen3-target eagle3 layer configs as llama.
        "model_type": "llama",
        "num_hidden_layers": 1,
        **_QWEN3_8B_LAYER_FIELDS,
    },
}

_DSPARK_OPTIONS = {
    "ce_loss_alpha": 1.0,
    "l1_loss_alpha": 0.0,
    "confidence_loss_alpha": 0.0,
    "num_anchors": 1024,
    "train_embed_and_head": True,
}


def test_dflash_keeps_slot_count_and_marks_bonus_anchor_layout():
    """block_size counts slots in both conventions and passes through; dflash
    is distinguished by the unsupervised bonus-anchor layout
    (sample_from_anchor=False), matching vLLM's 1 + N dflash query slots."""
    adapted = _adapt_speculators_dspark_config(_REDHAT_DFLASH_CONFIG, algo="dflash")
    assert adapted.block_size == 8
    assert adapted.sample_from_anchor is False


def test_dspark_defaults_to_next_token_layout():
    config = dict(_REDHAT_DFLASH_CONFIG, sample_from_anchor=True)
    adapted = _adapt_speculators_dspark_config(config, algo="dspark")
    assert adapted.block_size == 8
    assert adapted.sample_from_anchor is True


def test_dflash_aux_ids_shift_to_output_of_layer_convention():
    """vLLM aux id j = output of decoder layer j-1 (0 = embedding); the
    vendored capture indexes by output-of-layer, so ids shift by -1."""
    adapted = _adapt_speculators_dspark_config(_REDHAT_DFLASH_CONFIG, algo="dflash")
    assert adapted.target_layer_ids == [1, 9, 17, 25, 33]


def test_dflash_adapter_carries_drafter_fields():
    adapted = _adapt_speculators_dspark_config(_REDHAT_DFLASH_CONFIG, algo="dflash")
    assert adapted.mask_token_id == 151669
    assert adapted.draft_vocab_size == 32000
    assert adapted.markov_rank == 0
    assert adapted.enable_confidence_head is False
    assert adapted.architectures == ["DFlashDraftModel"]
    assert adapted.num_hidden_layers == 5


def test_dflash_validation_accepts_redhat_config():
    adapted = _adapt_speculators_dspark_config(_REDHAT_DFLASH_CONFIG, algo="dflash")
    validate_dspark_draft_config(adapted, _DSPARK_OPTIONS, algo="dflash")


def test_dflash_validation_rejects_markov_head():
    config = dict(_REDHAT_DFLASH_CONFIG, markov_rank=2, markov_head_type="mlp")
    adapted = _adapt_speculators_dspark_config(config, algo="dflash")
    with pytest.raises(ValueError, match="Markov"):
        validate_dspark_draft_config(adapted, _DSPARK_OPTIONS, algo="dflash")


def test_dspark_algo_rejects_bonus_anchor_layout():
    config = dict(
        _REDHAT_DFLASH_CONFIG,
        architectures=["Qwen3DSparkModel"],
        sample_from_anchor=False,
    )
    adapted = _adapt_speculators_dspark_config(config, algo="dspark")
    with pytest.raises(ValueError, match="sample_from_anchor"):
        validate_dspark_draft_config(adapted, _DSPARK_OPTIONS, algo="dspark")


def test_dflash_algo_rejects_next_token_layout():
    config = dict(_REDHAT_DFLASH_CONFIG, sample_from_anchor=True)
    adapted = _adapt_speculators_dspark_config(config, algo="dspark")
    with pytest.raises(ValueError, match="sample_from_anchor"):
        validate_dspark_draft_config(adapted, _DSPARK_OPTIONS, algo="dflash")


def test_dflash_validation_rejects_confidence_loss():
    adapted = _adapt_speculators_dspark_config(_REDHAT_DFLASH_CONFIG, algo="dflash")
    options = dict(_DSPARK_OPTIONS, confidence_loss_alpha=0.5)
    with pytest.raises(ValueError, match="confidence_loss_alpha"):
        validate_dspark_draft_config(adapted, options, algo="dflash")


def test_eagle3_adapter_accepts_llama_layer_config():
    adapted = _adapt_speculators_eagle3_config(
        _REDHAT_EAGLE3_CONFIG, "RedHatAI/Qwen3-8B-speculator.eagle3", 36
    )
    assert adapted.draft_vocab_size == 32000
    assert adapted.norm_before_residual is True
    assert adapted.num_hidden_layers == 1
    assert adapted.architectures == ["Eagle3Speculator"]


def test_eagle3_unpinned_aux_ids_use_vllm_default_shifted():
    """Qwen3-8B has 36 layers; vLLM's default aux selection is
    (2, N // 2, N - 3) in vLLM indexing, shifted by -1 for the trainer."""
    assert default_eagle3_aux_layer_ids_vllm(36) == [2, 18, 33]
    adapted = _adapt_speculators_eagle3_config(
        _REDHAT_EAGLE3_CONFIG, "RedHatAI/Qwen3-8B-speculator.eagle3", 36
    )
    assert adapted.target_layer_ids == [1, 17, 32]


def test_eagle3_pinned_aux_ids_take_precedence():
    config = dict(_REDHAT_EAGLE3_CONFIG, eagle_aux_hidden_state_layer_ids=[3, 20, 30])
    adapted = _adapt_speculators_eagle3_config(config, "pinned", 36)
    assert adapted.target_layer_ids == [2, 19, 29]


def test_eagle3_unpinned_aux_ids_require_target_layer_count():
    with pytest.raises(ValueError, match="aux layer"):
        _adapt_speculators_eagle3_config(_REDHAT_EAGLE3_CONFIG, "no-target", None)


def test_eagle3_missing_draft_vocab_size_rejected():
    config = dict(_REDHAT_EAGLE3_CONFIG)
    del config["draft_vocab_size"]
    with pytest.raises(ValueError, match="draft_vocab_size"):
        _adapt_speculators_eagle3_config(config, "no-vocab", 36)
