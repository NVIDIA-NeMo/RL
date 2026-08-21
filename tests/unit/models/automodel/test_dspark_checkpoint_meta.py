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

from nemo_rl.models.automodel.draft.integration import (
    validate_dspark_checkpoint_meta,
)

_TRAINING_META = {
    "model_name": "deepseek-ai/dspark_qwen3_8b_block7",
    "block_size": 7,
    "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33],
    "optimizer_layout": [
        {"name": "policy", "num_params": 4},
        {"name": "draft", "num_params": 2},
    ],
}


def test_weights_only_load_skips_optimizer_layout():
    """init_optimizer=False loads (eval/logprob workers) carry no expected
    layout and must accept a checkpoint saved from training."""
    expected = dict(_TRAINING_META, optimizer_layout=None)
    validate_dspark_checkpoint_meta(_TRAINING_META, expected)


def test_optimizer_resume_rejects_layout_mismatch():
    expected = dict(
        _TRAINING_META,
        optimizer_layout=[
            {"name": "draft", "num_params": 2},
            {"name": "policy", "num_params": 4},
        ],
    )
    with pytest.raises(ValueError, match="optimizer_layout"):
        validate_dspark_checkpoint_meta(_TRAINING_META, expected)


def test_optimizer_resume_rejects_stale_unnamed_layout():
    saved = dict(
        _TRAINING_META, optimizer_layout=[{"num_params": 4}, {"num_params": 2}]
    )
    with pytest.raises(ValueError, match="optimizer_layout"):
        validate_dspark_checkpoint_meta(saved, _TRAINING_META)


def test_architecture_mismatch_always_rejected():
    expected = dict(_TRAINING_META, block_size=4, optimizer_layout=None)
    with pytest.raises(ValueError, match="block_size"):
        validate_dspark_checkpoint_meta(_TRAINING_META, expected)


def test_draft_checkpoint_dir_is_sibling_of_weights_tree():
    """The draft DCP entry must not live under the policy weights directory:
    detect_checkpoint_format walks weights_path recursively and would
    mis-detect the safetensors policy as DCP after seeing .distcp files."""
    from nemo_rl.models.automodel.draft.integration import draft_checkpoint_dir

    weights = "/ckpt/step_3/policy/weights"
    draft_dir = draft_checkpoint_dir(weights)
    assert draft_dir == "/ckpt/step_3/policy/draft"
    assert not (draft_dir + "/").startswith(weights + "/")
    # Trailing slashes must not change the derivation.
    assert draft_checkpoint_dir(weights + "/") == draft_dir


def test_legacy_meta_without_algo_resumes_as_dspark():
    """Pre-versioning dspark checkpoints (no meta_version/algo) must remain
    loadable by a dspark run."""
    expected = dict(
        _TRAINING_META, meta_version=1, algo="dspark", optimizer_layout=None
    )
    validate_dspark_checkpoint_meta(_TRAINING_META, expected)


def test_algo_mismatch_rejected():
    saved = dict(_TRAINING_META, meta_version=1, algo="dflash")
    expected = dict(_TRAINING_META, meta_version=1, algo="dspark")
    with pytest.raises(ValueError, match="algo mismatch"):
        validate_dspark_checkpoint_meta(saved, expected)


def test_legacy_dspark_meta_rejected_by_eagle3_run():
    """A legacy (algo-less) checkpoint is dspark; an eagle3 run must refuse it."""
    expected = {
        "meta_version": 1,
        "algo": "eagle3",
        "aux_layer_ids": [1, 17, 33],
        "draft_vocab_size": 32000,
        "optimizer_layout": None,
    }
    with pytest.raises(ValueError, match="algo mismatch"):
        validate_dspark_checkpoint_meta(_TRAINING_META, expected)


def test_eagle3_meta_compares_per_algo_keys():
    saved = {
        "meta_version": 1,
        "algo": "eagle3",
        "aux_layer_ids": [1, 17, 33],
        "draft_vocab_size": 32000,
        "ttt_steps": 3,
        "optimizer_layout": None,
    }
    validate_dspark_checkpoint_meta(saved, dict(saved))
    with pytest.raises(ValueError, match="aux_layer_ids"):
        validate_dspark_checkpoint_meta(saved, dict(saved, aux_layer_ids=[2, 18, 34]))
    with pytest.raises(ValueError, match="draft_vocab_size"):
        validate_dspark_checkpoint_meta(saved, dict(saved, draft_vocab_size=64000))


def test_eagle3_meta_rejects_ttt_steps_mismatch():
    """Resume must not silently change the TTT unroll depth the draft was
    trained for."""
    saved = {
        "meta_version": 1,
        "algo": "eagle3",
        "aux_layer_ids": [1, 17, 33],
        "draft_vocab_size": 32000,
        "ttt_steps": 3,
        "optimizer_layout": None,
    }
    with pytest.raises(ValueError, match="ttt_steps"):
        validate_dspark_checkpoint_meta(saved, dict(saved, ttt_steps=5))


def test_eagle3_meta_record_requires_and_records_ttt_steps():
    from types import SimpleNamespace

    from nemo_rl.models.automodel.draft.integration import draft_meta_record

    draft_model = SimpleNamespace(
        config=SimpleNamespace(target_layer_ids=[1, 17, 33], draft_vocab_size=32000)
    )
    record = draft_meta_record(
        draft_model, "tiny-eagle3", None, algo="eagle3", ttt_steps=3
    )
    assert record["ttt_steps"] == 3
    assert record["aux_layer_ids"] == [1, 17, 33]

    with pytest.raises(ValueError, match="ttt_steps"):
        draft_meta_record(draft_model, "tiny-eagle3", None, algo="eagle3")


def test_dspark_draft_vocab_size_compared_only_when_recorded():
    """Legacy dspark metadata lacks draft_vocab_size; versioned metadata
    records it and mismatches must reject."""
    expected = dict(_TRAINING_META, draft_vocab_size=32000, optimizer_layout=None)
    validate_dspark_checkpoint_meta(_TRAINING_META, expected)

    saved = dict(_TRAINING_META, meta_version=1, algo="dspark", draft_vocab_size=8192)
    with pytest.raises(ValueError, match="draft_vocab_size"):
        validate_dspark_checkpoint_meta(saved, expected)
