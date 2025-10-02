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

from unittest.mock import MagicMock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import nemo_rl.algorithms.distillation as distil_mod
from nemo_rl.algorithms.distillation import (
    _default_distillation_save_state,
    check_vocab_equality,
    distillation_train,
    validate,
)
from nemo_rl.algorithms.loss_functions import DistillationLossFn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture
def mock_components():
    # Create mock components
    student_policy = MagicMock()
    student_policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {"global_valid_toks": [10]},
    }
    # Add generate method since student_generation will be set to student_policy
    student_policy.generate.return_value = {
        "output_ids": torch.randint(0, 8, (2, 10)),
        "generation_lengths": torch.tensor([5, 7]),
        "unpadded_sequence_lengths": torch.tensor([8, 10]),
        "logprobs": torch.randn(2, 10, 8),
    }

    teacher_policy = MagicMock()
    teacher_policy.get_topk_logits.return_value = {
        "topk_logits": torch.randn(2, 10, 64),
        "topk_indices": torch.randint(0, 8, (2, 10, 64)),
    }

    # Set student_generation to None to avoid Ray-related refit issues
    # This makes NEED_REFIT = False, so refit_policy_generation won't be called
    student_generation = None

    # Create a proper message log structure with token_ids (similar to SFT)
    # Use BatchedDataDict instead of regular dict to support repeat_interleave
    mock_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {
                        "token_ids": torch.tensor([1, 2, 3]),
                        "role": "user",
                        "content": "What is 1+1?",
                    },
                    {
                        "token_ids": torch.tensor([4, 5, 6]),
                        "role": "assistant",
                        "content": "The answer is 2.",
                    },
                ]
            ],
            "loss_multiplier": torch.tensor(
                [1.0]
            ),  # Make it 1D tensor for batch dimension
            "task_name": ["math"],
            "extra_env_info": [{}],
            "length": torch.tensor([6]),  # Make it 1D tensor for batch dimension
            "idx": torch.tensor([0]),  # Make it 1D tensor for batch dimension
        }
    )

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = DistillationLossFn(
        {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        }
    )

    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock environments
    task_to_env = {"math": MagicMock()}
    val_task_to_env = {"math": MagicMock()}

    # Create mock master config
    master_config = {
        "distillation": {
            "max_num_steps": 5,
            "val_period": 100,
            "val_batch_size": 1,
            "val_at_start": False,
            "max_val_samples": 10,
            "topk_logits_k": 64,
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 1,
            "max_rollout_turns": 0,  # No environment interaction needed for distillation
            "seed": 42,
        },
        "policy": {
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
            "max_total_sequence_length": 2048,
            "generation": {
                "colocated": {
                    "enabled": False,
                },
            },
        },
        "teacher": {
            "model_name": "test-teacher",
        },
        "loss_fn": {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        },
        "data": {
            "dataset_name": "test_dataset",
        },
        "logger": {
            "num_val_samples_to_print": 5,
        },
        "cluster": {
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
            "metric_name": None,
        },
    }

    return {
        "student_policy": student_policy,
        "teacher_policy": teacher_policy,
        "student_generation": student_generation,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "task_to_env": task_to_env,
        "val_task_to_env": val_task_to_env,
        "master_config": master_config,
    }


def test_distillation_train_max_steps(mock_components):
    """Test that training terminates correctly when maximum steps are reached."""
    mock_components["master_config"]["distillation"]["max_num_steps"] = 5

    distillation_save_state = _default_distillation_save_state()

    # Run training
    distillation_train(
        mock_components["student_policy"],
        mock_components["teacher_policy"],
        mock_components["student_generation"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["task_to_env"],
        mock_components["val_task_to_env"],
        mock_components["logger"],
        mock_components["checkpointer"],
        distillation_save_state,
        mock_components["master_config"],
    )

    assert mock_components["student_policy"].train.call_count == 5


def test_validate_function(mock_components):
    """Test independent validation function to ensure validation logic correctness."""
    # Run validation
    val_metrics, validation_timings = validate(
        mock_components["student_generation"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["val_task_to_env"],
        step=0,
        master_config=mock_components["master_config"],
    )

    # Verify validation results
    assert isinstance(val_metrics, dict)
    assert isinstance(validation_timings, dict)
    # For distillation, we don't need environment interaction since max_rollout_turns=0
    # The validation focuses on generation and teacher-student knowledge transfer
    # Note: validate() function itself doesn't call logger.log_metrics - that's done by the caller


def test_check_vocab_equality_pass(monkeypatch):
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    # Should not raise
    check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_vocab_mismatch_raises(monkeypatch):
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = {"a": 0, "c": 2}
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_length_mismatch_raises(monkeypatch):
    # Same vocab mapping but different __len__ values
    vocab = {"a": 0, "b": 1}
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = vocab
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = vocab
    teacher_tokenizer.__len__.return_value = 3

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_config_vocab_size_mismatch_raises(monkeypatch):
    vocab = {"a": 0, "b": 1}
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = vocab
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = vocab
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 3

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")
