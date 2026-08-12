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

from functools import partial
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.loss import NLLLossFn
from nemo_rl.algorithms.sft import (
    MasterConfig,
    SFTConfig,
    _get_sft_save_state,
    _initial_sft_save_state,
    setup,
    sft_train,
    validate,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_get_sft_save_state_handles_legacy_checkpoint_and_filters_metrics():
    assert _get_sft_save_state({}) == _initial_sft_save_state()

    loaded_state = {
        "epoch": 1,
        "step": 3,
        "total_steps": 13,
        "consumed_samples": 32,
        "val:loss": 0.25,
    }

    save_state = _get_sft_save_state(loaded_state)

    assert vars(save_state) == {
        "epoch": 1,
        "step": 3,
        "total_steps": 13,
        "consumed_samples": 32,
        "total_valid_tokens": 0,
    }
    assert "total_valid_tokens" not in loaded_state


@pytest.fixture
def mock_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {"global_valid_toks": [10]},
    }

    # Create a proper message log structure with token_ids
    mock_batch = {
        "message_log": [[{"token_ids": torch.tensor([1, 2, 3]), "role": "assistant"}]],
        "loss_multiplier": torch.tensor(1.0),
    }

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

    loss_fn = NLLLossFn()
    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock master config
    master_config = MasterConfig.model_construct(
        sft=SFTConfig.model_construct(
            max_num_steps=5,
            max_num_epochs=2,
            val_period=100,
            val_batches=1,
            val_global_batch_size=1,
            val_micro_batch_size=1,
            val_at_start=False,
            val_at_end=False,
            only_unmask_final=False,
        ),
        policy={
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
        },
        checkpointing={
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
        },
        cluster={
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
    )

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "master_config": master_config,
    }


def test_exit_on_max_steps(mock_components):
    """Test that training loop exits when max_num_steps is reached"""
    # Set max steps to 12, which is less than len(train_dataloader) * max_num_epochs
    mock_components["master_config"].sft.max_num_steps = 12

    sft_save_state = _initial_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we only trained for 12 steps.
    assert mock_components["policy"].train.call_count == 12


def test_exit_on_max_epochs(mock_components):
    """Test that training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_components["master_config"].sft.max_num_epochs = 2
    mock_components["master_config"].sft.max_num_steps = 100

    sft_save_state = _initial_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we trained for exactly two epochs (20 batches).
    assert mock_components["policy"].train.call_count == 20


def test_exit_on_timeout(mock_components, capsys):
    """Test that training loop exits when timeout is reached"""
    # Set max steps and epochs to large numbers
    mock_components["master_config"].sft.max_num_steps = 100
    mock_components["master_config"].sft.max_num_epochs = 10

    sft_save_state = _initial_sft_save_state()

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.sft.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        # Create a side_effect that returns False 7 times, then True
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Run training
        sft_train(
            mock_components["policy"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["master_config"],
            mock_components["logger"],
            mock_components["checkpointer"],
            sft_save_state,
        )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_components["policy"].train.call_count == 8

        # Verify the timeout message was printed and is near the end (not followed by more training)
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # Verify no new epoch started after timeout (which would indicate a bug where break was used instead of return)
        remaining_lines = output_lines[timeout_line_idx:]
        for line in remaining_lines:
            assert "Epoch" not in line or "Epoch 1/10" in line, (
                f"Training continued to next epoch after timeout: {line}"
            )


def test_training_with_disabled_validation(mock_components):
    """Test that training works when validation is disabled (val_dataloader=None, val_period<=0)"""
    mock_components["master_config"].sft.val_period = 0
    mock_components["master_config"].sft.max_num_steps = 5
    mock_components["master_config"].sft.max_num_epochs = 1

    sft_save_state = _initial_sft_save_state()

    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        None,  # val_dataloader is None
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    assert mock_components["policy"].train.call_count == 5


def test_training_with_negative_val_period(mock_components):
    """Test that training works when val_period is negative (validation disabled)"""
    mock_components["master_config"].sft.val_period = -1
    mock_components["master_config"].sft.max_num_steps = 3
    mock_components["master_config"].sft.max_num_epochs = 1

    sft_save_state = _initial_sft_save_state()

    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        None,  # val_dataloader is None
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    assert mock_components["policy"].train.call_count == 3


def test_ft_save_period_triggers_periodic_saves(mock_components):
    """ft_save_period triggers checkpoint saves independent of save_period."""
    cfg = mock_components["master_config"]
    cfg.sft.val_period = 0
    cfg.sft.max_num_steps = 5
    cfg.sft.max_num_epochs = 1
    cfg.checkpointing["enabled"] = True
    cfg.checkpointing["save_period"] = 100  # only the final step would save
    cfg.checkpointing["ft_save_period"] = 2
    cfg.checkpointing["metric_name"] = None

    checkpointer = mock_components["checkpointer"]
    checkpointer.init_tmp_checkpoint.return_value = "/tmp/ft_ckpt_test/tmp_step"

    sft_save_state = _initial_sft_save_state()

    with patch("nemo_rl.algorithms.sft.torch.save"):
        sft_train(
            mock_components["policy"],
            mock_components["train_dataloader"],
            None,
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            cfg,
            mock_components["logger"],
            checkpointer,
            sft_save_state,
        )

    # ft_save_period=2 -> steps 2, 4; save_period=100 contributes only the last
    # step (5). Each save calls init_tmp_checkpoint(step, ...).
    saved_steps = [c.args[0] for c in checkpointer.init_tmp_checkpoint.call_args_list]
    assert saved_steps == [2, 4, 5]


def test_setup_rejects_only_unmask_final_for_direct_packed_sft_before_side_effects(
    mock_components,
):
    master_config = mock_components["master_config"]
    master_config.sft.only_unmask_final = True
    master_config.data = {"shuffle": False, "num_workers": 0}
    master_config.logger = {}
    master_config.policy["megatron_cfg"] = {
        "enabled": True,
        "context_parallel_size": 1,
    }
    master_config.policy["train_micro_batch_size"] = 1
    master_config.policy["dynamic_batching"] = {"enabled": False}
    master_config.policy["sequence_packing"] = {"enabled": False}
    master_config.policy["draft"] = {"enabled": False}
    master_config.policy["router_replay"] = {"enabled": False}
    train_dataset = MagicMock()
    train_dataset.task_data_processors = {"megatron_sft_packed": MagicMock()}

    with patch(
        "nemo_rl.algorithms.sft.Logger",
        side_effect=AssertionError("setup continued into logger initialization"),
    ):
        with pytest.raises(
            ValueError,
            match=r"sft\.only_unmask_final=true.*direct Megatron-LM prepacked SFT",
        ):
            setup(
                master_config,
                mock_components["tokenizer"],
                train_dataset,
                None,
            )


def test_setup_rejects_missing_megatron_config_for_direct_packed_sft(
    mock_components,
):
    master_config = mock_components["master_config"]
    master_config.data = {"shuffle": False, "num_workers": 0}
    master_config.logger = {}
    master_config.policy["train_micro_batch_size"] = 1
    master_config.policy["dynamic_batching"] = {"enabled": False}
    master_config.policy["sequence_packing"] = {"enabled": False}
    master_config.policy["draft"] = {"enabled": False}
    master_config.policy["router_replay"] = {"enabled": False}
    train_dataset = MagicMock()
    train_dataset.task_data_processors = {"megatron_sft_packed": MagicMock()}

    with pytest.raises(ValueError, match="requires the Megatron backend"):
        setup(
            master_config,
            mock_components["tokenizer"],
            train_dataset,
            None,
        )


@pytest.mark.parametrize(
    ("policy_update", "sft_update", "message"),
    [
        ({"megatron_cfg": {"enabled": False}}, {}, "requires the Megatron backend"),
        ({"draft": {"enabled": True}}, {}, "does not support draft training"),
        (
            {"dynamic_batching": {"enabled": True}},
            {},
            "requires dynamic batching to be disabled",
        ),
        (
            {"train_micro_batch_size": 2},
            {},
            "requires policy.train_micro_batch_size=1",
        ),
        (
            {"router_replay": {"enabled": True}},
            {},
            "does not support router replay",
        ),
        ({}, {"val_micro_batch_size": 2}, "requires sft.val_micro_batch_size=1"),
    ],
)
def test_setup_rejects_incompatible_direct_packed_policy_before_side_effects(
    mock_components,
    policy_update,
    sft_update,
    message,
):
    master_config = mock_components["master_config"]
    master_config.data = {"shuffle": False, "num_workers": 0}
    master_config.logger = {}
    master_config.policy.update(
        {
            "megatron_cfg": {
                "enabled": True,
                "use_fused_linear_logprobs": False,
                "context_parallel_size": 1,
            },
            "train_micro_batch_size": 1,
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
            "draft": {"enabled": False},
            "router_replay": {"enabled": False},
        }
    )
    master_config.policy.update(policy_update)
    for key, value in sft_update.items():
        setattr(master_config.sft, key, value)

    train_dataset = MagicMock()
    train_dataset.task_data_processors = {"megatron_sft_packed": MagicMock()}
    val_dataset = MagicMock()
    val_dataset.task_data_processors = {"megatron_sft_packed": MagicMock()}

    with patch(
        "nemo_rl.algorithms.sft.Logger",
        side_effect=AssertionError("setup continued into logger initialization"),
    ):
        with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
            setup(
                master_config,
                mock_components["tokenizer"],
                train_dataset,
                val_dataset,
            )


def test_setup_allows_sequence_packing_for_direct_packed_context_parallel_sft(
    mock_components,
):
    master_config = mock_components["master_config"]
    master_config.data = {"shuffle": False, "num_workers": 0}
    master_config.logger = {}
    master_config.policy.update(
        {
            "megatron_cfg": {
                "enabled": True,
                "use_fused_linear_logprobs": False,
                "context_parallel_size": 2,
            },
            "train_micro_batch_size": 1,
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": True},
            "draft": {"enabled": False},
            "router_replay": {"enabled": False},
        }
    )
    processor = partial(MagicMock(), context_parallel_size=2)
    train_dataset = MagicMock()
    train_dataset.task_data_processors = {
        "megatron_sft_packed": (MagicMock(), processor)
    }

    with patch(
        "nemo_rl.algorithms.sft.Logger",
        side_effect=AssertionError("setup continued into logger initialization"),
    ):
        with pytest.raises(
            AssertionError,
            match="setup continued into logger initialization",
        ):
            setup(
                master_config,
                mock_components["tokenizer"],
                train_dataset,
                None,
            )


def test_setup_rejects_direct_packed_context_parallel_mismatch_before_side_effects(
    mock_components,
):
    master_config = mock_components["master_config"]
    master_config.data = {"shuffle": False, "num_workers": 0}
    master_config.logger = {}
    master_config.policy.update(
        {
            "megatron_cfg": {
                "enabled": True,
                "use_fused_linear_logprobs": False,
                "context_parallel_size": 2,
            },
            "train_micro_batch_size": 1,
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
            "draft": {"enabled": False},
            "router_replay": {"enabled": False},
        }
    )
    processor = partial(MagicMock(), context_parallel_size=1)
    train_dataset = MagicMock()
    train_dataset.task_data_processors = {
        "megatron_sft_packed": (MagicMock(), processor)
    }

    with patch(
        "nemo_rl.algorithms.sft.Logger",
        side_effect=AssertionError("setup continued into logger initialization"),
    ):
        with pytest.raises(
            ValueError,
            match=r"prepared for context_parallel_size=1.*context_parallel_size=2",
        ):
            setup(
                master_config,
                mock_components["tokenizer"],
                train_dataset,
                None,
            )


def test_setup_rejects_mixed_direct_packed_and_regular_datasets_before_side_effects(
    mock_components,
):
    master_config = mock_components["master_config"]
    master_config.data = {"shuffle": False, "num_workers": 0}
    master_config.logger = {}
    master_config.policy.update(
        {
            "megatron_cfg": {
                "enabled": True,
                "use_fused_linear_logprobs": False,
                "context_parallel_size": 1,
            },
            "train_micro_batch_size": 1,
            "dynamic_batching": {"enabled": False},
            "sequence_packing": {"enabled": False},
            "draft": {"enabled": False},
            "router_replay": {"enabled": False},
        }
    )
    train_dataset = MagicMock()
    train_dataset.task_data_processors = {
        "megatron_sft_packed": MagicMock(),
        "regular_sft": MagicMock(),
    }

    with patch(
        "nemo_rl.algorithms.sft.Logger",
        side_effect=AssertionError("setup continued into logger initialization"),
    ):
        with pytest.raises(
            ValueError,
            match="cannot mix direct Megatron-LM prepacked and regular datasets",
        ):
            setup(
                master_config,
                mock_components["tokenizer"],
                train_dataset,
                None,
            )


def _direct_packed_batch(batch_size: int = 1) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.arange(batch_size * 4).reshape(batch_size, 4),
            "target_ids": torch.arange(1, batch_size * 4 + 1).reshape(batch_size, 4),
            "token_mask": torch.ones(batch_size, 4),
            "position_ids": torch.arange(4).repeat(batch_size, 1),
            "input_lengths": torch.full((batch_size,), 4),
            "sample_mask": torch.ones(batch_size),
            "packed_cu_seqlens": torch.tensor([[0, 4]], dtype=torch.int32).repeat(
                batch_size, 1
            ),
            "packed_cu_seqlens_lengths": torch.full((batch_size,), 2),
            "packed_max_seqlen": torch.full((batch_size,), 4),
        }
    )


def _packed_sft_row(row_id: int, context_parallel_size: int) -> dict[str, object]:
    return {
        "input_ids": torch.tensor([row_id, 1, 2, 3]),
        "target_ids": torch.tensor([1, 2, 3, 4]),
        "token_mask": torch.ones(4),
        "position_ids": torch.arange(4),
        "packed_cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "packed_max_seqlen": 4,
        "packed_context_parallel_size": context_parallel_size,
        "length": 4,
        "loss_multiplier": 1.0,
        "idx": row_id,
        "task_name": "megatron_sft_packed",
    }


def test_sft_collate_preserves_packed_row_order_before_data_parallel_sharding():
    from nemo_rl.algorithms.sft import _build_sft_collate_fn

    collate_fn = _build_sft_collate_fn(
        {
            "megatron_cfg": {
                "enabled": True,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 2,
            }
        },
        {"num_nodes": 1, "gpus_per_node": 8},
    )

    batch = collate_fn(
        [_packed_sft_row(row_id, context_parallel_size=2) for row_id in range(8)]
    )

    assert batch["input_ids"][:, 0].tolist() == list(range(8))


def test_sft_collate_rejects_rows_prepared_for_different_context_parallel_size():
    from nemo_rl.algorithms.sft import _build_sft_collate_fn

    collate_fn = _build_sft_collate_fn(
        {
            "megatron_cfg": {
                "enabled": True,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 2,
            }
        },
        {"num_nodes": 1, "gpus_per_node": 2},
    )

    with pytest.raises(
        ValueError,
        match=r"prepared for context_parallel_size=1.*context_parallel_size=2",
    ):
        collate_fn([_packed_sft_row(0, context_parallel_size=1)])


def test_sft_train_bypasses_online_message_repacking_for_direct_packed_rows(
    mock_components,
):
    direct_batch = _direct_packed_batch()
    train_dataloader = mock_components["train_dataloader"]
    train_dataloader.__iter__ = lambda self: iter([direct_batch])
    train_dataloader.__len__ = MagicMock(return_value=1)
    mock_components["master_config"].sft.max_num_steps = 1
    mock_components["master_config"].sft.max_num_epochs = 1

    with (
        patch("nemo_rl.algorithms.sft.add_loss_mask_to_message_log") as add_mask,
        patch("nemo_rl.algorithms.sft.batched_message_log_to_flat_message") as flatten,
    ):
        sft_train(
            mock_components["policy"],
            train_dataloader,
            None,
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["master_config"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _initial_sft_save_state(),
        )

    assert (add_mask.call_count, flatten.call_count) == (0, 0)
    assert mock_components["policy"].train.call_args.args[0] is direct_batch


def test_validate_bypasses_online_message_repacking_for_direct_packed_rows(
    mock_components,
):
    direct_batch = _direct_packed_batch()
    val_dataloader = mock_components["val_dataloader"]
    val_dataloader.__iter__ = lambda self: iter([direct_batch])
    val_dataloader.__len__ = MagicMock(return_value=1)
    mock_components["policy"].sharding_annotations.get_axis_size.return_value = 1

    with (
        patch("nemo_rl.algorithms.sft.add_loss_mask_to_message_log") as add_mask,
        patch("nemo_rl.algorithms.sft.batched_message_log_to_flat_message") as flatten,
    ):
        validate(
            mock_components["policy"],
            val_dataloader,
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            step=0,
            master_config=mock_components["master_config"],
            val_batches=1,
            val_batch_size=1,
            val_mbs=1,
        )

    assert (add_mask.call_count, flatten.call_count) == (0, 0)
    assert mock_components["policy"].train.call_args.args[0] is direct_batch


def test_sft_train_keeps_repacking_legacy_online_message_batches(mock_components):
    train_dataloader = mock_components["train_dataloader"]
    train_dataloader.__len__ = MagicMock(return_value=1)
    mock_components["master_config"].sft.max_num_steps = 1
    mock_components["master_config"].sft.max_num_epochs = 1

    sft_train(
        mock_components["policy"],
        train_dataloader,
        None,
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["checkpointer"],
        _initial_sft_save_state(),
    )

    trained_batch = mock_components["policy"].train.call_args.args[0]
    assert "message_log" not in trained_batch and torch.equal(
        trained_batch["input_ids"][0, :3], torch.tensor([1, 2, 3])
    )
