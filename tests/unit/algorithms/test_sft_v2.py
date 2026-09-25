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

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from nemo_rl.algorithms.sft_v2 import (
    SFTSingleControllerActor,
    SFTV2SaveState,
    _max_train_steps,
)
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.models.policy.lm_policy import Policy

_ACTOR_CLS = SFTSingleControllerActor.__ray_metadata__.modified_class


@pytest.mark.parametrize(
    ("config_name", "use_processor"), [("sft.yaml", False), ("sft_vlm_3B.yaml", True)]
)
def test_tokenizer_mode_is_defined_by_config(config_name, use_processor):
    from nemo_rl.utils.config import load_config

    config = load_config(
        Path(__file__).resolve().parents[3] / "examples" / "configs" / config_name
    )
    assert config.policy.tokenizer.use_processor is use_processor


def _envelope(rank: int, *, source_count: int = 1) -> StepEnvelope:
    return StepEnvelope(
        meta=KVBatchMeta(
            partition_id=f"p{rank}",
            task_name="train",
            sample_ids=[f"s{rank}"],
            fields=["input_ids"],
            sequence_lengths=[8],
        ),
        logical_rank=rank,
        logical_world_size=2,
        source_ids=tuple(
            f"source-{rank}-{source_index}" for source_index in range(source_count)
        ),
        field_names=("input_ids",),
        sequence_lengths=(8,),
        load_seconds=0.1 + rank * 0.1,
        valid_tokens=4,
    )


def _controller() -> object:
    controller = object.__new__(_ACTOR_CLS)
    controller._trainer = MagicMock()
    controller._trainer.finish_train_step.return_value = {
        "loss": 1.0,
        "grad_norm": 0.5,
        "all_mb_metrics": {},
    }
    controller._master_config = SimpleNamespace()
    controller._save_state = SFTV2SaveState(0, 0, 0, "hash")
    controller._load_envelopes = MagicMock(return_value=[_envelope(0), _envelope(1)])
    controller._owner_call = MagicMock(return_value=[None, None])
    controller._loss_fn = object()
    return controller


def _valid_setup_config(
    *,
    sft_overrides: dict[str, Any] | None = None,
    data_overrides: dict[str, Any] | None = None,
    policy_overrides: dict[str, dict[str, Any]] | None = None,
    metric_name: str | None = None,
) -> SimpleNamespace:
    sft = {
        "seed": 0,
        "val_period": 0,
        "val_at_start": False,
        "val_at_end": False,
    }
    sft.update(sft_overrides or {})
    data = {
        "backend": "energon",
        "validation": None,
        "max_input_seq_length": 128,
        "energon": SimpleNamespace(packing_buffer_size=None),
    }
    data.update(data_overrides or {})
    policy = {
        "megatron_cfg": {
            "enabled": True,
            "context_parallel_size": 1,
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
        },
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "make_sequence_length_divisible_by": 1,
    }
    for section, values in (policy_overrides or {}).items():
        policy[section].update(values)
    return SimpleNamespace(
        sft=SimpleNamespace(**sft),
        data=data,
        policy=policy,
        checkpointing={"metric_name": metric_name},
    )


def test_train_step_orders_split_policy_lifecycle_and_commit() -> None:
    controller = _controller()
    controller._load_envelopes.return_value = [
        _envelope(0, source_count=2),
        _envelope(1),
    ]

    metrics = controller._run_train_step()

    controller._trainer.begin_train_step.assert_called_once_with(controller._loss_fn)
    controller._trainer.train_placed_microbatches.assert_called_once()
    controller._trainer.finish_train_step.assert_called_once_with()
    controller._owner_call.assert_called_once_with("commit_sft_batch")
    assert controller._save_state.total_steps == 1
    assert controller._save_state.consumed_samples == 3
    assert metrics["valid_tokens"] == 8
    assert metrics["source_samples"] == 3
    assert metrics["physical_packs"] == 2


def test_train_step_aborts_policy_and_loader_on_training_failure() -> None:
    controller = _controller()
    controller._trainer.train_placed_microbatches.side_effect = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        controller._run_train_step()

    controller._trainer.abort_train_step.assert_called_once_with()
    controller._owner_call.assert_called_once_with("abort_sft_batch")
    assert controller._save_state.total_steps == 0


def test_train_step_mfu_excludes_loading_and_loader_commit() -> None:
    controller = _controller()
    controller._trainer.finish_train_step.return_value.update(
        total_flops=1e15, theoretical_tflops=500.0, num_ranks=2
    )
    # Loading takes 10 s, training 4 s, and loader commit another 6 s.
    with patch(
        "nemo_rl.algorithms.sft_v2.time.monotonic",
        side_effect=[0.0, 10.0, 14.0, 20.0, 20.0],
    ):
        metrics = controller._run_train_step()
    assert metrics["policy_time"] == 4.0
    assert metrics["total_step_time"] == 20.0
    assert metrics["train_fp_utilization"] == pytest.approx(0.5)


def test_train_step_omits_mfu_for_unsupported_model() -> None:
    assert "train_fp_utilization" not in _controller()._run_train_step()


def test_run_logs_mfu_at_the_optimizer_step() -> None:
    controller = _controller()
    controller._max_steps = 1
    controller._logger = MagicMock()
    controller._timeout = MagicMock()
    controller._timeout.check_save.return_value = False
    controller._should_save = MagicMock(return_value=False)
    controller._checkpoint_metric = MagicMock(return_value={})
    controller._close_loaders = MagicMock()
    controller._checkpointer = MagicMock()
    controller._trainer.finish_train_step.return_value.update(
        total_flops=1e15, theoretical_tflops=500.0, flops_from_bridge=1.0
    )
    with patch(
        "nemo_rl.algorithms.sft_v2.time.monotonic",
        side_effect=[0.0, 10.0, 14.0, 20.0, 20.0],
    ):
        controller.run()
    logged, step = controller._logger.log_metrics.call_args.args
    assert step == 1
    assert logged["train_fp_utilization"] == pytest.approx(0.5)
    assert logged["flops_from_bridge"] == 1.0


def _save_controller(**checkpointing: Any) -> object:
    controller = _controller()
    controller._master_config.checkpointing = {
        "enabled": True,
        "save_period": 10,
        "metric_name": None,
        **checkpointing,
    }
    controller._max_steps = 25
    return controller


def test_should_save_honors_save_period_ft_period_and_timeout() -> None:
    controller = _save_controller(ft_save_period=4)

    controller._save_state.total_steps = 3
    assert not controller._should_save(save_by_timeout=False)
    # A timeout save has to land between save_period boundaries, or a preempted
    # run loses everything since the last periodic save.
    assert controller._should_save(save_by_timeout=True)

    for step in (4, 10, 25):  # ft_save_period, save_period, final step
        controller._save_state.total_steps = step
        assert controller._should_save(save_by_timeout=False)


def test_should_save_is_disabled_by_the_checkpointing_flag() -> None:
    controller = _save_controller(enabled=False)
    controller._save_state.total_steps = 10

    assert not controller._should_save(save_by_timeout=True)


def test_run_stops_after_a_timeout_checkpoint() -> None:
    controller = _save_controller()
    controller._max_steps = 5
    controller._logger = MagicMock()
    controller._checkpointer = MagicMock()
    controller._close_loaders = MagicMock()
    controller._save_checkpoint = MagicMock()
    controller._timeout = MagicMock()
    controller._timeout.check_save.side_effect = [False, True]

    def advance() -> dict[str, Any]:
        controller._save_state.total_steps += 1
        return {}

    controller._run_train_step = MagicMock(side_effect=advance)
    controller.run()

    # check_save latches after firing once, so the loop must exit instead of
    # training unsaved until the walltime kill.
    assert controller._run_train_step.call_count == 2
    controller._save_checkpoint.assert_called_once_with({})


@pytest.mark.parametrize(("step", "is_final"), [(10, False), (25, True)])
def test_save_checkpoint_uses_policy_signature_and_terminal_step(
    tmp_path: Path, step: int, is_final: bool
) -> None:
    controller = _save_controller()
    controller._save_state.total_steps = step
    # Enforce the public signature so removed or missing keywords fail this test.
    controller._trainer = create_autospec(Policy, instance=True)
    controller._placement_plan = SimpleNamespace(logical_world_size=2)
    controller._loader_state_dicts = MagicMock(return_value=[{}, {}])
    controller._checkpointer = MagicMock()
    controller._checkpointer.init_tmp_checkpoint.return_value = str(tmp_path)
    controller._checkpointer.save_optimizer = True

    controller._save_checkpoint({})

    controller._trainer.save_checkpoint.assert_called_once_with(
        weights_path=str(tmp_path / "policy" / "weights"),
        optimizer_path=str(tmp_path / "policy" / "optimizer"),
        tokenizer_path=str(tmp_path / "policy" / "tokenizer"),
        is_final_checkpoint=is_final,
    )


def test_checkpoint_metric_tags_the_configured_train_metric() -> None:
    controller = _save_controller(metric_name="train:loss")

    assert controller._checkpoint_metric({"loss": 1.5, "grad_norm": 0.5}) == {
        "train:loss": 1.5
    }
    # CheckpointManager looks the value up under the full prefixed name.
    assert _save_controller()._checkpoint_metric({"loss": 1.5}) == {}


def test_checkpoint_metric_rejects_a_metric_no_step_produces() -> None:
    controller = _save_controller(metric_name="train:accuracy")

    # Every step reports the same keys, so a name that misses once misses
    # always -- fail on step 1 rather than at the first checkpoint.
    with pytest.raises(ValueError, match="did not produce"):
        controller._checkpoint_metric({"loss": 1.5})


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        ({"data_overrides": {"backend": "hf"}}, "requires data.backend=energon"),
        (
            {"policy_overrides": {"megatron_cfg": {"enabled": False}}},
            "only the Megatron policy",
        ),
        (
            {"policy_overrides": {"sequence_packing": {"enabled": True}}},
            "fixed batching",
        ),
        (
            {"policy_overrides": {"dynamic_batching": {"enabled": True}}},
            "fixed batching",
        ),
        ({"sft_overrides": {"val_period": 10}}, "has no validation loop"),
        (
            {"data_overrides": {"validation": {"path": "/dataset"}}},
            "reads no validation source",
        ),
        (
            {"data_overrides": {"max_input_seq_length": None}},
            "max_input_seq_length",
        ),
    ],
)
def test_setup_rejects_configs_sft_v2_cannot_run(
    config_overrides: dict[str, Any], message: str
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(**config_overrides)

    with pytest.raises(ValueError, match=message):
        setup_sft_v2(config, MagicMock())


def test_setup_rejects_a_validation_checkpoint_metric() -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    with pytest.raises(ValueError, match="training metrics only"):
        setup_sft_v2(
            _valid_setup_config(metric_name="val:val_loss"),
            MagicMock(),
        )


@pytest.mark.parametrize(
    (
        "sequence_overrides",
        "dynamic_enabled",
        "megatron_overrides",
        "pad_multiple",
        "max_sequence_length",
        "message",
    ),
    [
        (
            {"fuse_loss": False},
            False,
            {},
            1,
            128,
            "Energon packing requires sequence_packing enabled with fuse_loss.",
        ),
        (
            {"algorithm": "unknown"},
            False,
            {},
            1,
            128,
            "Energon SFT requires a supported packing algorithm.",
        ),
        (
            {},
            True,
            {},
            1,
            128,
            "Energon packing does not support dynamic batching.",
        ),
        (
            {},
            False,
            {},
            4,
            130,
            "Energon packing requires max_input_seq_length to be divisible by make_sequence_length_divisible_by.",
        ),
        (
            {},
            False,
            {"tensor_model_parallel_size": 2, "sequence_parallel": True},
            1,
            128,
            "Energon packing requires make_sequence_length_divisible_by to be a multiple of 2.",
        ),
    ],
)
def test_setup_rejects_invalid_energon_packing_config(
    sequence_overrides: dict[str, Any],
    dynamic_enabled: bool,
    megatron_overrides: dict[str, Any],
    pad_multiple: int,
    max_sequence_length: int,
    message: str,
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    sequence_packing = {
        "enabled": True,
        "fuse_loss": True,
        "algorithm": "greedy_knapsack",
    }
    sequence_packing.update(sequence_overrides)
    config = _valid_setup_config(
        data_overrides={
            "max_input_seq_length": max_sequence_length,
            "energon": SimpleNamespace(packing_buffer_size=64),
        },
        policy_overrides={
            "megatron_cfg": megatron_overrides,
            "sequence_packing": sequence_packing,
            "dynamic_batching": {"enabled": dynamic_enabled},
        },
    )
    config.policy["make_sequence_length_divisible_by"] = pad_multiple

    with pytest.raises(ValueError, match=re.escape(message)):
        setup_sft_v2(config, MagicMock())


def test_setup_loaders_enables_packing_from_energon_buffer() -> None:
    controller = object.__new__(_ACTOR_CLS)
    controller._master_config = SimpleNamespace(
        data={
            "max_input_seq_length": 128,
            "energon": SimpleNamespace(packing_buffer_size=64),
        },
        policy={
            "train_global_batch_size": 4,
            "sequence_packing": {
                "algorithm": "balanced_greedy_knapsack",
                "max_sequences_per_bin": 16,
            },
            "make_sequence_length_divisible_by": 8,
        },
        sft=SimpleNamespace(only_unmask_final=False),
    )
    controller._placement_plan = SimpleNamespace(
        logical_world_size=2,
        placement_hash="placement",
    )
    controller._loader_states = None
    controller._trainer = MagicMock()
    futures = [object(), object()]
    controller._trainer.worker_group.run_all_workers_single_data.return_value = futures

    with patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[True, True]):
        controller._setup_loaders()

    kwargs = (
        controller._trainer.worker_group.run_all_workers_single_data.call_args.kwargs
    )
    assert kwargs["packing_algorithm"] == "balanced_greedy_knapsack"
    assert kwargs["max_sequences_per_bin"] == 16


@pytest.mark.parametrize(
    ("megatron_overrides", "policy_multiple", "message"),
    [
        ({"context_parallel_size": 2}, 2, "multiple of 4"),
        (
            {
                "moe_token_dispatcher_type": "flex",
                "moe_flex_dispatcher_backend": "hybridep",
            },
            1,
            "HybridEP",
        ),
        (
            {"fp8_cfg": {"enabled": True, "fp8_recipe": "blockwise"}},
            1,
            "FP8 packed-token alignment",
        ),
    ],
)
def test_setup_rejects_unsupported_energon_packing_layouts(
    megatron_overrides: dict[str, Any], policy_multiple: int, message: str
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(
        data_overrides={
            "max_input_seq_length": 130,
            "energon": SimpleNamespace(packing_buffer_size=64),
        },
        policy_overrides={
            "megatron_cfg": megatron_overrides,
            "sequence_packing": {
                "enabled": True,
                "fuse_loss": True,
                "algorithm": "greedy_knapsack",
            },
        },
    )
    config.policy["make_sequence_length_divisible_by"] = policy_multiple

    with pytest.raises(ValueError, match=message):
        setup_sft_v2(config, MagicMock())


def test_restore_rejects_changed_placement() -> None:
    from nemo_rl.algorithms.sft_v2 import _restore_save_state

    with pytest.raises(ValueError, match="placement changed"):
        _restore_save_state(
            {"placement_hash": "old", "total_steps": 3}, placement_hash="new"
        )


def test_max_steps_is_bounded_by_virtual_epochs() -> None:
    config = SimpleNamespace(
        sft=SimpleNamespace(max_num_steps=100, max_num_epochs=3),
        data={
            "train": {
                "path": "/dataset",
                "split": "train",
                "virtual_epoch_length": 7,
            }
        },
    )

    assert _max_train_steps(config) == 21
