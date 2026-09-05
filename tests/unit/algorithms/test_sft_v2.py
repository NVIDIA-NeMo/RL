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

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.sft_v2 import (
    SFTSingleControllerActor,
    SFTV2SaveState,
    _max_train_steps,
)
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data_plane import KVBatchMeta

_ACTOR_CLS = SFTSingleControllerActor.__ray_metadata__.modified_class


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
    }
    data.update(data_overrides or {})
    policy = {
        "megatron_cfg": {"enabled": True},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
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
            "fixed NeMo-RL batching",
        ),
        (
            {"policy_overrides": {"dynamic_batching": {"enabled": True}}},
            "fixed NeMo-RL batching",
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
