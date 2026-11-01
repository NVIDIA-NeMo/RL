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
from unittest.mock import MagicMock, call, create_autospec, patch

import pytest

from nemo_rl.algorithms.sft_v2 import (
    SFT_V2_TEED_METRICS,
    SFTSingleControllerActor,
    SFTV2SaveState,
    _max_train_steps,
)
from nemo_rl.data.energon.config import EnergonLoaderConfig
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.telemetry.instrumentation import TRACE_CARRIER_KWARG

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
    controller._tracer = None
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
        "defer_async_checkpoint_finalize": False,
    }
    sft.update(sft_overrides or {})
    data = {
        "backend": "energon",
        "validation": None,
        "max_input_seq_length": 128,
        "energon": SimpleNamespace(
            task_encoder=SimpleNamespace(packing=None), topology_mapper="default"
        ),
    }
    data.update(data_overrides or {})
    policy = {
        "megatron_cfg": {
            "enabled": True,
            "context_parallel_size": 1,
            "tensor_model_parallel_size": 1,
            "sequence_parallel": False,
            "use_fused_linear_logprobs": True,
        },
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "make_sequence_length_divisible_by": 1,
        "tokenizer": {},
        "train_global_batch_size": 8,
    }
    for section, values in (policy_overrides or {}).items():
        policy[section].update(values)
    # Most callers raise during validation and never read the rest. The fields below are
    # what `setup_sft_v2` needs once it gets past that, so one config serves both.
    return SimpleNamespace(
        sft=SimpleNamespace(**sft),
        data=data,
        policy=policy,
        checkpointing={"metric_name": metric_name},
        cluster=SimpleNamespace(
            num_nodes=1,
            segment_size=1,
            gpus_per_node=8,
            master_port_range_low=1,
            master_port_range_high=2,
        ),
        data_plane=SimpleNamespace(),
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
    assert {row.logger_key for row in SFT_V2_TEED_METRICS} <= set(metrics)


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
    with patch("nemo_rl.algorithms.sft_v2.shutdown_telemetry") as shutdown:
        controller.run()

    # check_save latches after firing once, so the loop must exit instead of
    # training unsaved until the walltime kill.
    assert controller._run_train_step.call_count == 2
    controller._save_checkpoint.assert_called_once_with({})
    shutdown.assert_called_once_with()


def test_loader_dispatch_carries_the_controller_trace() -> None:
    controller = _controller()
    controller._master_config = SimpleNamespace(
        sft=SimpleNamespace(only_unmask_final=False),
        policy={"make_sequence_length_divisible_by": 1},
    )
    controller._placement_plan = SimpleNamespace(logical_world_size=2)
    controller._trainer.worker_group.run_all_workers_single_data.return_value = [
        object(),
        object(),
    ]
    carrier = {"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"}

    with (
        patch(
            "nemo_rl.algorithms.sft_v2.trace_context_kwargs",
            return_value={TRACE_CARRIER_KWARG: carrier},
        ),
        patch(
            "nemo_rl.algorithms.sft_v2.ray.get",
            return_value=[_envelope(0), _envelope(1)],
        ),
    ):
        _ACTOR_CLS._load_envelopes(controller)

    kwargs = (
        controller._trainer.worker_group.run_all_workers_single_data.call_args.kwargs
    )
    assert kwargs[TRACE_CARRIER_KWARG] == carrier


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


def _deferred_save_controller(tmp_path: Path) -> object:
    controller = _save_controller()
    controller._save_state.total_steps = 10
    controller._defer_checkpoint_finalize = True
    controller._trainer = create_autospec(Policy, instance=True)
    controller._placement_plan = SimpleNamespace(logical_world_size=2)
    controller._loader_state_dicts = MagicMock(return_value=[{}, {}])
    controller._checkpointer = MagicMock()
    controller._checkpointer.init_tmp_checkpoint.return_value = str(tmp_path)
    controller._checkpointer.save_optimizer = True
    return controller


def test_save_checkpoint_finalizes_in_background_by_default(tmp_path: Path) -> None:
    controller = _deferred_save_controller(tmp_path)
    controller._defer_checkpoint_finalize = False

    controller._save_checkpoint({})

    controller._checkpointer.begin_finalization.assert_called_once_with(
        str(tmp_path), wait_fn=controller._trainer.finalize_async_save
    )
    assert controller._pending_checkpoint_path is None


def test_deferred_save_leaves_the_checkpoint_pending(tmp_path: Path) -> None:
    controller = _deferred_save_controller(tmp_path)

    controller._save_checkpoint({})

    controller._trainer.save_checkpoint.assert_called_once()
    controller._trainer.finalize_async_save.assert_not_called()
    controller._checkpointer.begin_finalization.assert_not_called()
    assert controller._pending_checkpoint_path == str(tmp_path)


def test_pending_save_publishes_only_after_a_successful_poll(tmp_path: Path) -> None:
    controller = _deferred_save_controller(tmp_path)
    controller._pending_checkpoint_path = str(tmp_path)
    controller._trainer.poll_async_save.side_effect = [False, True]

    # Writes still running: training continues, nothing is published.
    controller._finish_pending_save(blocking=False)
    controller._checkpointer.begin_finalization.assert_not_called()
    assert controller._pending_checkpoint_path == str(tmp_path)

    controller._finish_pending_save(blocking=False)
    controller._checkpointer.begin_finalization.assert_called_once_with(str(tmp_path))
    assert controller._pending_checkpoint_path is None
    controller._trainer.finalize_async_save.assert_not_called()

    # Nothing pending: no further worker calls.
    controller._finish_pending_save(blocking=False)
    assert controller._trainer.poll_async_save.call_count == 2


def test_blocking_finish_waits_for_the_write_then_publishes(tmp_path: Path) -> None:
    controller = _deferred_save_controller(tmp_path)
    controller._pending_checkpoint_path = str(tmp_path)
    order = MagicMock()
    order.attach_mock(controller._trainer.finalize_async_save, "finalize")
    order.attach_mock(controller._checkpointer.begin_finalization, "publish")

    controller._finish_pending_save(blocking=True)

    assert order.mock_calls == [call.finalize(), call.publish(str(tmp_path))]
    controller._trainer.poll_async_save.assert_not_called()
    assert controller._pending_checkpoint_path is None


def test_next_save_publishes_the_pending_one_first(tmp_path: Path) -> None:
    controller = _deferred_save_controller(tmp_path)
    previous = str(tmp_path / "tmp_step_5")
    controller._pending_checkpoint_path = previous
    order = MagicMock()
    order.attach_mock(controller._trainer.finalize_async_save, "finalize")
    order.attach_mock(controller._checkpointer.begin_finalization, "publish")
    order.attach_mock(controller._trainer.save_checkpoint, "save")

    controller._save_checkpoint({})

    assert [c[0] for c in order.mock_calls] == ["finalize", "publish", "save"]
    assert order.mock_calls[1] == call.publish(previous)
    assert controller._pending_checkpoint_path == str(tmp_path)


def test_run_polls_before_each_step_and_finalizes_at_exit() -> None:
    controller = _save_controller()
    controller._max_steps = 3
    controller._logger = MagicMock()
    controller._checkpointer = MagicMock()
    controller._close_loaders = MagicMock()
    controller._save_checkpoint = MagicMock()
    controller._timeout = MagicMock()
    controller._timeout.check_save.return_value = False
    controller._finish_pending_save = MagicMock()
    order = MagicMock()
    order.attach_mock(controller._finish_pending_save, "finish")

    def advance() -> dict[str, Any]:
        order.step()
        controller._save_state.total_steps += 1
        return {}

    controller._run_train_step = MagicMock(side_effect=advance)
    with patch("nemo_rl.algorithms.sft_v2.shutdown_telemetry"):
        controller.run()

    assert order.mock_calls == [
        call.finish(blocking=False),
        call.step(),
        call.finish(blocking=False),
        call.step(),
        call.finish(blocking=False),
        call.step(),
        call.finish(blocking=True),
        # Teardown's publish; a no-op once the exit path has published.
        call.finish(blocking=True),
    ]


def _failing_run_controller(tmp_path: Path) -> object:
    """Deferred-save controller whose step after a pending save raises."""
    controller = _deferred_save_controller(tmp_path)
    controller._pending_checkpoint_path = str(tmp_path)
    # The step-10 save is pending; step 11 then raises.
    controller._max_steps = controller._save_state.total_steps + 1
    controller._logger = MagicMock()
    controller._close_loaders = MagicMock()
    controller._timeout = MagicMock()
    controller._run_train_step = MagicMock(
        side_effect=RuntimeError("injected train failure")
    )
    # The writes are still running at the step's poll.
    controller._trainer.poll_async_save.return_value = False
    return controller


def test_run_failure_still_publishes_the_pending_save(tmp_path: Path) -> None:
    controller = _failing_run_controller(tmp_path)
    order = MagicMock()
    order.attach_mock(controller._trainer.finalize_async_save, "finalize")
    order.attach_mock(controller._checkpointer.begin_finalization, "publish")
    order.attach_mock(controller._checkpointer.shutdown, "close")

    with (
        patch("nemo_rl.algorithms.sft_v2.shutdown_telemetry"),
        pytest.raises(RuntimeError, match="injected train failure"),
    ):
        controller.run()

    # Published before checkpoint close, which joins the rename.
    assert order.mock_calls == [
        call.finalize(),
        call.publish(str(tmp_path)),
        call.close(),
    ]
    assert controller._pending_checkpoint_path is None


def test_failed_teardown_publish_does_not_mask_the_run_failure(
    tmp_path: Path,
) -> None:
    controller = _failing_run_controller(tmp_path)
    controller._trainer.finalize_async_save.side_effect = RuntimeError(
        "injected finalize failure"
    )

    with (
        patch("nemo_rl.algorithms.sft_v2.shutdown_telemetry"),
        pytest.raises(RuntimeError, match="injected train failure"),
        pytest.warns(UserWarning, match="pending checkpoint publish failed"),
    ):
        controller.run()

    controller._checkpointer.begin_finalization.assert_not_called()
    controller._checkpointer.shutdown.assert_called_once()


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
            {"sft_overrides": {"defer_async_checkpoint_finalize": True}},
            "requires policy.megatron_cfg.checkpoint.async_save",
        ),
        (
            {
                "sft_overrides": {"defer_async_checkpoint_finalize": True},
                "policy_overrides": {
                    "megatron_cfg": {"checkpoint": {"async_save": False}}
                },
            },
            "requires policy.megatron_cfg.checkpoint.async_save",
        ),
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
            "energon": SimpleNamespace(
                task_encoder=SimpleNamespace(
                    packing=SimpleNamespace(
                        name=sequence_packing.get("algorithm", "greedy_knapsack"),
                        options=SimpleNamespace(),
                    )
                )
            ),
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


@pytest.mark.parametrize(
    "field", ["max_sequence_length", "sequence_length_pad_multiple"]
)
def test_packing_options_reject_removed_geometry_fields(field: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        EnergonLoaderConfig(
            model_family="qwen",
            task_encoder={
                "packing": {
                    "name": "greedy_knapsack",
                    "buffer_size": 64,
                    "options": {field: 8},
                }
            },
        )


@pytest.mark.parametrize("restored", [False, True])
def test_setup_loaders_passes_canonical_geometry_and_nested_packing(
    restored: bool,
) -> None:
    loader_config = EnergonLoaderConfig(
        model_family="qwen",
        task_encoder={
            "packing": {
                "name": "balanced_greedy_knapsack",
                "buffer_size": 64,
                "options": {"max_sequences_per_bin": 16, "balanced_knapsack_delta": 2},
            }
        },
    )
    controller = object.__new__(_ACTOR_CLS)
    controller._master_config = SimpleNamespace(
        data={
            "max_input_seq_length": 128,
            "energon": loader_config,
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
    controller._loader_states = [{"shard": 0}, {"shard": 1}] if restored else None
    controller._trainer = MagicMock()
    futures = [object(), object()]
    controller._trainer.worker_group.run_all_workers_single_data.return_value = futures
    controller._trainer.worker_group.run_all_workers_multiple_data.return_value = (
        futures
    )

    with patch("nemo_rl.algorithms.sft_v2.ray.get", return_value=[True, True]):
        controller._setup_loaders()

    if restored:
        call_kwargs = controller._trainer.worker_group.run_all_workers_multiple_data.call_args.kwargs
        assert call_kwargs["restored_state"] == controller._loader_states
        kwargs = call_kwargs["common_kwargs"]
    else:
        kwargs = controller._trainer.worker_group.run_all_workers_single_data.call_args.kwargs
    assert kwargs["data_config"]["energon"] is loader_config
    assert kwargs["batch_size"] == 2
    assert kwargs["max_sequence_length"] == 128
    assert kwargs["sequence_length_pad_multiple"] == 8
    assert (
        not {
            "packing_algorithm",
            "max_sequences_per_bin",
        }
        & kwargs.keys()
    )


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
            "energon": SimpleNamespace(
                task_encoder=SimpleNamespace(
                    packing=SimpleNamespace(
                        name="greedy_knapsack",
                        options=SimpleNamespace(),
                    )
                )
            ),
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


@pytest.mark.parametrize("pad_multiple", [0, -1])
def test_setup_rejects_nonpositive_canonical_packing_alignment(
    pad_multiple: int,
) -> None:
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(
        data_overrides={
            "energon": EnergonLoaderConfig(
                model_family="qwen",
                task_encoder={
                    "packing": {
                        "name": "greedy_knapsack",
                        "buffer_size": 64,
                        "options": {},
                    }
                },
            )
        },
        policy_overrides={"sequence_packing": {"enabled": True, "fuse_loss": True}},
    )
    config.policy["make_sequence_length_divisible_by"] = pad_multiple
    with pytest.raises(ValueError, match="to be positive"):
        setup_sft_v2(config, MagicMock())


@pytest.mark.parametrize(
    "pretrained", [None, {"format": "megatron_bridge", "path": "/checkpoint"}]
)
def test_setup_propagates_pretrained_checkpoint_before_validation(pretrained):
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    config = _valid_setup_config(data_overrides={"backend": "invalid"})
    config.checkpointing["pretrained_checkpoint"] = pretrained
    with pytest.raises(ValueError, match="data.backend=energon"):
        setup_sft_v2(config, MagicMock())
    assert config.policy.get("pretrained_checkpoint") == pretrained


def test_setup_keeps_policy_pretrained_checkpoint_when_checkpointing_is_null():
    from nemo_rl.algorithms.sft_v2 import setup_sft_v2

    existing = {"format": "megatron_bridge", "path": "/policy-level"}
    config = _valid_setup_config(data_overrides={"backend": "invalid"})
    config.policy["pretrained_checkpoint"] = existing
    config.checkpointing["pretrained_checkpoint"] = None
    with pytest.raises(ValueError, match="data.backend=energon"):
        setup_sft_v2(config, MagicMock())
    assert config.policy["pretrained_checkpoint"] == existing


def test_setup_charges_the_placement_wait_to_cluster_init() -> None:
    """The placement groups are created before the policy, so the wait is measured.

    ``RayVirtualCluster.__init__`` only stores fields; the bundles are scheduled on the
    first ``get_placement_groups()``, which blocks on ``pg.ready()``. Drop that call and
    ``RayWorkerGroup`` triggers it during the ``TQPolicy`` build instead, so a wait that
    can dominate a large start is reported as ``policy_init`` and ``cluster_init`` reads
    zero. Every other test here raises during validation, so this is the only one that
    reaches the timing at all.

    The stubs advance a fake clock themselves, which makes each phase an exact number
    rather than something that has to be bounded loosely against a real one.
    """
    from nemo_rl.algorithms import sft_v2

    calls: list[str] = []
    clock = {"t": 0.0}

    def advance(label: str, seconds: float, result: Any = None):
        def _advance(*args: Any, **kwargs: Any) -> Any:
            calls.append(label)
            clock["t"] += seconds
            return result

        return _advance

    cluster = MagicMock()
    cluster.get_placement_groups.side_effect = advance("placement", 7.0)
    probe = MagicMock()
    probe.get_latest_checkpoint_path.return_value = None
    probe.load_training_info.return_value = None
    probe.get_resume_paths.return_value = (None, None)

    with (
        patch.object(sft_v2, "time", SimpleNamespace(monotonic=lambda: clock["t"])),
        patch.object(sft_v2, "set_seed"),
        patch.object(sft_v2, "CheckpointManager", return_value=probe),
        patch.object(
            sft_v2, "prepare_segment_topology", return_value=(None, None, None)
        ),
        patch.object(sft_v2, "RayVirtualCluster", return_value=cluster),
        patch.object(
            sft_v2, "TQPolicy", side_effect=advance("policy", 3.0, MagicMock())
        ),
        patch.object(sft_v2, "resolve_topology_mapper") as resolve_mapper,
        patch.object(sft_v2, "NLLLossFn"),
        patch.object(sft_v2, "_restore_save_state"),
        patch.object(sft_v2, "_max_train_steps", return_value=10),
    ):
        resolve_mapper.return_value.map.return_value = SimpleNamespace(
            logical_world_size=4, placement_hash="hash"
        )
        actor_args = sft_v2.setup_sft_v2(_valid_setup_config(), MagicMock())

    assert calls == ["placement", "policy"], (
        f"the placement wait must be taken before the policy build, got {calls}"
    )
    timing = actor_args.setup_timing
    assert timing.extras["cluster_init_time_s"] == 7.0
    assert timing.policy_init_time_s == 3.0, (
        "policy_init must cover the policy build alone, not the placement wait"
    )
    assert timing.total_setup_time_s == 10.0
