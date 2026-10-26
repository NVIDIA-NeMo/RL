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

"""Tests for the SingleController simulator stand-ins (rollout_only slice)."""

from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict

from examples import run_single_controller_simulator
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
    TokenCaptureConfig,
)
from nemo_rl.algorithms.single_controller_utils.simulator import (
    GENERATION_LOGPROBS_FIELD,
    PREV_LOGPROBS_FIELD,
    REFERENCE_LOGPROBS_FIELD,
    CpuTrainer,
    NoOpWeightSynchronizer,
    SimulatorSettings,
    apply_simulator_overrides,
    reject_online_quantization,
    validate_simulator_config,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient

_PARTITION = "rollout_data"
_DP_CFG: dict[str, Any] = {
    "enabled": True,
    "impl": "transfer_queue",
    "backend": "simple",
}


# ── settings ────────────────────────────────────────────────────────────────


def test_settings_default_to_rollout_only_with_cpu_trainer() -> None:
    settings = SimulatorSettings()
    assert settings.mode == "rollout_only"
    assert settings.uses_cpu_trainer


@pytest.mark.parametrize("mode", ["train_only", "train_rollout"])
def test_settings_reject_unimplemented_modes(mode: str) -> None:
    with pytest.raises(NotImplementedError, match=mode):
        SimulatorSettings(mode=mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["automodel", "mcore"])
def test_settings_reject_real_train_backends_in_rollout_only(backend: str) -> None:
    with pytest.raises(NotImplementedError, match="cpu"):
        SimulatorSettings(train_backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "bogus"},
        {"train_backend": "gpu"},
        {"train_time_s": -1.0},
        {"refit_time_s": -0.5},
        {"num_nodes": 0},
        {"num_steps": 0},
    ],
)
def test_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SimulatorSettings(**kwargs)


# ── config overrides ────────────────────────────────────────────────────────


def _raw_config(*, colocated: bool = False) -> dict[str, Any]:
    return {
        "grpo": {"max_num_steps": 1000, "max_num_epochs": 1},
        "policy": {
            "generation": {
                "backend": "vllm",
                "colocated": {
                    "enabled": colocated,
                    "resources": {"num_nodes": 2, "gpus_per_node": 8},
                },
            }
        },
        "cluster": {"num_nodes": 6, "gpus_per_node": 8},
        "data": {"train": {}, "validation": {"data_path": "val.jsonl"}},
        "env": {"nemo_gym": {"is_trajectory_collection": True, "config": {}}},
    }


def test_apply_overrides_sets_steps_nodes_and_drops_validation() -> None:
    raw = _raw_config()
    apply_simulator_overrides(
        raw, SimulatorSettings(num_steps=20, num_nodes=4, train_time_s=1.0)
    )
    assert raw["grpo"]["max_num_steps"] == 20
    # Widened so a short dataset cannot clamp the requested steps away.
    assert raw["grpo"]["max_num_epochs"] == 20
    assert raw["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 4
    assert raw["cluster"]["num_nodes"] == 4
    assert raw["data"]["validation"] is None
    assert "is_trajectory_collection" not in raw["env"]["nemo_gym"]


def test_apply_overrides_keeps_recipe_values_when_flags_absent() -> None:
    raw = _raw_config()
    apply_simulator_overrides(raw, SimulatorSettings())
    assert raw["grpo"]["max_num_steps"] == 1000
    assert raw["grpo"]["max_num_epochs"] == 1
    assert raw["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 2
    assert raw["cluster"]["num_nodes"] == 6
    # Always dropped: a rollout-only run has no validation pass.
    assert raw["data"]["validation"] is None


def test_apply_overrides_keeps_larger_recipe_epoch_count() -> None:
    raw = _raw_config()
    raw["grpo"]["max_num_epochs"] = 50
    apply_simulator_overrides(raw, SimulatorSettings(num_steps=3))
    assert raw["grpo"]["max_num_epochs"] == 50


def test_apply_overrides_colocated_recipe_only_touches_cluster_nodes() -> None:
    raw = _raw_config(colocated=True)
    apply_simulator_overrides(raw, SimulatorSettings(num_nodes=3))
    assert raw["cluster"]["num_nodes"] == 3
    assert raw["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 2


# ── validation ──────────────────────────────────────────────────────────────


def _master_config(
    *,
    backend: str = "vllm",
    vllm_cfg: dict[str, Any] | None = None,
    generation_extra: dict[str, Any] | None = None,
    checkpointing_enabled: bool = False,
    ppo: bool = False,
    tis_type: str | None = None,
    defer_routes: bool = False,
) -> MasterConfig:
    generation: dict[str, Any] = {
        "backend": backend,
        "colocated": {"enabled": False, "resources": {"num_nodes": 1}},
        "vllm_cfg": dict(vllm_cfg or {}),
        **(generation_extra or {}),
    }
    return MasterConfig.model_construct(
        policy={"generation": generation},
        checkpointing={"enabled": checkpointing_enabled},
        async_rl=AsyncRLConfig(),
        grpo=None if ppo else GRPOConfig(async_grpo=None),
        ppo=SimpleNamespace() if ppo else None,
        loss_fn=SimpleNamespace(truncated_importance_sampling_type=tis_type),
        token_capture=TokenCaptureConfig(defer_routed_experts_to_policy=defer_routes),
        on_policy_distillation=None,
    )


def test_validate_accepts_vllm_rollout_only_recipe() -> None:
    validate_simulator_config(_master_config(), SimulatorSettings(refit_time_s=5.0))


def test_validate_rejects_ppo() -> None:
    with pytest.raises(NotImplementedError, match="PPO"):
        validate_simulator_config(_master_config(ppo=True), SimulatorSettings())


@pytest.mark.parametrize("backend", ["megatron", "trtllm", "sglang", "dynamo"])
def test_validate_rejects_non_vllm_backends(backend: str) -> None:
    with pytest.raises(NotImplementedError, match="vllm"):
        validate_simulator_config(_master_config(backend=backend), SimulatorSettings())


def test_validate_rejects_checkpointing() -> None:
    with pytest.raises(ValueError, match="checkpointing.enabled=false"):
        validate_simulator_config(
            _master_config(checkpointing_enabled=True), SimulatorSettings()
        )


def test_validate_rejects_refittable_fp8_kv_cache() -> None:
    with pytest.raises(NotImplementedError, match="kv_cache_dtype"):
        validate_simulator_config(
            _master_config(vllm_cfg={"kv_cache_dtype": "fp8"}), SimulatorSettings()
        )


def test_validate_rejects_deferred_routed_experts() -> None:
    with pytest.raises(NotImplementedError, match="defer_routed_experts_to_policy"):
        validate_simulator_config(
            _master_config(defer_routes=True), SimulatorSettings()
        )


def test_validate_rejects_adaptive_ppo_clip() -> None:
    with pytest.raises(NotImplementedError, match="adaptive-ppo-clip"):
        validate_simulator_config(
            _master_config(tis_type="adaptive-ppo-clip"), SimulatorSettings()
        )


def test_validate_rejects_refit_hold_at_or_over_the_refit_timeout() -> None:
    config = _master_config()
    timeout = config.async_rl.generation_fleet_health.refit_timeout_s
    assert timeout is not None
    with pytest.raises(ValueError, match="refit_timeout_s"):
        validate_simulator_config(config, SimulatorSettings(refit_time_s=timeout))
    config.async_rl.generation_fleet_health.refit_timeout_s = None
    validate_simulator_config(config, SimulatorSettings(refit_time_s=timeout))


@pytest.mark.parametrize(
    ("generation", "name"),
    [
        ({"backend": "vllm", "vllm_cfg": {"precision": "fp8"}}, "precision=fp8"),
        ({"backend": "trtllm", "trtllm_cfg": {"precision": "FP8"}}, "precision=fp8"),
        ({"backend": "vllm", "vllm_cfg": {}, "quant_cfg": {"x": 1}}, "quant_cfg"),
        ({"backend": "vllm", "vllm_cfg": {}, "real_quant": True}, "real_quant"),
        (
            {
                "backend": "vllm",
                "vllm_cfg": {},
                "nvfp4_pertoken_rollout": {"enabled": True},
            },
            "nvfp4_pertoken_rollout",
        ),
    ],
)
def test_reject_online_quantization_names_the_offending_setting(
    generation: dict[str, Any], name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        reject_online_quantization(generation)  # type: ignore[arg-type]


def test_reject_online_quantization_passes_unquantized_and_other_backends() -> None:
    reject_online_quantization({"backend": "vllm", "vllm_cfg": {"precision": "bf16"}})  # type: ignore[arg-type]
    reject_online_quantization({"backend": "megatron"})  # type: ignore[arg-type]


# ── NoOpWeightSynchronizer ──────────────────────────────────────────────────


def test_noop_weight_synchronizer_is_never_stale_and_holds_the_refit_time() -> None:
    synchronizer = NoOpWeightSynchronizer(refit_time_s=0.05)
    assert synchronizer.is_stale is False
    synchronizer.init_communicator()
    started = time.perf_counter()
    assert synchronizer.sync_weights(kv_scales=None) is None
    assert time.perf_counter() - started >= 0.05
    assert synchronizer.sync_count == 1
    # No transport of its own, so there is no membership to reconcile.
    assert synchronizer.reconcile_communicator([0]) is None
    synchronizer.shutdown()


def test_noop_weight_synchronizer_rejects_negative_hold() -> None:
    with pytest.raises(ValueError):
        NoOpWeightSynchronizer(refit_time_s=-1.0)


# ── CpuTrainer ──────────────────────────────────────────────────────────────


def _trainer_with_group() -> tuple[
    NoOpDataPlaneClient, CpuTrainer, KVBatchMeta, torch.Tensor
]:
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id=_PARTITION,
        fields=[
            "input_ids",
            GENERATION_LOGPROBS_FIELD,
            PREV_LOGPROBS_FIELD,
            REFERENCE_LOGPROBS_FIELD,
        ],
        num_samples=4,
        consumer_tasks=["prev_lp", "ref_lp", "train"],
    )
    sample_ids = ["grp_g0", "grp_g1"]
    logprobs = torch.tensor([[-0.1, -0.2, -0.3], [-1.0, -2.0, -3.0]])
    client.put_samples(
        sample_ids=sample_ids,
        partition_id=_PARTITION,
        fields=TensorDict(
            {
                "input_ids": torch.arange(6).reshape(2, 3),
                GENERATION_LOGPROBS_FIELD: logprobs,
            },
            batch_size=(2,),
        ),
    )
    meta = KVBatchMeta(
        partition_id=_PARTITION,
        task_name="train",
        sample_ids=sample_ids,
        fields=["input_ids", GENERATION_LOGPROBS_FIELD],
        sequence_lengths=[3, 3],
        tags=[{}, {}],
    )
    trainer = CpuTrainer(dp_cfg=_DP_CFG, policy_config={}, dp_client=client)
    return client, trainer, meta, logprobs


def test_cpu_trainer_adopts_the_given_client_without_bootstrapping() -> None:
    client, trainer, _, _ = _trainer_with_group()
    assert trainer.dp_client is client
    assert trainer.cfg == {}


def test_cpu_trainer_logprob_stages_copy_the_generation_column() -> None:
    client, trainer, meta, logprobs = _trainer_with_group()

    trainer.get_logprobs_from_meta(meta)
    trainer.get_reference_policy_logprobs_from_meta(meta, micro_batch_size=1)

    rows = client.get_samples(
        sample_ids=meta.sample_ids,
        partition_id=_PARTITION,
        select_fields=[PREV_LOGPROBS_FIELD, REFERENCE_LOGPROBS_FIELD],
    )
    assert torch.equal(rows.get(PREV_LOGPROBS_FIELD), logprobs)
    assert torch.equal(rows.get(REFERENCE_LOGPROBS_FIELD), logprobs)


def test_cpu_trainer_train_step_counts_rows_and_returns_no_metrics() -> None:
    _, trainer, meta, _ = _trainer_with_group()
    trainer.prepare_for_training()
    trainer.begin_train_step(loss_fn=None)
    trainer.train_microbatches_from_meta(meta, train_fields=["input_ids"])
    trainer.train_microbatches_from_meta(meta)
    assert trainer.step_rows == 2 * meta.size
    assert trainer.finish_train_step() == {}
    assert trainer.steps_completed == 1


def test_cpu_trainer_rejects_out_of_order_step_calls() -> None:
    _, trainer, meta, _ = _trainer_with_group()
    with pytest.raises(RuntimeError, match="before begin"):
        trainer.train_microbatches_from_meta(meta)
    with pytest.raises(RuntimeError, match="without an open step"):
        trainer.finish_train_step()
    trainer.begin_train_step(loss_fn=None)
    with pytest.raises(RuntimeError, match="already open"):
        trainer.begin_train_step(loss_fn=None)
    trainer.abort_train_step()
    trainer.begin_train_step(loss_fn=None)


def test_cpu_trainer_gpu_phases_are_no_ops_and_read_through_the_data_plane() -> None:
    _, trainer, meta, _ = _trainer_with_group()
    for method in (
        trainer.prepare_for_training,
        trainer.finish_inference,
        trainer.offload_to_cpu,
        trainer.offload_before_refit,
        trainer.offload_after_refit,
        trainer.pause_train_step_with_offloading,
        trainer.sync_params_before_refit,
        trainer.start_gpu_profiling,
        trainer.stop_gpu_profiling,
    ):
        assert method() is None
    trainer.prepare_for_lp_inference(keep_train_buffers=True)
    data = trainer.read_from_dataplane(meta, select_fields=["input_ids"])
    assert data["input_ids"].shape == (2, 3)
    assert trainer.shutdown() is True


def test_cpu_trainer_refuses_what_only_a_real_trainer_can_do() -> None:
    _, trainer, _, _ = _trainer_with_group()
    with pytest.raises(NotImplementedError):
        trainer.calibrate_qkv_fp8_scales(None, include_q=True)
    with pytest.raises(NotImplementedError):
        trainer.save_checkpoint(weights_path="x")
    with pytest.raises(NotImplementedError):
        trainer.finalize_async_save()
    with pytest.raises(NotImplementedError):
        _ = trainer.sharding_annotations
    with pytest.raises(NotImplementedError):
        _ = trainer.worker_group


# ── the controller's simulated train step ───────────────────────────────────


def _bare_controller(settings: SimulatorSettings | None, *, inflight: int):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._simulator = settings
    ctrl._inflight_rollouts = inflight
    ctrl._dispatched_rollouts = set()
    ctrl._SIMULATED_TRAIN_POLL_S = 0.01
    return ctrl


def test_simulated_training_is_free_outside_a_simulator_run() -> None:
    ctrl = _bare_controller(None, inflight=1)
    assert asyncio.run(ctrl._simulate_training()) == (0.0, 0.0)
    ctrl = _bare_controller(SimulatorSettings(train_time_s=0.0), inflight=1)
    assert asyncio.run(ctrl._simulate_training()) == (0.0, 0.0)


def test_simulated_training_holds_the_full_time_while_rollouts_run() -> None:
    ctrl = _bare_controller(SimulatorSettings(train_time_s=0.1), inflight=1)
    held, skipped = asyncio.run(ctrl._simulate_training())
    assert held >= 0.1
    assert skipped == 0.0


def test_simulated_training_ends_early_once_rollouts_go_idle() -> None:
    ctrl = _bare_controller(SimulatorSettings(train_time_s=5.0), inflight=1)

    async def run() -> tuple[float, float]:
        async def go_idle() -> None:
            await asyncio.sleep(0.05)
            ctrl._inflight_rollouts = 0

        idle = asyncio.create_task(go_idle())
        result = await ctrl._simulate_training()
        await idle
        return result

    held, skipped = asyncio.run(run())
    assert held < 1.0
    assert skipped > 3.0
    assert abs(held + skipped - 5.0) < 0.5


def test_simulated_training_treats_an_unstarted_dispatch_as_busy() -> None:
    ctrl = _bare_controller(SimulatorSettings(train_time_s=0.1), inflight=0)
    ctrl._dispatched_rollouts = {object()}
    held, skipped = asyncio.run(ctrl._simulate_training())
    assert held >= 0.1
    assert skipped == 0.0


# ── entrypoint flags ────────────────────────────────────────────────────────


def test_entrypoint_parses_flags_and_passes_overrides_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_controller_simulator.py",
            "--config",
            "sc.yaml",
            "--num-steps",
            "7",
            "--num-nodes",
            "2",
            "--train-time-s",
            "12.5",
            "--refit-time-s",
            "3",
            "grpo.num_prompts_per_step=8",
            "++async_rl.max_inflight_prompts=8",
        ],
    )
    args, overrides = run_single_controller_simulator.parse_args()
    settings = run_single_controller_simulator.settings_from_args(args)
    assert args.config == "sc.yaml"
    # Flags left out stay None on the namespace; the dataclass supplies them.
    assert args.mode is None and args.train_backend is None
    assert settings == SimulatorSettings(
        mode="rollout_only",
        train_backend="cpu",
        train_time_s=12.5,
        refit_time_s=3.0,
        num_nodes=2,
        num_steps=7,
    )
    assert overrides == [
        "grpo.num_prompts_per_step=8",
        "++async_rl.max_inflight_prompts=8",
    ]


def test_entrypoint_rejects_unimplemented_mode_before_loading_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_controller_simulator.py",
            "--config",
            "sc.yaml",
            "--mode",
            "train_only",
        ],
    )
    args, _ = run_single_controller_simulator.parse_args()
    with pytest.raises(NotImplementedError, match="train_only"):
        run_single_controller_simulator.settings_from_args(args)
