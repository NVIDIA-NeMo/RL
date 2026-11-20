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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller_utils import simulator as simulator_module
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
    RefitActorConfig,
    TokenCaptureConfig,
)
from nemo_rl.algorithms.single_controller_utils.simulator import (
    NoOpTrainer,
    NoOpWeightSynchronizer,
    SGLangNoOpWeightSynchronizer,
    SimulatorConfig,
    apply_simulator_overrides,
    build_simulated_weight_synchronizer,
    reject_online_quantization,
    simulate_train_step,
    validate_simulator_config,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.observability import MetricsDataPlaneClient
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.config import load_config

_SIMULATOR_EXEMPLAR = (
    Path(__file__).parents[3] / "examples/configs/simulator/rollout_only.yaml"
)

_PARTITION = "rollout_data"
_DP_CFG: dict[str, Any] = {
    "enabled": True,
    "impl": "transfer_queue",
    "backend": "simple",
}


# ── simulator config ────────────────────────────────────────────────────────


def test_config_defaults_to_rollout_only_with_the_no_op_trainer() -> None:
    settings = SimulatorConfig()
    assert settings.mode == "rollout_only"
    assert settings.uses_no_op_trainer


def test_shipped_simulator_yaml_states_the_schema_defaults() -> None:
    """The exemplar documents every key at the value the schema defaults it to."""
    shipped = OmegaConf.to_container(load_config(_SIMULATOR_EXEMPLAR), resolve=True)
    assert isinstance(shipped, dict)
    assert shipped == SimulatorConfig().model_dump()
    assert SimulatorConfig.model_validate(shipped) == SimulatorConfig()


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
        raw, SimulatorConfig(num_steps=20, num_nodes=4, train_time_s=1.0)
    )
    assert raw["grpo"]["max_num_steps"] == 20
    # Widened so a short dataset cannot clamp the requested steps away.
    assert raw["grpo"]["max_num_epochs"] == 20
    assert raw["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 4
    assert raw["cluster"]["num_nodes"] == 4
    assert raw["data"]["validation"] is None
    assert "is_trajectory_collection" not in raw["env"]["nemo_gym"]


def test_apply_overrides_sizes_sglang_engines_from_num_nodes() -> None:
    raw = _raw_config()
    generation = raw["policy"]["generation"]
    generation["backend"] = "sglang"
    generation["sglang_cfg"] = {"sglang_server_config": {"num_gpus": 1}}
    apply_simulator_overrides(raw, SimulatorConfig(num_nodes=4))
    # SGLang counts its engines from num_gpus, so it follows the node count.
    assert generation["sglang_cfg"]["sglang_server_config"]["num_gpus"] == 4 * 8
    assert generation["colocated"]["resources"]["num_nodes"] == 4


def test_apply_overrides_keeps_recipe_values_when_keys_are_unset() -> None:
    raw = _raw_config()
    apply_simulator_overrides(raw, SimulatorConfig())
    assert raw["grpo"]["max_num_steps"] == 1000
    assert raw["grpo"]["max_num_epochs"] == 1
    assert raw["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 2
    assert raw["cluster"]["num_nodes"] == 6
    # Always dropped: a rollout-only run has no validation pass.
    assert raw["data"]["validation"] is None


def test_apply_overrides_keeps_larger_recipe_epoch_count() -> None:
    raw = _raw_config()
    raw["grpo"]["max_num_epochs"] = 50
    apply_simulator_overrides(raw, SimulatorConfig(num_steps=3))
    assert raw["grpo"]["max_num_epochs"] == 50


def test_apply_overrides_colocated_recipe_only_touches_cluster_nodes() -> None:
    raw = _raw_config(colocated=True)
    apply_simulator_overrides(raw, SimulatorConfig(num_nodes=3))
    assert raw["cluster"]["num_nodes"] == 3
    assert raw["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 2


def test_apply_overrides_defaults_load_format_to_auto() -> None:
    raw = _raw_config()
    raw["policy"]["generation"]["vllm_cfg"] = {}
    apply_simulator_overrides(raw, SimulatorConfig())
    # No refit follows to fill dummy weights in, so the engine loads the checkpoint.
    assert raw["policy"]["generation"]["vllm_cfg"]["load_format"] == "auto"


def test_apply_overrides_keeps_explicit_load_format() -> None:
    raw = _raw_config()
    raw["policy"]["generation"]["vllm_cfg"] = {"load_format": "safetensors"}
    apply_simulator_overrides(raw, SimulatorConfig())
    assert raw["policy"]["generation"]["vllm_cfg"]["load_format"] == "safetensors"


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
    forest_training: bool = False,
    sglang_pause_mode: str = "retract",
    sglang_needs_offload: bool = False,
    policy_extra: dict[str, Any] | None = None,
    pretrained_checkpoint: Any = None,
    async_rl: AsyncRLConfig | None = None,
) -> MasterConfig:
    generation: dict[str, Any] = {
        "backend": backend,
        "colocated": {"enabled": False, "resources": {"num_nodes": 1}},
        "vllm_cfg": dict(vllm_cfg or {}),
        **(generation_extra or {}),
    }
    if backend == "sglang":
        generation["sglang_cfg"] = {
            "sglang_server_config": {
                "pause_generation_mode": sglang_pause_mode,
                "needs_offload": sglang_needs_offload,
            }
        }
    return MasterConfig.model_construct(
        policy={"generation": generation, **(policy_extra or {})},
        checkpointing=CheckpointingConfig.model_construct(
            enabled=checkpointing_enabled,
            pretrained_checkpoint=pretrained_checkpoint,
        ),
        async_rl=async_rl or AsyncRLConfig(),
        grpo=None if ppo else GRPOConfig(async_grpo=None),
        ppo=SimpleNamespace() if ppo else None,
        loss_fn=SimpleNamespace(truncated_importance_sampling_type=tis_type),
        token_capture=TokenCaptureConfig(
            defer_routed_experts_to_policy=defer_routes,
            forest_training=forest_training,
        ),
        on_policy_distillation=None,
    )


def test_validate_accepts_vllm_rollout_only_recipe() -> None:
    validate_simulator_config(_master_config(), SimulatorConfig(refit_time_s=5.0))


def test_validate_rejects_ppo() -> None:
    with pytest.raises(NotImplementedError, match="PPO"):
        validate_simulator_config(_master_config(ppo=True), SimulatorConfig())


@pytest.mark.parametrize("backend", ["megatron", "trtllm"])
def test_validate_rejects_unsupported_backends(backend: str) -> None:
    with pytest.raises(NotImplementedError, match="vllm"):
        validate_simulator_config(_master_config(backend=backend), SimulatorConfig())


def test_validate_accepts_sglang_rollout_only_recipe() -> None:
    # No vllm_cfg at all: the vLLM-only checks must not be reached.
    master_config = _master_config(backend="sglang")
    del master_config.policy["generation"]["vllm_cfg"]
    validate_simulator_config(master_config, SimulatorConfig(refit_time_s=5.0))


def test_validate_rejects_sglang_in_place_pause() -> None:
    with pytest.raises(ValueError, match="in_place"):
        validate_simulator_config(
            _master_config(backend="sglang", sglang_pause_mode="in_place"),
            SimulatorConfig(),
        )


def test_validate_rejects_sglang_needs_offload() -> None:
    # Setup releases the engine memory and the no-op refit never restores it.
    with pytest.raises(ValueError, match="needs_offload"):
        validate_simulator_config(
            _master_config(backend="sglang", sglang_needs_offload=True),
            SimulatorConfig(),
        )


def test_validate_rejects_dynamo_serving() -> None:
    with pytest.raises(NotImplementedError, match="policy.generation.dynamo=true"):
        validate_simulator_config(
            _master_config(generation_extra={"dynamo": True}), SimulatorConfig()
        )


def test_validate_rejects_dummy_load_format() -> None:
    with pytest.raises(ValueError, match="load_format='dummy'"):
        validate_simulator_config(
            _master_config(vllm_cfg={"load_format": "dummy"}), SimulatorConfig()
        )


def test_validate_rejects_checkpointing() -> None:
    with pytest.raises(ValueError, match="checkpointing.enabled=false"):
        validate_simulator_config(
            _master_config(checkpointing_enabled=True), SimulatorConfig()
        )


@pytest.mark.parametrize(
    ("policy_extra", "pretrained_checkpoint"),
    [
        ({"megatron_cfg": {"enabled": True}}, SimpleNamespace(path="/ckpt")),
        (
            {
                "megatron_cfg": {"enabled": True},
                "pretrained_checkpoint": {"path": "/ckpt"},
            },
            None,
        ),
        (
            {
                "megatron_cfg": {
                    "enabled": True,
                    "peft": {"enabled": True, "restore_from": "/donor"},
                }
            },
            None,
        ),
    ],
)
def test_validate_rejects_trainer_side_initial_weights(
    policy_extra: dict[str, Any], pretrained_checkpoint: Any
) -> None:
    """The engine serves policy.model_name, so a trainer-only checkpoint would be ignored."""
    with pytest.raises(ValueError, match="trainer-side initial checkpoints"):
        validate_simulator_config(
            _master_config(
                policy_extra=policy_extra, pretrained_checkpoint=pretrained_checkpoint
            ),
            SimulatorConfig(),
        )


def test_validate_accepts_megatron_recipe_without_initial_weights() -> None:
    megatron = {"enabled": True, "peft": {"enabled": True, "restore_from": None}}
    validate_simulator_config(
        _master_config(policy_extra={"megatron_cfg": megatron}), SimulatorConfig()
    )


def test_validate_rejects_the_training_data_dump() -> None:
    with pytest.raises(NotImplementedError, match="log_full_train_data"):
        validate_simulator_config(
            _master_config(async_rl=AsyncRLConfig(log_full_train_data=True)),
            SimulatorConfig(),
        )


def test_validate_rejects_the_refit_actor() -> None:
    async_rl = AsyncRLConfig(refit_actor=RefitActorConfig(enabled=True))
    with pytest.raises(NotImplementedError, match="refit_actor"):
        validate_simulator_config(_master_config(async_rl=async_rl), SimulatorConfig())


def test_validate_rejects_refittable_fp8_kv_cache() -> None:
    with pytest.raises(NotImplementedError, match="kv_cache_dtype"):
        validate_simulator_config(
            _master_config(vllm_cfg={"kv_cache_dtype": "fp8"}), SimulatorConfig()
        )


def test_validate_rejects_deferred_routed_experts() -> None:
    with pytest.raises(NotImplementedError, match="defer_routed_experts_to_policy"):
        validate_simulator_config(_master_config(defer_routes=True), SimulatorConfig())


def test_validate_rejects_forest_training() -> None:
    with pytest.raises(NotImplementedError, match="forest_training"):
        validate_simulator_config(
            _master_config(forest_training=True), SimulatorConfig()
        )


def test_validate_rejects_adaptive_ppo_clip() -> None:
    with pytest.raises(NotImplementedError, match="adaptive-ppo-clip"):
        validate_simulator_config(
            _master_config(tis_type="adaptive-ppo-clip"), SimulatorConfig()
        )


def test_validate_rejects_refit_hold_at_or_over_the_refit_timeout() -> None:
    config = _master_config()
    timeout = config.async_rl.generation_fleet_health.refit_timeout_s
    assert timeout is not None
    with pytest.raises(ValueError, match="refit_timeout_s"):
        validate_simulator_config(config, SimulatorConfig(refit_time_s=timeout))
    config.async_rl.generation_fleet_health.refit_timeout_s = None
    validate_simulator_config(config, SimulatorConfig(refit_time_s=timeout))


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
    # No transport of its own, so there is no membership to reconcile.
    assert synchronizer.reconcile_communicator([0]) is None
    synchronizer.shutdown()


def test_noop_weight_synchronizer_rejects_negative_hold() -> None:
    with pytest.raises(ValueError):
        NoOpWeightSynchronizer(refit_time_s=-1.0)


class _FakeSGLangGeneration:
    """Records the refit lifecycle calls an SGLang synchronizer makes."""

    def __init__(self, *, invalidate_ok: bool = True) -> None:
        self.pause_generation_mode = "retract"
        self.calls: list[tuple[str, Any]] = []
        self._invalidate_ok = invalidate_ok

    def pause_generation(self, mode: str) -> None:
        self.calls.append(("pause_generation", mode))

    def invalidate_kv_cache(self) -> bool:
        self.calls.append(("invalidate_kv_cache", None))
        return self._invalidate_ok

    def continue_generation(self) -> None:
        self.calls.append(("continue_generation", None))


def test_sglang_noop_synchronizer_pauses_flushes_holds_and_resumes() -> None:
    gen = _FakeSGLangGeneration()
    synchronizer = SGLangNoOpWeightSynchronizer(generation=gen, refit_time_s=0.05)
    assert synchronizer.is_stale is False
    started = time.perf_counter()
    assert synchronizer.sync_weights(kv_scales=None) is None
    assert time.perf_counter() - started >= 0.05
    assert gen.calls == [
        ("pause_generation", "retract"),
        ("invalidate_kv_cache", None),
        ("continue_generation", None),
    ]


def test_sglang_noop_synchronizer_resumes_when_the_flush_fails() -> None:
    gen = _FakeSGLangGeneration(invalidate_ok=False)
    synchronizer = SGLangNoOpWeightSynchronizer(generation=gen, refit_time_s=0.0)
    with pytest.raises(RuntimeError, match="KV cache"):
        synchronizer.sync_weights(kv_scales=None)
    assert gen.calls[-1] == ("continue_generation", None)


def test_build_simulated_weight_synchronizer_dispatches_on_the_backend() -> None:
    gen = _FakeSGLangGeneration()
    vllm_sync = build_simulated_weight_synchronizer(
        generation_backend="vllm", generation=gen, refit_time_s=1.0
    )
    assert type(vllm_sync) is NoOpWeightSynchronizer
    assert vllm_sync._refit_time_s == 1.0
    sglang_sync = build_simulated_weight_synchronizer(
        generation_backend="sglang", generation=gen, refit_time_s=2.0
    )
    assert isinstance(sglang_sync, SGLangNoOpWeightSynchronizer)
    assert sglang_sync._refit_time_s == 2.0
    with pytest.raises(NotImplementedError, match="megatron"):
        build_simulated_weight_synchronizer(
            generation_backend="megatron", generation=gen, refit_time_s=0.0
        )


# ── NoOpTrainer ─────────────────────────────────────────────────────────────


def _metrics_client() -> MetricsDataPlaneClient:
    client = MetricsDataPlaneClient(NoOpDataPlaneClient())
    client.register_partition(
        partition_id=_PARTITION, fields=["x"], num_samples=1, consumer_tasks=["t"]
    )
    return client


def test_no_op_trainer_adopts_the_given_client_without_bootstrapping() -> None:
    client = NoOpDataPlaneClient()
    trainer = NoOpTrainer(dp_cfg=_DP_CFG, policy_config={}, dp_client=client)
    assert trainer.dp_client is client
    assert trainer.cfg == {}
    assert trainer.sync_params_before_refit() is None
    assert trainer.shutdown() is True


def test_no_op_trainer_reports_no_data_plane_metrics_without_observability() -> None:
    trainer = NoOpTrainer(
        dp_cfg=_DP_CFG, policy_config={}, dp_client=NoOpDataPlaneClient()
    )
    assert trainer.get_data_plane_step_metrics(1.0) is None


def test_no_op_trainer_data_plane_metrics_cover_the_registered_rollout_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = NoOpTrainer(dp_cfg=_DP_CFG, policy_config={}, dp_client=_metrics_client())
    metrics, scope = trainer.get_data_plane_step_metrics(1.0)
    assert scope == "driver"
    assert metrics

    def broken() -> list[dict[str, Any]]:
        raise RuntimeError("worker gone")

    actor = SimpleNamespace(
        get_data_plane_snapshot=SimpleNamespace(
            remote=lambda: _metrics_client().snapshot()
        )
    )
    monkeypatch.setattr(simulator_module.ray, "get", lambda refs: refs)
    trainer.add_data_plane_snapshot_source(actor)
    trainer.add_data_plane_snapshot_fn(broken)
    trainer.add_data_plane_snapshot_fn(lambda: [_metrics_client().snapshot()])
    # The controller's own client is a second driver-side process view.
    metrics, scope = trainer.get_data_plane_step_metrics(1.0, _metrics_client())
    assert scope == "cluster"
    # Driver client, trainer client, the actor and the one callable that answered.
    assert metrics["now/n_processes"] == 4


# ── the simulated train step ────────────────────────────────────────────────


def test_simulated_train_step_is_free_without_a_hold() -> None:
    assert asyncio.run(simulate_train_step(0.0, lambda: False)) == (0.0, 0.0)


def test_simulated_train_step_holds_the_full_time_while_rollouts_run() -> None:
    held, skipped = asyncio.run(simulate_train_step(0.1, lambda: False))
    assert held >= 0.1
    assert skipped == 0.0


def test_simulated_train_step_ends_within_one_poll_of_rollouts_going_idle() -> None:
    idle = False

    async def run() -> tuple[float, float]:
        async def go_idle() -> None:
            nonlocal idle
            await asyncio.sleep(0.2)
            idle = True

        task = asyncio.create_task(go_idle())
        result = await simulate_train_step(5.0, lambda: idle)
        await task
        return result

    held, skipped = asyncio.run(run())
    # Idle at 0.2 s, seen by the next 50 ms poll; the rest of the hold is skipped.
    assert 0.2 <= held < 0.2 + 4 * simulator_module._SIMULATED_TRAIN_POLL_S
    assert held + skipped == pytest.approx(5.0, abs=0.05)


def test_simulated_train_step_skips_the_whole_hold_when_already_idle() -> None:
    held, skipped = asyncio.run(simulate_train_step(5.0, lambda: True))
    assert held < 0.05
    assert skipped == pytest.approx(5.0, abs=0.05)
