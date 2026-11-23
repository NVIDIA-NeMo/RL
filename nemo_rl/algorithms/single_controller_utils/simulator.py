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

"""Stand-ins that let the SingleController run one side of the RL loop alone.

A simulator run (``examples/run_single_controller_simulator.py``) keeps the
SingleController's own pumps, data plane, sampler and refit lifecycle, and swaps
one side of the loop for a mock that uses no GPU. This module holds the
``rollout_only`` slice: the generation backend, NeMo-Gym and the TransferQueue
data plane are real, while the trainer is :class:`NoOpTrainer` and the weight
transfer is :class:`NoOpWeightSynchronizer`. The train pump consumes each
selected batch without training on it, the weight version still advances and
generation is still paused and resumed around every refit, so the rollout side
sees the same step rhythm it sees in training, minus the GPU work.
"""

from __future__ import annotations

import asyncio
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

import ray
from pydantic import BaseModel, NonNegativeFloat, PositiveInt

from nemo_rl.algorithms import opd as opd_module
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    is_ppo_run,
)
from nemo_rl.data_plane import is_metrics_client
from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import DataPlaneClient, DataPlaneRuntimeConfig
from nemo_rl.data_plane.observability import reduce_step_snapshots
from nemo_rl.models.generation.dynamo.config import dynamo_enabled
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm.config import (
    REFITTABLE_FP8_KV_CACHE_DTYPES,
    VllmConfig,
    parse_nvfp4_pertoken_rollout,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

logger = logging.getLogger(__name__)

SimulatorMode = Literal["rollout_only"]
TrainBackend = Literal["cpu"]

# Poll every 50 ms to limit idle time charged to simulated training.
_SIMULATED_TRAIN_POLL_S: float = 0.05


class SimulatorConfig(BaseModel, extra="forbid", frozen=True):
    """What a simulator run replaces, and how long the stand-ins take.

    The schema of the simulator YAML (``examples/configs/simulator/``), which
    ``examples/run_single_controller_simulator.py`` loads next to an SC recipe.
    """

    # Which side of the loop is real.
    mode: SimulatorMode = "rollout_only"
    # The trainer. cpu is the no-op stand-in: it trains nothing and uses no GPU.
    train_backend: TrainBackend = "cpu"
    # Hold once after all chunks arrive; per-chunk forward/backward overlap is omitted.
    # End early when idle: admission waits for a version advance or, with windowed
    # sampling, buffer capacity released at step cleanup. Log the skipped time.
    train_time_s: NonNegativeFloat = 0.0
    # Seconds the stand-in weight transfer holds inside the refit pause, where
    # the real transfer would keep generation frozen.
    refit_time_s: NonNegativeFloat = 0.0
    # Inference nodes of this allocation; overrides the recipe's generation
    # node count. None keeps the recipe's.
    num_nodes: Optional[PositiveInt] = None
    # Steps to run; overrides the recipe's max_num_steps. None keeps the recipe's.
    num_steps: Optional[PositiveInt] = None

    @property
    def uses_no_op_trainer(self) -> bool:
        """Whether setup builds a :class:`NoOpTrainer` in place of ``TQPolicy``."""
        return self.train_backend == "cpu"


def apply_simulator_overrides(
    raw_config: dict[str, Any], settings: SimulatorConfig
) -> None:
    """Override, in place, the few recipe fields a simulator run must own.

    Applied to the resolved config dict before ``MasterConfig`` validates it, so
    the recipe stays the source of everything else: the backend, the rollout
    shape, the sampler, the engines and the Gym settings.

    Overridden:
      - ``max_num_steps`` (and ``max_num_epochs``, so the step count is not
        clamped by a short dataset) when ``num_steps`` is set.
      - the inference node count when ``num_nodes`` is set: every node of a
        rollout-only run serves generation, so the generation resources and
        ``cluster.num_nodes`` are the same number.
      - ``data.validation``: there is no validation pass without a trainer.
      - ``policy.generation.vllm_cfg.load_format`` defaults to ``auto``: the
        engine serves the checkpoint, since no refit follows to replace the
        ``dummy`` weights a training run starts from. An explicit ``dummy`` is
        rejected by :func:`validate_simulator_config`.
      - ``env.nemo_gym.is_trajectory_collection``: a NeMo-RL control-flow knob
        that run_grpo_nemo_gym.py pops before setup so it never reaches Gym's
        global config.
    """
    # Both algorithm blocks are Optional on MasterConfig; exactly one is set.
    algo_key = "grpo" if raw_config.get("grpo") is not None else "ppo"
    algo = raw_config.get(algo_key)
    if settings.num_steps is not None and algo is not None:
        algo["max_num_steps"] = settings.num_steps
        # Allow num_steps even when each epoch holds only one batch.
        recipe_epochs = algo.get("max_num_epochs")
        algo["max_num_epochs"] = (
            settings.num_steps
            if recipe_epochs is None
            else max(int(recipe_epochs), settings.num_steps)
        )
    if settings.num_nodes is not None:
        colocated = raw_config["policy"]["generation"]["colocated"]
        if not colocated["enabled"]:
            colocated["resources"]["num_nodes"] = settings.num_nodes
        raw_config["cluster"]["num_nodes"] = settings.num_nodes
    raw_config["data"]["validation"] = None
    generation = raw_config["policy"]["generation"]
    if generation["backend"] == "vllm" and "vllm_cfg" in generation:
        # Where configure_generation_config would default a training run to
        # "dummy" (the first refit fills the weights in), there is no refit here.
        generation["vllm_cfg"].setdefault("load_format", "auto")
    nemo_gym = raw_config["env"].get("nemo_gym")
    if isinstance(nemo_gym, dict):
        nemo_gym.pop("is_trajectory_collection", None)


def reject_online_quantization(generation_config: GenerationConfig) -> None:
    """Refuse engines that quantize from the weights a refit sends.

    fp8 precision, vLLM NVFP4 per-token rollout and ModelOpt quant_cfg /
    real_quant all quantize at refit time. A simulator run has no refit, so
    such an engine would serve whatever the checkpoint holds, which is not what
    the recipe trains with. Point a quantized recipe at a pre-quantized
    checkpoint instead.
    """
    config = cast(dict[str, Any], generation_config)
    backend = config["backend"]
    if backend not in ("vllm", "trtllm"):
        return
    name = f"{backend}_cfg"
    backend_cfg = config.get(name)
    precision = backend_cfg.get("precision") if backend_cfg is not None else None
    online: list[str] = []
    if precision is not None and str(precision).lower() == "fp8":
        online.append(f"{name}.precision=fp8")
    if backend == "vllm":
        if parse_nvfp4_pertoken_rollout(cast(VllmConfig, config)) is not None:
            online.append("generation.nvfp4_pertoken_rollout")
        if config.get("quant_cfg") is not None:
            online.append("generation.quant_cfg")
        if config.get("real_quant"):
            online.append("generation.real_quant")
    if online:
        raise ValueError(
            f"{', '.join(online)} quantizes the engine online from the weights a "
            "trainer refit sends; a simulator run has no refit. Use an "
            "unquantized engine, or point the model at a pre-quantized checkpoint."
        )


def validate_simulator_config(
    master_config: MasterConfig, settings: SimulatorConfig
) -> None:
    """Refuse the recipes a rollout-only run cannot serve faithfully.

    Runs after ``validate_single_controller_config``, so the SC-wide rules
    (non-colocated vLLM, one optimizer step per RL step, no validation) are
    already enforced and only the trainer-less gaps are named here.
    """
    if not settings.uses_no_op_trainer:
        return
    if is_ppo_run(master_config):
        raise NotImplementedError(
            "rollout_only does not support PPO: the critic is a second trainer, "
            "and the cpu train backend stands in for neither."
        )
    generation_config = cast(dict[str, Any], master_config.policy["generation"])
    backend = generation_config["backend"]
    if backend != "vllm":
        raise NotImplementedError(
            "rollout_only currently supports policy.generation.backend='vllm' "
            f"only; got {backend!r}. Megatron generation serves the trainer's own "
            "weights, so it cannot run without one; trtllm and sglang have "
            "not been validated under the SingleController refit lifecycle "
            "this run drives."
        )
    if dynamo_enabled(generation_config):
        raise NotImplementedError(
            "rollout_only does not support policy.generation.dynamo=true yet: "
            "Dynamo serving has not been validated under the refit lifecycle "
            "this run drives. Set policy.generation.dynamo=false."
        )
    reject_online_quantization(master_config.policy["generation"])
    if generation_config["vllm_cfg"].get("load_format") == "dummy":
        raise ValueError(
            "rollout_only cannot run with vllm_cfg.load_format='dummy': no refit "
            "follows to replace the dummy weights, so the engine would serve "
            "random weights for the whole run. Drop the override; the driver "
            "defaults load_format to 'auto'."
        )
    kv_cache_dtype = generation_config["vllm_cfg"].get("kv_cache_dtype")
    if kv_cache_dtype in REFITTABLE_FP8_KV_CACHE_DTYPES:
        raise NotImplementedError(
            f"vllm_cfg.kv_cache_dtype={kv_cache_dtype!r} calibrates its KV scales "
            "on the trainer at every refit, which rollout_only has no trainer for."
        )
    megatron = cast(Optional[dict[str, Any]], master_config.policy.get("megatron_cfg"))
    if megatron is not None and megatron["enabled"]:
        peft = megatron.get("peft")
        if (
            master_config.checkpointing.pretrained_checkpoint is not None
            or master_config.policy.get("pretrained_checkpoint") is not None
            or (
                peft is not None
                and peft["enabled"]
                and peft.get("restore_from") is not None
            )
        ):
            raise ValueError(
                "rollout_only cannot load trainer-side initial checkpoints. "
                "Use an HF-exported initial policy via policy.model_name and "
                "remove the trainer-only checkpoint selectors."
            )
    if master_config.checkpointing.enabled:
        raise ValueError(
            "rollout_only requires checkpointing.enabled=false: the cpu train "
            "backend has no model state to save, and the replay-buffer "
            "checkpoint is anchored to a trainer checkpoint."
        )
    if opd_module.is_opd_enabled(master_config):
        raise NotImplementedError(
            "rollout_only does not support on-policy distillation: teacher "
            "logprobs are a trainer-side stage."
        )
    if master_config.token_capture.defer_routed_experts_to_policy:
        raise NotImplementedError(
            "token_capture.defer_routed_experts_to_policy assembles routes on "
            "the Megatron policy workers, which rollout_only does not build."
        )
    if master_config.token_capture.forest_training:
        raise NotImplementedError(
            "token_capture.forest_training validates forest padding on the "
            "trainer's workers at setup, which rollout_only does not build."
        )
    if master_config.loss_fn.truncated_importance_sampling_type == "adaptive-ppo-clip":
        raise NotImplementedError(
            "loss_fn.truncated_importance_sampling_type='adaptive-ppo-clip' reads "
            "the step's approx_entropy from the trainer, which the cpu train "
            "backend does not produce."
        )
    if master_config.async_rl.log_full_train_data:
        raise NotImplementedError(
            "async_rl.log_full_train_data dumps the rows the advantage stage "
            "prepares for training, which rollout_only does not run."
        )
    if master_config.async_rl.refit_actor.enabled:
        raise NotImplementedError(
            "async_rl.refit_actor.enabled moves a real weight transfer into its "
            "own actor; rollout_only has no transfer to move."
        )
    if master_config.async_rl.trainer_warmup.enabled:
        raise NotImplementedError(
            "async_rl.trainer_warmup.enabled runs a synthetic training pass on the "
            "trainer, which rollout_only does not have."
        )
    refit_timeout_s = master_config.async_rl.generation_fleet_health.refit_timeout_s
    if refit_timeout_s is not None and settings.refit_time_s >= refit_timeout_s:
        raise ValueError(
            f"simulator.refit_time_s ({settings.refit_time_s}) must be below "
            "async_rl.generation_fleet_health.refit_timeout_s "
            f"({refit_timeout_s}); otherwise every simulated refit is aborted as "
            "a stuck transfer. Raise the timeout or set it to null."
        )


async def simulate_train_step(
    train_time_s: float, rollouts_idle: Callable[[], bool]
) -> tuple[float, float]:
    """Hold where the optimizer step would run on a simulator run.

    One hold per step, after every chunk of the step has been consumed; forward
    and backward work that overlaps later chunks of the same step is not
    modeled. The hold ends early once ``rollouts_idle`` reports that nothing is
    generating: admission waits for the trainer version to advance or, with
    windowed sampling, for buffer capacity released at step cleanup, and neither
    happens before this returns, so the rest of the hold is idle on both sides.
    Idleness is polled every ``_SIMULATED_TRAIN_POLL_S``, which bounds the idle
    time charged to the held part.

    Args:
        train_time_s: Seconds to hold.
        rollouts_idle: Whether no rollout is dispatched or in flight; evaluated
            on every poll.

    Returns:
        ``(held_s, idle_skipped_s)``.
    """
    if train_time_s <= 0:
        return 0.0, 0.0
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + train_time_s
    while (remaining := deadline - loop.time()) > 0:
        if rollouts_idle():
            return loop.time() - started, remaining
        await asyncio.sleep(min(remaining, _SIMULATED_TRAIN_POLL_S))
    return loop.time() - started, 0.0


class NoOpWeightSynchronizer(WeightSynchronizer):
    """A refit that moves no weights.

    The engine already serves the checkpoint (``load_format: auto``), so there
    is nothing to deliver: ``is_stale`` is False from construction and the
    SingleController skips its startup sync. ``sync_weights`` holds
    ``refit_time_s`` so the generation pause around it lasts as long as a real
    transfer would.
    """

    def __init__(self, *, refit_time_s: float = 0.0) -> None:
        if refit_time_s < 0:
            raise ValueError("refit_time_s must be non-negative")
        self._refit_time_s = refit_time_s

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, float]]:
        """Hold for the simulated transfer time; nothing is transferred."""
        del timer, kv_scales
        if self._refit_time_s > 0:
            time.sleep(self._refit_time_s)
        return None

    @property
    def is_stale(self) -> bool:
        """Never: the engine loaded the real checkpoint at startup."""
        return False

    def init_communicator(self) -> None:
        """No transport, so nothing to initialize."""

    def shutdown(self) -> None:
        """No transport, so nothing to release."""


class NoOpTrainer:
    """Stand-in for ``TQPolicy`` that touches only the data plane, never a GPU.

    Owns the TransferQueue bootstrap ``TQPolicy`` normally performs, so the
    rest of setup (connect-only clients, partition registration, the replay
    buffer) is unchanged. The train pump consumes each selected batch without
    calling a trainer, so this class answers only what the shared step close
    and refit path ask of one: data-plane metrics and the pre-refit parameter
    sync, which has nothing to gather here.
    """

    def __init__(
        self,
        *,
        dp_cfg: DataPlaneRuntimeConfig,
        policy_config: PolicyConfig,
        storage_unit_node_ids: Optional[list[str]] = None,
        dp_client: Optional[DataPlaneClient] = None,
    ) -> None:
        """Bootstrap the data plane (or adopt ``dp_client``) and hold the config.

        Args:
            dp_cfg: ``master_config.data_plane``; bootstraps the TQ controller
                exactly as ``TQPolicy`` does.
            policy_config: ``master_config.policy``; read by the callers that
                expect ``trainer.cfg``.
            storage_unit_node_ids: Storage-unit plan forwarded to the bootstrap.
            dp_client: An already-built client to use instead of bootstrapping
                one. Tests pass an in-process client here.
        """
        self.cfg = policy_config
        if dp_client is None:
            dp_client = build_data_plane_client(
                dp_cfg,
                bootstrap=True,
                checkpointing=False,
                storage_unit_node_ids=storage_unit_node_ids,
            )
        self.dp_client = dp_client
        self._prev_cluster_snapshot: dict[str, Any] = {}
        self._extra_snapshot_sources: list[Any] = []
        self._extra_snapshot_fns: list[Callable[[], list[dict[str, Any]]]] = []

    def load_data_plane_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        """Restore TQ through the bootstrap client, as ``TQPolicy`` does."""
        return self.dp_client.load_checkpoint(checkpoint_dir)

    def shutdown(self) -> bool:
        """Close the data-plane client; there are no workers to stop."""
        try:
            self.dp_client.close()
        except Exception as error:  # noqa: BLE001 - mirrors TQPolicy.shutdown
            warnings.warn(f"Error closing data-plane client: {error}")
        return True

    def add_data_plane_snapshot_fn(
        self, fn: Callable[[], list[dict[str, Any]]]
    ) -> None:
        """Include a group of data-plane clients read through one call."""
        self._extra_snapshot_fns.append(fn)

    def add_data_plane_snapshot_source(self, actor: Any) -> None:
        """Include a non-worker actor's data-plane client in the cluster view."""
        self._extra_snapshot_sources.append(actor)

    def get_data_plane_step_metrics(
        self, step_time_s: float, driver_client: Any = None
    ) -> tuple[dict[str, float], str] | None:
        """This step's data-plane cost and the scope it covers, or ``None``.

        The contract of ``TQPolicy.get_data_plane_step_metrics``. The cluster
        view is the driver's clients plus the registered rollout-side sources;
        it falls back to the driver's alone when none of them answered.

        Args:
            step_time_s: Step wall time, for the share of the step.
            driver_client: The controller's client; defaults to this trainer's.

        Returns:
            ``(metrics, scope)`` with scope ``"cluster"`` or ``"driver"``, or
            None when data-plane observability is off.
        """
        driver_client = driver_client or self.dp_client
        if not is_metrics_client(driver_client):
            return None
        collect_started = time.perf_counter()
        snapshots = [driver_client.snapshot(reset_step_window=True)]
        if self.dp_client is not driver_client and is_metrics_client(self.dp_client):
            snapshots.append(self.dp_client.snapshot(reset_step_window=True))
        if self._extra_snapshot_sources:
            try:
                extra = ray.get(
                    [
                        source.get_data_plane_snapshot.remote()
                        for source in self._extra_snapshot_sources
                    ]
                )
            except Exception as error:  # noqa: BLE001 - metrics must never fail a step
                logger.warning("data-plane snapshot from actors failed: %s", error)
            else:
                snapshots.extend(snapshot for snapshot in extra if snapshot)
        for fn in self._extra_snapshot_fns:
            try:
                snapshots.extend(fn())
            except Exception as error:  # noqa: BLE001 - metrics must never fail a step
                logger.warning("data-plane snapshot from %s failed: %s", fn, error)
        collect_ms = (time.perf_counter() - collect_started) * 1e3
        metrics, scope, self._prev_cluster_snapshot = reduce_step_snapshots(
            driver_client,
            snapshots,
            self._prev_cluster_snapshot,
            step_time_s,
            collect_ms,
        )
        return metrics, scope

    def sync_params_before_refit(self) -> None:
        """No-op: no deferred all-gathers to materialize."""

    def start_gpu_profiling(self) -> None:
        """No-op: ``maybe_gpu_profile_step`` may address the trainer."""

    def stop_gpu_profiling(self) -> None:
        """No-op."""
