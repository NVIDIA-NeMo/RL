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
data plane are real, while the trainer is :class:`CpuTrainer` and the weight
transfer is :class:`NoOpWeightSynchronizer`. The weight version still advances
and generation is still paused and resumed around every refit, so the rollout
side sees the same step rhythm it sees in training, minus the GPU work.

The ``train_only`` and ``train_rollout`` modes and the ``cpu`` rollout backend
are the later slices of the same driver and are rejected here with an explicit
error rather than silently running something else.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, cast

from tensordict import TensorDict

from nemo_rl.algorithms import opd as opd_module
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    is_ppo_run,
)
from nemo_rl.data_plane.driver_mixin import TQDriverMixin
from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import (
    DataPlaneClient,
    DataPlaneRuntimeConfig,
    KVBatchMeta,
)
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm.config import (
    REFITTABLE_FP8_KV_CACHE_DTYPES,
    VllmConfig,
    parse_nvfp4_pertoken_rollout,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

SimulatorMode = Literal["rollout_only", "train_only", "train_rollout"]
TrainBackend = Literal["cpu", "automodel", "mcore"]

SIMULATOR_MODES: tuple[str, ...] = ("rollout_only", "train_only", "train_rollout")
TRAIN_BACKENDS: tuple[str, ...] = ("cpu", "automodel", "mcore")

# Columns the CpuTrainer fills in place of the policy / reference forwards.
GENERATION_LOGPROBS_FIELD = "generation_logprobs"
PREV_LOGPROBS_FIELD = "prev_logprobs"
REFERENCE_LOGPROBS_FIELD = "reference_policy_logprobs"


@dataclass(frozen=True)
class SimulatorSettings:
    """What a simulator run replaces, and how long the stand-ins take.

    Attributes:
        mode: Which side of the loop is real. Only ``rollout_only`` is implemented.
        train_backend: The trainer. ``rollout_only`` requires ``cpu``, the mock.
        train_time_s: Seconds the mock trainer holds each step where
            ``policy.train()`` would run. Generation keeps serving during the
            hold, as it does beside a non-colocated trainer. The hold ends early
            once no rollout is in flight, because nothing on the rollout side
            can change until the version advances; the skipped seconds are
            logged so the e2e step time can still be read off the run.
        refit_time_s: Seconds the mock weight transfer holds inside the refit
            pause, where the real transfer would keep generation frozen.
        num_nodes: Inference nodes of this allocation; overrides the recipe's
            generation node count. None keeps the recipe's.
        num_steps: Steps to run; overrides the recipe's ``max_num_steps``. None
            keeps the recipe's.
    """

    mode: SimulatorMode = "rollout_only"
    train_backend: TrainBackend = "cpu"
    train_time_s: float = 0.0
    refit_time_s: float = 0.0
    num_nodes: Optional[int] = None
    num_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in SIMULATOR_MODES:
            raise ValueError(
                f"simulator mode must be one of {SIMULATOR_MODES}; got {self.mode!r}"
            )
        if self.train_backend not in TRAIN_BACKENDS:
            raise ValueError(
                f"train_backend must be one of {TRAIN_BACKENDS}; got "
                f"{self.train_backend!r}"
            )
        if self.mode != "rollout_only":
            raise NotImplementedError(
                f"simulator mode {self.mode!r} is not implemented yet; only "
                "'rollout_only' is. 'train_only' needs the cpu rollout backend "
                "(synthetic rollouts), 'train_rollout' is the ordinary "
                "run_grpo_single_controller.py run."
            )
        if self.train_backend != "cpu":
            raise NotImplementedError(
                "rollout_only stubs the trainer out, so train_backend must be "
                f"'cpu'; got {self.train_backend!r}. A real train backend is the "
                "ordinary run_grpo_single_controller.py run."
            )
        if self.train_time_s < 0:
            raise ValueError("train_time_s must be non-negative")
        if self.refit_time_s < 0:
            raise ValueError("refit_time_s must be non-negative")
        if self.num_nodes is not None and self.num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if self.num_steps is not None and self.num_steps < 1:
            raise ValueError("num_steps must be positive")

    @property
    def uses_cpu_trainer(self) -> bool:
        """Whether setup builds a :class:`CpuTrainer` in place of ``TQPolicy``."""
        return self.train_backend == "cpu"


def apply_simulator_overrides(
    raw_config: dict[str, Any], settings: SimulatorSettings
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
      - ``env.nemo_gym.is_trajectory_collection``: a NeMo-RL control-flow knob
        that run_grpo_nemo_gym.py pops before setup so it never reaches Gym's
        global config.
    """
    # Both algorithm blocks are Optional on MasterConfig; exactly one is set.
    algo_key = "grpo" if raw_config.get("grpo") is not None else "ppo"
    algo = raw_config.get(algo_key)
    if settings.num_steps is not None and algo is not None:
        algo["max_num_steps"] = settings.num_steps
        # One epoch yields at least one step whenever the dataset holds a full
        # batch, so num_steps epochs can never clamp num_steps steps away. The
        # recipe may omit max_num_epochs and take the schema default.
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
    master_config: MasterConfig, settings: SimulatorSettings
) -> None:
    """Refuse the recipes a rollout-only run cannot serve faithfully.

    Runs after ``validate_single_controller_config``, so the SC-wide rules
    (non-colocated vLLM, one optimizer step per RL step, no validation) are
    already enforced and only the trainer-less gaps are named here.
    """
    if not settings.uses_cpu_trainer:
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
            "weights, so it cannot run without one; trtllm, sglang and dynamo "
            "have not been validated under the SingleController refit "
            "lifecycle this run drives."
        )
    reject_online_quantization(master_config.policy["generation"])
    kv_cache_dtype = generation_config["vllm_cfg"].get("kv_cache_dtype")
    if kv_cache_dtype in REFITTABLE_FP8_KV_CACHE_DTYPES:
        raise NotImplementedError(
            f"vllm_cfg.kv_cache_dtype={kv_cache_dtype!r} calibrates its KV scales "
            "on the trainer at every refit, which rollout_only has no trainer for."
        )
    if master_config.checkpointing["enabled"]:
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
    if master_config.loss_fn.truncated_importance_sampling_type == "adaptive-ppo-clip":
        raise NotImplementedError(
            "loss_fn.truncated_importance_sampling_type='adaptive-ppo-clip' reads "
            "the step's approx_entropy from the trainer, which the cpu train "
            "backend does not produce."
        )
    refit_timeout_s = master_config.async_rl.generation_fleet_health.refit_timeout_s
    if refit_timeout_s is not None and settings.refit_time_s >= refit_timeout_s:
        raise ValueError(
            f"refit_time_s ({settings.refit_time_s}) must be below "
            "async_rl.generation_fleet_health.refit_timeout_s "
            f"({refit_timeout_s}); otherwise every simulated refit is aborted as "
            "a stuck transfer. Raise the timeout or set it to null."
        )


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
        self.sync_count = 0

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
        self.sync_count += 1
        return None

    @property
    def refit_time_s(self) -> float:
        """Seconds each ``sync_weights`` holds inside the refit pause."""
        return self._refit_time_s

    @property
    def is_stale(self) -> bool:
        """Never: the engine loaded the real checkpoint at startup."""
        return False

    def init_communicator(self) -> None:
        """No transport, so nothing to initialize."""

    def shutdown(self) -> None:
        """No transport, so nothing to release."""


class CpuTrainer(TQDriverMixin):
    """Stand-in for ``TQPolicy`` that touches only the data plane, never a GPU.

    Owns the TransferQueue bootstrap ``TQPolicy`` normally performs, so the
    rest of setup (connect-only clients, partition registration, the replay
    buffer) is unchanged. Every GPU phase transition the SingleController
    drives is a no-op; the logprob stages copy ``generation_logprobs`` into
    the ``prev_logprobs`` / ``reference_policy_logprobs`` columns so the
    advantage stage reads the columns it expects and reports a zero
    generation/training mismatch; the train step accepts its microbatches
    and returns an empty metrics dict.

    What only a real trainer can do -- FP8 KV-scale calibration and
    checkpointing -- raises ``NotImplementedError``. Setup rejects the configs
    that would reach those paths before any engine is built.
    """

    def __init__(
        self,
        *,
        dp_cfg: DataPlaneRuntimeConfig,
        policy_config: PolicyConfig,
        checkpointing: bool = False,
        storage_unit_node_ids: Optional[list[str]] = None,
        dp_client: Optional[DataPlaneClient] = None,
    ) -> None:
        """Bootstrap the data plane (or adopt ``dp_client``) and hold the config.

        Args:
            dp_cfg: ``master_config.data_plane``; bootstraps the TQ controller
                exactly as ``TQPolicy`` does.
            policy_config: ``master_config.policy``; read by the callers that
                expect ``trainer.cfg``.
            checkpointing: TQ bootstrap mode; always False on a simulator run.
            storage_unit_node_ids: Storage-unit plan forwarded to the bootstrap.
            dp_client: An already-built client to use instead of bootstrapping
                one. Tests pass an in-process client here.
        """
        self.cfg = policy_config
        self.dp_cfg = dp_cfg
        if dp_client is None:
            dp_client = build_data_plane_client(
                dp_cfg,
                bootstrap=True,
                checkpointing=checkpointing,
                storage_unit_node_ids=storage_unit_node_ids,
            )
        self.dp_client = dp_client
        self._step_open = False
        self._step_rows = 0
        self.steps_completed = 0

    # ── data-plane lifecycle (what TQPolicy owns) ───────────────────────────

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

    # ── GPU phase transitions: nothing to move ──────────────────────────────

    def prepare_for_training(self) -> None:
        """No-op: no parameters to bring to GPU."""

    def prepare_for_lp_inference(self, *, keep_train_buffers: bool = False) -> None:
        """No-op: the logprob stages below read the data plane only."""
        del keep_train_buffers

    def finish_inference(self) -> None:
        """No-op."""

    def offload_to_cpu(self) -> None:
        """No-op: there is nothing resident on a GPU."""

    def offload_before_refit(self) -> None:
        """No-op."""

    def offload_after_refit(self) -> None:
        """No-op."""

    def pause_train_step_with_offloading(self) -> None:
        """No-op: there are no gradients to park."""

    def sync_params_before_refit(self) -> None:
        """No-op: no deferred all-gathers to materialize."""

    def start_gpu_profiling(self) -> None:
        """No-op: ``maybe_gpu_profile_step`` may address the trainer."""

    def stop_gpu_profiling(self) -> None:
        """No-op."""

    # ── logprob stages: copy the generation column ──────────────────────────

    def get_logprobs_from_meta(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        """Write ``prev_logprobs`` as a copy of ``generation_logprobs``."""
        del micro_batch_size, timer
        self._copy_column(meta, GENERATION_LOGPROBS_FIELD, PREV_LOGPROBS_FIELD)

    def get_reference_policy_logprobs_from_meta(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        """Write ``reference_policy_logprobs`` as a copy of ``generation_logprobs``."""
        del micro_batch_size, timer
        self._copy_column(meta, GENERATION_LOGPROBS_FIELD, REFERENCE_LOGPROBS_FIELD)

    def _copy_column(self, meta: KVBatchMeta, source: str, target: str) -> None:
        """Copy one column under the same sample ids, row layout untouched.

        A direct ``get_samples`` / ``put_samples`` pair rather than the padded
        read/write helpers: the copy then has exactly the per-row shape the
        rollout wrote, which is what the advantage stage compares it against.
        """
        rows = self.dp_client.get_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=[source],
        )
        self.dp_client.put_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=TensorDict({target: rows.get(source)}, batch_size=rows.batch_size),
        )

    # ── train step: accept the batch, return no metrics ─────────────────────

    def begin_train_step(self, loss_fn: Any) -> None:
        """Open a step; the loss is not evaluated."""
        del loss_fn
        if self._step_open:
            raise RuntimeError("CpuTrainer: a train step is already open")
        self._step_open = True
        self._step_rows = 0

    def train_microbatches_from_meta(
        self,
        meta: KVBatchMeta,
        *,
        train_fields: Optional[list[str]] = None,
    ) -> None:
        """Count the rows of a chunk; no forward or backward runs."""
        del train_fields
        if not self._step_open:
            raise RuntimeError("CpuTrainer: train_microbatches_from_meta before begin")
        self._step_rows += meta.size

    def finish_train_step(self) -> dict[str, Any]:
        """Close the step. Returns no metrics: nothing was trained."""
        if not self._step_open:
            raise RuntimeError("CpuTrainer: finish_train_step without an open step")
        self._step_open = False
        self.steps_completed += 1
        return {}

    def abort_train_step(self) -> None:
        """Drop the open step."""
        self._step_open = False
        self._step_rows = 0

    @property
    def step_rows(self) -> int:
        """Rows accepted by the open (or most recently closed) step."""
        return self._step_rows

    # ── only a real trainer can do these; setup refuses the configs that need them

    def calibrate_qkv_fp8_scales(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Raise: KV-scale calibration needs the policy's forward pass."""
        raise NotImplementedError(
            "CpuTrainer cannot calibrate FP8 KV scales; rollout_only rejects "
            "refittable fp8 KV caches at setup"
        )

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Raise: there is no model state to save."""
        raise NotImplementedError("CpuTrainer has no checkpoint to save")

    def finalize_async_save(self, *args: Any, **kwargs: Any) -> None:
        """Raise: see :meth:`save_checkpoint`."""
        raise NotImplementedError("CpuTrainer has no checkpoint to finalize")

    @property
    def sharding_annotations(self) -> Any:
        """Raise: no worker group, so no sharding."""
        raise NotImplementedError("CpuTrainer has no worker group")

    @property
    def worker_group(self) -> Any:
        """Raise: no worker group."""
        raise NotImplementedError("CpuTrainer has no worker group")
