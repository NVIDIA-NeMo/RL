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

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, get_args

from nemo_rl.telemetry.vocabulary import (
    PERFORMANCE_PREFIX,
    TIMING_TRAIN_PREFIX,
    VALIDATION_PREFIX,
    TeedMetric,
    as_scalar,
    register_teed_metrics,
)

#: Prefix every algorithm logs its setup timings under.
SETUP_TIMING_PREFIX = "timing/setup"

#: Single-Controller snapshot-attempt dict names, shared with the OTel tee; not in
#: ``rollout_checkpoint.py``, which imports the GRPO stack.
ROLLOUT_CHECKPOINT_OUTCOME_PREFIX = "rollout/checkpoint_outcome"
ROLLOUT_CHECKPOINT_ATTEMPT_DURATION_KEY = "attempt_duration_seconds"
ROLLOUT_CHECKPOINT_SECONDS_SINCE_LAST_SUCCESS_KEY = "seconds_since_last_success"

RolloutCheckpointAttemptOutcome = Literal["completed", "failed", "skipped"]
ROLLOUT_CHECKPOINT_ATTEMPT_OUTCOMES: tuple[RolloutCheckpointAttemptOutcome, ...] = (
    get_args(RolloutCheckpointAttemptOutcome)
)

RolloutCheckpointAttemptReason = Literal[
    "completed",
    "gym_unavailable",
    "invariant_error",
    "io_error",
    "missing_trainer_anchor",
    "no_data_plane_mutations",
    "optimizer_commit_in_progress",
    "timeout",
    "trainer_state_changed",
]
ROLLOUT_CHECKPOINT_ATTEMPT_REASONS: tuple[RolloutCheckpointAttemptReason, ...] = (
    get_args(RolloutCheckpointAttemptReason)
)


def rollout_checkpoint_reason_key(reason: RolloutCheckpointAttemptReason) -> str:
    """The one-hot flag key an attempt logs its *reason* under."""
    return f"reason_{reason}"


# Keys several training loops log under the "train" prefix. Shared constants
# rather than a literal per call site so a rename reaches every producer and
# the OTel declaration below at once.
REWARD_KEY = "reward"
LOSS_KEY = "loss"
GRAD_NORM_KEY = "grad_norm"
LEARNING_RATE_KEY = "lr"
MEAN_GEN_TOKENS_PER_SAMPLE_KEY = "mean_gen_tokens_per_sample"

#: Teed rows for the keys above, which GRPO, PPO, SFT, OPD and the single
#: controller all build from the same producers.
TRAINING_TEED_METRICS = (
    TeedMetric(REWARD_KEY, "rl.reward.mean", description="Mean rollout reward."),
    TeedMetric(LOSS_KEY, "rl.policy.loss", description="Policy training loss."),
    TeedMetric(GRAD_NORM_KEY, "rl.grad_norm", description="Gradient norm."),
    TeedMetric(
        LEARNING_RATE_KEY, "rl.learning_rate", description="Optimizer learning rate."
    ),
    TeedMetric(
        MEAN_GEN_TOKENS_PER_SAMPLE_KEY,
        "rl.response.length.mean",
        unit="{token}",
        description="Mean generated tokens per sample.",
    ),
)

register_teed_metrics(TRAINING_TEED_METRICS)


# Keys the validation dicts carry, which GRPO, distillation and the single
# controller all build the same way. `accuracy` and `avg_length` are always
# present; `pass_k` only under grouped validation.
VAL_ACCURACY_KEY = "accuracy"
VAL_AVG_LENGTH_KEY = "avg_length"
VAL_PASS_K_KEY = "pass_k"

#: Teed rows for the validation dicts, which reach the Logger under
#: ``validation`` rather than ``train``. Declared with that prefix because
#: ``accuracy`` is not a reserved word: a training dict carrying its own
#: ``accuracy`` must not land on the validation series.
VALIDATION_TEED_METRICS = (
    TeedMetric(
        VAL_ACCURACY_KEY,
        "rl.validation.accuracy",
        description="Validation accuracy (mean env reward over the val set).",
        prefixes=(VALIDATION_PREFIX,),
    ),
    TeedMetric(
        VAL_AVG_LENGTH_KEY,
        "rl.validation.response.length.mean",
        unit="{token}",
        description="Mean validation response length.",
        prefixes=(VALIDATION_PREFIX,),
    ),
    TeedMetric(
        VAL_PASS_K_KEY,
        "rl.validation.pass_k",
        description="Validation pass@k; absent unless validation is grouped.",
        prefixes=(VALIDATION_PREFIX,),
    ),
)

register_teed_metrics(VALIDATION_TEED_METRICS)


# Keys `print_performance_metrics` returns, plus the throughput one the step
# loop injects into its Timer dict. Both arrive under their own prefix.
TOKEN_IMBALANCE_KEY = "average_token_imbalance"
WORKER_IDLE_RATIO_KEY = "training_worker_idle_time_ratio"
TRAIN_FLOPS_PER_GPU_KEY = "train_flops_per_gpu"
TRAIN_FP_UTILIZATION_KEY = "train_fp_utilization"
VALID_TOKENS_PER_SEC_PER_GPU_KEY = "valid_tokens_per_sec_per_gpu"

#: Teed rows for the throughput and utilization numbers. These have no span
#: equivalent -- a span can time a phase but cannot say what fraction of the
#: GPU's arithmetic the phase used -- so without these the question "is this
#: run fast for its hardware" is unanswerable from traces alone.
PERFORMANCE_TEED_METRICS = (
    TeedMetric(
        TRAIN_FP_UTILIZATION_KEY,
        "rl.performance.fp_utilization",
        unit="1",
        description="Model floating-point utilization (MFU) as a 0-1 ratio.",
        prefixes=(PERFORMANCE_PREFIX,),
    ),
    TeedMetric(
        TRAIN_FLOPS_PER_GPU_KEY,
        "rl.performance.tflops_per_gpu",
        unit="TFLOP/s",
        description="Achieved training TFLOP/s per GPU.",
        prefixes=(PERFORMANCE_PREFIX,),
    ),
    TeedMetric(
        WORKER_IDLE_RATIO_KEY,
        "rl.performance.worker_idle_ratio",
        unit="1",
        description="Fraction of the step a training worker spent idle.",
        prefixes=(PERFORMANCE_PREFIX,),
    ),
    TeedMetric(
        TOKEN_IMBALANCE_KEY,
        "rl.performance.token_imbalance",
        description="Mean token-count imbalance across training workers.",
        prefixes=(PERFORMANCE_PREFIX,),
    ),
    TeedMetric(
        VALID_TOKENS_PER_SEC_PER_GPU_KEY,
        "rl.performance.valid_tokens_per_second_per_gpu",
        unit="{token}/s",
        description="Supervised token throughput per GPU for the step.",
        prefixes=(TIMING_TRAIN_PREFIX,),
    ),
)

register_teed_metrics(PERFORMANCE_TEED_METRICS)


@dataclass
class SetupTimingMetrics:
    """Driver-side per-phase timings collected during setup."""

    # Generation-backend init.
    generation_init_time_s: Optional[float] = None
    # When overlapping NeMo Gym init, the total decomposes as reserve + load.
    generation_init_reserve_time_s: Optional[float] = None
    generation_init_load_time_s: Optional[float] = None

    # sft_v2 only: elsewhere the placement wait falls inside policy_init.
    cluster_init_time_s: Optional[float] = None
    policy_init_time_s: Optional[float] = None
    # PPO only: the critic shares the training GPUs, so it is built after the policy.
    value_init_time_s: Optional[float] = None
    nemo_gym_init_time_s: Optional[float] = None
    collective_init_time_s: Optional[float] = None
    # Non-colocated megatron's post-init weight sync into the engine.
    weight_sync_time_s: Optional[float] = None

    # Non-colocated only. (grpo.py only)
    parallel_wall_time_s: Optional[float] = None
    parallel_init_enabled: Optional[float] = None

    # Optional setup phases. OPD teacher timings are shared by legacy GRPO and SC;
    # sparse refit and checkpoint-engine timings remain legacy-GRPO-only.
    teacher_reservation_time_s: Optional[float] = None
    teacher_model_init_time_s: Optional[float] = None
    teacher_init_time_s: Optional[float] = None
    vllm_checkpoint_engine_init_time_s: Optional[float] = None

    total_setup_time_s: Optional[float] = None
    worker_setup_time_s: Optional[float] = None
    other_setup_time_s: Optional[float] = None

    # Overflow bucket for dynamic-keyed metrics (e.g. one entry per active
    # sparse refit transport: vllm_<transport>_sparse_init_time_s).
    extras: dict[str, float] = field(default_factory=dict)

    # How a field name says "this is a duration". Not annotated, so it stays a
    # class attribute rather than becoming a field.
    _DURATION_SUFFIXES = ("_time_s", "_s")

    def to_metrics_dict(self) -> dict[str, Any]:
        """Serialize for Logger.log_metrics; drops unset (None) fields."""
        base = {
            k: v for k, v in asdict(self).items() if k != "extras" and v is not None
        }
        base.update(self.extras)
        return base

    @classmethod
    def phase_name(cls, key: str) -> Optional[str]:
        """Phase a serialized key names, or None when it is not a duration.

        Both suffixes are stripped so the phase reads as ``generation_init``
        rather than ``generation_init_time_s``. Requiring one of them is also
        the filter: it excludes non-durations that ride the same dict, such as
        ``parallel_init_enabled``.
        """
        for suffix in cls._DURATION_SUFFIXES:
            if key.endswith(suffix):
                # Guards against a bare "_s" or "_time_s" key naming nothing.
                return key[: -len(suffix)] or None
        return None

    @classmethod
    def phase_seconds(cls, metrics: dict[str, Any]) -> dict[str, float]:
        """Extract ``{phase: seconds}`` from a serialized timing dict.

        Reads a plain dict rather than an instance because ``extras`` and
        ppo.py's hand-built dict of the same shape have to go through it too.
        """
        seconds: dict[str, float] = {}
        for key, raw in metrics.items():
            phase = cls.phase_name(key)
            if phase is None:
                continue
            value = as_scalar(raw)
            if value is None:
                continue
            seconds[phase] = value
        return seconds


def print_setup_timing_summary(metrics: SetupTimingMetrics) -> None:
    """Print the setup-phase summary block.

    Args:
        metrics: Populated timing metrics.
    """
    print("\n▶ Worker Initialization Timing:")

    assert metrics.generation_init_time_s is not None
    if metrics.generation_init_reserve_time_s:
        # gym-on: an address was reserved, so the total decomposes as reserve + load.
        print(
            f"  Generation init: {metrics.generation_init_time_s:.1f}s"
            f" (reserve {metrics.generation_init_reserve_time_s:.1f}s"
            f" + load {metrics.generation_init_load_time_s:.1f}s)"
        )
    else:
        print(f"  Generation init: {metrics.generation_init_time_s:.1f}s")

    print(f"  Policy init: {metrics.policy_init_time_s:.1f}s")

    if metrics.value_init_time_s:
        print(f"  Value init: {metrics.value_init_time_s:.1f}s")

    if metrics.nemo_gym_init_time_s:
        print(f"  NeMo-Gym init: {metrics.nemo_gym_init_time_s:.1f}s")

    if metrics.teacher_init_time_s:
        print(f"  Teacher init: {metrics.teacher_init_time_s:.1f}s")

    print(f"  Other setup: {metrics.other_setup_time_s:.1f}s")
    print(f"  Total setup: {metrics.total_setup_time_s:.1f}s", flush=True)
