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
from typing import Any, Optional

from nemo_rl.telemetry.vocabulary import (
    TeedMetric,
    as_scalar,
    register_teed_metrics,
)

#: Prefix every algorithm logs its setup timings under.
SETUP_TIMING_PREFIX = "timing/setup"

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


@dataclass
class SetupTimingMetrics:
    """Driver-side per-phase timings collected during setup."""

    # Generation-backend init.
    generation_init_time_s: Optional[float] = None
    # When overlapping NeMo Gym init, the total decomposes as reserve + load.
    generation_init_reserve_time_s: Optional[float] = None
    generation_init_load_time_s: Optional[float] = None

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
