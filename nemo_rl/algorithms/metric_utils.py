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


@dataclass
class SetupTimingMetrics:
    """Driver-side per-phase timings collected during setup."""

    # grpo.py only: generation-backend init (exactly one populated per run).
    vllm_init_time_s: Optional[float] = None
    sglang_init_time_s: Optional[float] = None
    trtllm_init_time_s: Optional[float] = None
    megatron_generation_init_time_s: Optional[float] = None

    # SC only: generation init (reserve + load).
    generation_init_time_s: Optional[float] = None
    generation_init_reserve_time_s: Optional[float] = None
    generation_init_load_time_s: Optional[float] = None

    policy_init_time_s: Optional[float] = None
    nemo_gym_init_time_s: Optional[float] = None
    collective_init_time_s: Optional[float] = None

    # Non-colocated only. (grpo.py only)
    parallel_wall_time_s: Optional[float] = None
    parallel_init_enabled: Optional[float] = None

    # grpo-only phases (non-colocated OPD teachers, sparse refit, checkpoint-engine).
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

    def to_metrics_dict(self) -> dict[str, Any]:
        """Serialize for Logger.log_metrics; drops unset (None) fields."""
        base = {
            k: v for k, v in asdict(self).items() if k != "extras" and v is not None
        }
        base.update(self.extras)
        return base


def print_setup_timing_summary(
    metrics: SetupTimingMetrics, gen_init_time_key: Optional[str] = None
) -> None:
    """Print the setup-phase summary block.

    Args:
        metrics: Populated timing metrics.
        gen_init_time_key: grpo.py passes the backend-specific field name
            (e.g. "vllm_init_time_s"); SC leaves it None and the summary
            reads generation_init_time_s (+ optional reserve/load split).
    """
    print("\n▶ Worker Initialization Timing:")

    if metrics.generation_init_reserve_time_s:
        # SC + gym-on path
        print(
            f"  Generation init: {metrics.generation_init_time_s:.1f}s"
            f" (reserve {metrics.generation_init_reserve_time_s:.1f}s"
            f" + load {metrics.generation_init_load_time_s:.1f}s)"
        )
    elif gen_init_time_key is None:
        # SC + gym-off path
        assert metrics.generation_init_time_s is not None
        print(f"  Generation init: {metrics.generation_init_time_s:.1f}s")
    else:
        # grpo.py path
        assert metrics.generation_init_time_s is None
        print(f"  Generation init: {getattr(metrics, gen_init_time_key):.1f}s")

    print(f"  Policy init: {metrics.policy_init_time_s:.1f}s")

    if metrics.nemo_gym_init_time_s:
        print(f"  NeMo-Gym init: {metrics.nemo_gym_init_time_s:.1f}s")

    if metrics.teacher_init_time_s:
        print(f"  Teacher init: {metrics.teacher_init_time_s:.1f}s")

    print(f"  Other setup: {metrics.other_setup_time_s:.1f}s")
    print(f"  Total setup: {metrics.total_setup_time_s:.1f}s", flush=True)
