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

"""Best-effort mirroring of NeMo-RL scalar metrics into nemo-lens.

Kept out of ``nemo_rl.utils.logger`` (which pulls in torch/ray/wandb/etc.) so
the mapping stays importable and testable without the heavy training stack.
``nemo_rl.utils.logger.Logger.log_metrics`` calls :func:`tee_rl_metrics_to_otel`
after its normal fan-out to the file/wandb/mlflow backends.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from nemo_rl.telemetry.setup import get_telemetry

logger = logging.getLogger(__name__)

# Map raw Logger metric keys (under the "train"/"" prefix) to
# ``record_rl_metrics`` gauge fields. The first present candidate key wins.
# Best-effort: unmatched keys and non-scalar values are silently skipped.
_RL_OTEL_METRIC_MAP: dict[str, tuple[str, ...]] = {
    "reward_mean": ("reward", "reward_mean", "mean_reward"),
    "kl_divergence": ("kl", "kl_divergence", "mean_kl"),
    "policy_loss": ("loss", "policy_loss"),
    "value_loss": ("value_loss", "critic_loss"),
    "entropy": ("entropy",),
    "response_length_mean": ("mean_gen_tokens_per_sample", "response_length_mean"),
    "grad_norm": ("grad_norm",),
    "learning_rate": ("lr", "learning_rate"),
    "tokens_per_sec": ("valid_tokens_per_sec_per_gpu", "tokens_per_sec"),
}


def map_rl_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract the ``record_rl_metrics`` kwargs present in a raw metrics dict.

    Pure function (no OTel side effects) so it is trivially unit-testable.
    """
    kwargs: dict[str, float] = {}
    for field, candidates in _RL_OTEL_METRIC_MAP.items():
        for key in candidates:
            value = metrics.get(key)
            # Exclude bools (a subclass of int) and non-numeric values.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            kwargs[field] = float(value)
            break
    return kwargs


def tee_rl_metrics_to_otel(metrics: dict[str, Any], prefix: Optional[str]) -> None:
    """Mirror standard RL scalar metrics into nemo-lens (no-op unless exporting).

    Only the driver's per-step ``train`` metrics are teed. The OTel instruments
    are touched only when telemetry is actively exporting and nemo-lens is
    installed; everything else short-circuits to a no-op.
    """
    if prefix not in ("train", ""):
        return
    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return
    try:
        from nemo.lens.instruments.rl import record_rl_metrics
    except ImportError:
        return

    kwargs = map_rl_metrics(metrics)
    if not kwargs:
        return
    try:
        record_rl_metrics(telemetry.meter, **kwargs)
    except Exception:
        logger.debug("nemo-lens: failed to tee RL metrics", exc_info=True)
