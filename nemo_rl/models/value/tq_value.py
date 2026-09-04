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
"""TQ-mediated Value: meta-driven 1-hop counterpart to Value, mirroring TQPolicy."""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data_plane import DataPlaneConfig, KVBatchMeta, build_data_plane_client
from nemo_rl.data_plane.driver_mixin import TQDriverMixin
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import DP_VALUE_TRAIN_FIELDS, VALUE_SEED_FIELDS
from nemo_rl.models.value.lm_value import Value
from nemo_rl.utils.timer import Timer

_REPLICATED_AXES = ["context_parallel", "tensor_parallel", "pipeline_parallel"]


def _aggregate_train_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble per-rank value-train results into Value.train's return shape."""
    out: dict[str, Any] = {
        "loss": results[0]["global_loss"],
        "grad_norm": results[0]["grad_norm"],
    }
    all_mb_metrics: dict[str, list[Any]] = {}
    for r in results:
        for k, v in r["all_mb_metrics"].items():
            all_mb_metrics.setdefault(k, []).extend(v)
    out["all_mb_metrics"] = all_mb_metrics
    return out


class TQValue(TQDriverMixin, Value):
    """TQ-mediated counterpart to Value, taking an extra dp_cfg.

    Attaches to the TQ controller rather than bootstrapping it: the TQPolicy
    built alongside this critic already did that. Partition lifecycle stays
    with the caller.

    TODO(#2625): the value workers have no split begin/microbatch/finish train
    API yet, so one train_from_meta call is one optimizer step and the
    SingleController requires a PPO step to be a single streaming chunk.
    """

    def __init__(
        self,
        *args: Any,
        dp_cfg: DataPlaneConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        dp_world = self.sharding_annotations.get_axis_size("data_parallel")
        if dp_world <= 0:
            raise ValueError(
                f"TQValue requires data_parallel axis size > 0, got {dp_world}. "
                f"Check cluster config (gpus_per_node * num_nodes) vs. "
                f"TP/PP/CP sizes."
            )
        self.dp_cfg = dp_cfg
        self.dp_client = build_data_plane_client(dp_cfg, bootstrap=False)
        ray.get(
            self.worker_group.run_all_workers_single_data(
                "setup_data_plane", cfg=dp_cfg
            )
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    def shutdown(self) -> bool:  # type: ignore[override]
        """Close the TQ client before shutting down the worker group."""
        try:
            self.dp_client.close()
        except Exception as e:
            warnings.warn(f"Error closing data-plane client: {e}")
        return super().shutdown()

    # ── 1-hop entrypoints (KVBatchMeta in, no re-fan-out) ──────────────────

    def get_values_from_meta(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        """1-hop counterpart to get_values.

        Returns nothing: the per-token prediction lands in TQ under values via
        the worker-side leader write-back, so the GAE stage reads it from there
        rather than through Ray.

        Args:
            meta: Full-step batch metadata consumed by all DP ranks.
            micro_batch_size: Inference micro batch size; None uses the config default.
            timer: Optional timer for nested get_values measurements.
        """
        spa, dba = self._packing_args("logprob_mb_tokens")
        value_meta = self._isolated_meta(
            meta,
            fields=list(VALUE_SEED_FIELDS),
            task_name="value_fwd",
        )
        with timer.time("get_values/shard_meta") if timer else nullcontext():
            metas, _ = shard_meta_for_dp(
                value_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=None,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )
        with timer.time("get_values/submit_value_futures") if timer else nullcontext():
            futures = self.worker_group.run_all_workers_sharded_data(
                "get_values_presharded",
                meta=metas,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=_REPLICATED_AXES,
                output_is_replicated=_REPLICATED_AXES,
                common_kwargs={"micro_batch_size": micro_batch_size},
            )
        # Wait for completion; per-rank returns are None.
        self.worker_group.get_all_worker_results(futures)

    def train_from_meta(
        self,
        meta: KVBatchMeta,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> dict[str, Any]:
        """1-hop counterpart to train. One call is one optimizer step.

        Args:
            meta: Full-step batch metadata consumed by all DP ranks.
            loss_fn: Value loss; MseValueLossFn in the PPO path.
            eval_mode: Run forward only, without an optimizer step.
            gbs: Global batch size; defaults to the config's train_global_batch_size.
            mbs: Micro batch size; defaults to the config's train_micro_batch_size.
            timer: Optional timer for nested value_training measurements.

        Returns:
            Aggregated training-step output dict.
        """
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]

        spa, dba = self._packing_args("train_mb_tokens")
        train_meta = self._isolated_meta(
            meta,
            fields=list(DP_VALUE_TRAIN_FIELDS),
            task_name="value_train",
        )
        with timer.time("value_training/shard_meta") if timer else nullcontext():
            dp_metas, _ = shard_meta_for_dp(
                train_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=batch_size,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )

        with (
            timer.time("value_training/submit_training_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "train_presharded",
                meta=dp_metas,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=_REPLICATED_AXES,
                output_is_replicated=_REPLICATED_AXES,
                common_kwargs={
                    "loss_fn": loss_fn,
                    "eval_mode": eval_mode,
                    "gbs": batch_size,
                    "mbs": micro_batch_size,
                },
            )
        return _aggregate_train_results(
            self.worker_group.get_all_worker_results(futures)
        )
