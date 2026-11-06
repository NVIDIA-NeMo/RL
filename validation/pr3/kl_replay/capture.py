# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Capture one real SC update for read-only, same-weight worker replay.

This entrypoint is used only by the diagnostic jobs. It neither changes the
ordinary worker implementation nor writes to the source convergence checkpoint.
Capture timestamps include CPU/I/O overhead and are not a throughput benchmark.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import ray
import torch

from examples import run_grpo_single_controller as launcher
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data_plane import materialize
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS
from nemo_rl.utils.checkpoint import CheckpointManager

BASE = SingleControllerActor.__ray_metadata__.modified_class


@ray.remote(
    num_cpus=1,
    num_gpus=0,
    runtime_env={"env_vars": {"PR3_KL_DIR": os.environ["PR3_KL_DIR"]}},
)
class CaptureController(BASE):  # pragma: no cover
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.directory = Path(os.environ["PR3_KL_DIR"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capture_index = 0
        generate = self._rollout_manager.generate_and_push

        async def record_generation(*values: Any, **named: Any) -> Any:
            group = named.get("lineage_group_id")
            self.event("generation_start", group=group)
            try:
                return await generate(*values, **named)
            finally:
                self.event("generation_end", group=group)

        self._rollout_manager.generate_and_push = record_generation
        log = self._logger.log_metrics

        def record_metrics(metrics: dict[str, Any], *values: Any, **named: Any) -> Any:
            prefix = named.get("prefix", "")
            if prefix in {"train", "timing/train"}:
                self.event(
                    "metrics",
                    prefix=prefix,
                    metrics={
                        key: float(value)
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                )
            return log(metrics, *values, **named)

        self._logger.log_metrics = record_metrics

    def event(self, event: str, **values: Any) -> None:
        with (self.directory / "controller.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": event,
                        "step": self._train_steps,
                        "time_ns": time.time_ns(),
                        **values,
                    },
                    default=str,
                )
                + "\n"
            )

    async def _capture_meta(self, updated: KVBatchMeta) -> None:
        versions = {tag["weight_version"] for tag in updated.tags}
        if len(versions) != 1:
            raise ValueError(
                f"LA0 capture requires one rollout version, got {versions}"
            )
        version = versions.pop()
        directory = self.directory / f"step_{version}"
        directory.mkdir(parents=True, exist_ok=True)
        # LP workers delta-write columns without expanding the caller's meta.
        # Match TQPolicy's train fetch contract rather than meta.fields here.
        fields = [
            name
            for name in DP_TRAIN_FIELDS
            if name != "reference_policy_logprobs" or self._reference_logprobs_required
        ]
        wire = await self._call_dp(
            "get_samples",
            sample_ids=updated.sample_ids,
            partition_id=updated.partition_id,
            select_fields=fields,
        )
        batch = materialize(wire, layout="padded", tags=updated.tags)
        payload = {
            "ids": updated.sample_ids,
            "tags": updated.tags,
            "data": {
                key: value.detach().cpu()
                for key, value in batch.items()
                if torch.is_tensor(value)
            },
            "step_before_update": version,
            "trainer_version": version,
            "capture_point": "after_backward_before_tq_cleanup",
            "extra_info": updated.extra_info,
        }
        torch.save(payload, directory / f"chunk-{self.capture_index:03d}.pt")
        self.capture_index += 1
        self.event("captured", ids=updated.sample_ids, trainer_version=version)

    async def _cleanup_consumed_metas_unlocked(
        self, cut: Any, metas: list[KVBatchMeta]
    ) -> None:
        # Capture only after all backward work has finished. Fetching/writing
        # tensors before training would perturb the very overlap being tested.
        start = time.perf_counter()
        for meta in metas:
            await self._capture_meta(meta)
        self.event("capture_complete", seconds=time.perf_counter() - start)
        await super()._cleanup_consumed_metas_unlocked(cut, metas)

    async def _train_pump_body(self, lifecycle: Any) -> None:
        train = self._trainer.train_microbatches_from_meta

        def record_training(meta: KVBatchMeta, **kwargs: Any) -> Any:
            self.event("training_start", ids=meta.sample_ids)
            try:
                return train(meta, **kwargs)
            finally:
                self.event("training_end", ids=meta.sample_ids)

        self._trainer.train_microbatches_from_meta = record_training
        await super()._train_pump_body(lifecycle)


if __name__ == "__main__":
    if os.environ["USER"] != "root" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Capture must run inside an allocated container")
    checkpoint = os.environ.get("PR3_KL_CHECKPOINT")
    if checkpoint:
        source = Path(checkpoint).resolve()
        if not (source / "policy/optimizer").is_dir():
            raise FileNotFoundError(source)
        CheckpointManager.get_latest_checkpoint_path = lambda self: str(source)
    if os.environ.get("PR3_KL_PROFILE") == "1":
        # Only this diagnostic job inherits these tracing flags. Worker classes
        # stay in their existing isolated AutoModel/vLLM environments.
        from nemo_rl.distributed import virtual_cluster
        from nemo_rl.distributed.ray_actor_environment_registry import (
            ACTOR_ENVIRONMENT_REGISTRY,
        )

        worker_name = "validation.pr3.kl_replay.policy_profile.ProfilePolicyWorker"
        ACTOR_ENVIRONMENT_REGISTRY[worker_name] = (
            "/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python"
        )
        virtual_cluster._WORKER_PROCESS_SETUP_HOOK = (
            "validation.pr3.kl_replay.worker_setup.setup"
        )
        sys.argv.append(f"policy.worker_extension_cls_fqn={worker_name}")
    launcher.SingleControllerActor = CaptureController
    launcher.main()
