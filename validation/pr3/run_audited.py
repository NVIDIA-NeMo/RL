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


"""Exercise the ordinary streaming recipe with isolated audit and cancellation hooks."""

import asyncio
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import ray

from examples import run_grpo_single_controller as launcher
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.distributed import virtual_cluster
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)

_BASE = SingleControllerActor.__ray_metadata__.modified_class


@ray.remote(num_cpus=1, num_gpus=0)
class AuditedController(_BASE):
    """Record actual streaming boundaries and cancel a partial step when requested."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.audit_dir = Path(os.environ["PR3_AUDIT_DIR"])
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_session = uuid.uuid4().hex
        self.audit_lock = threading.Lock()
        self.release_tail = threading.Event()
        if os.environ.get("PR3_GATE_TAIL") == "1":
            generate = self._rollout_manager.generate_and_push
            admitted = 0

            async def gated_generate(*args: Any, **kwargs: Any) -> Any:
                nonlocal admitted
                admitted += 1
                if admitted > 2:
                    while not self.release_tail.is_set():
                        await asyncio.sleep(0.001)
                return await generate(*args, **kwargs)

            self._rollout_manager.generate_and_push = gated_generate
        self.event("init")

    def event(self, event: str, **values: Any) -> None:
        row = {
            "event": event,
            "session": self.audit_session,
            "step": self._train_steps,
            "version": self._trainer_version,
            "time_ns": time.monotonic_ns(),
            **values,
        }
        with self.audit_lock, (self.audit_dir / "controller.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    async def _train_pump_body(self, lifecycle: Any) -> None:
        original = self._trainer.train_microbatches_from_meta
        dispatched: dict[int, list[int]] = {}

        if os.environ.get("PR3_TWO_CHUNKS_BEFORE_CANCEL") == "1":
            select = self._sampler.select

            async def small_chunks(**kwargs: Any) -> Any:
                kwargs["max_prompt_groups"] = min(kwargs["max_prompt_groups"], 2)
                return await select(**kwargs)

            self._sampler.select = small_chunks

        def dispatch(meta: Any, **kwargs: Any) -> Any:
            self.release_tail.set()
            self.event(
                "dispatch_start",
                ids=meta.sample_ids,
                tags=meta.tags,
                inflight=self._inflight_rollouts,
            )
            result = original(meta, **kwargs)
            sizes = dispatched.setdefault(self._train_steps, [])
            sizes.append(len(meta.sample_ids))
            self.event(
                "dispatch_end", ids=meta.sample_ids, inflight=self._inflight_rollouts
            )
            if (
                os.environ.get("PR3_CANCEL_PARTIAL") == "1"
                and self._train_steps >= 2
                and (
                    len(sizes) >= 2
                    or os.environ.get("PR3_TWO_CHUNKS_BEFORE_CANCEL") != "1"
                )
                and sum(sizes) < self._master_config.policy["train_global_batch_size"]
            ):
                self.event("cancel_partial", ids=meta.sample_ids)
                lifecycle.stop_requested = True
            return result

        self._trainer.train_microbatches_from_meta = dispatch
        abort = self._trainer.abort_train_step

        def abort_step() -> None:
            self.event("abort_start")
            abort()
            self.event("abort_end")

        self._trainer.abort_train_step = abort_step
        await super()._train_pump_body(lifecycle)

    async def _save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        if self._train_steps == 2 and os.environ.get("PR3_WAIT_MIXED") == "1":
            deadline = time.monotonic() + 60
            while not (
                sum(self._buffer.ready_list) >= 2 and self._inflight_rollouts > 0
            ):
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "audit could not establish ready plus in-flight checkpoint work"
                    )
                await asyncio.sleep(0.01)
            self.event(
                "mixed_checkpoint_ready",
                ready=sum(self._buffer.ready_list),
                inflight=self._inflight_rollouts,
            )
        await super()._save_checkpoint(*args, **kwargs)

    async def _save_data_plane_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        replay = kwargs.get("replay_metadata") or {}
        self.event(
            "checkpoint_cut",
            ready=[group["group_id"] for group in replay.get("groups", [])],
            inflight=self._inflight_rollouts,
            unfinished=[
                {
                    "group_id": group.group_id,
                    "target_step": group.target_step,
                    "start_weight_version": group.start_weight_version,
                    "phase": group.phase,
                    "status": group.status,
                    "sealed": group.sealed_generation_indices,
                }
                for group in self._rollout_manager.recovery_ledger.groups()
            ],
        )
        await super()._save_data_plane_checkpoint(*args, **kwargs)

    async def _cleanup_consumed_metas_unlocked(self, cut: Any, metas: Any) -> None:
        self.event(
            "consume", ids=[sample for meta in metas for sample in meta.sample_ids]
        )
        await super()._cleanup_consumed_metas_unlocked(cut, metas)


if __name__ == "__main__":
    ACTOR_ENVIRONMENT_REGISTRY["validation.pr3.audit_policy.AuditedPolicyWorker"] = (
        "/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python"
    )
    virtual_cluster._WORKER_PROCESS_SETUP_HOOK = "validation.pr3.audit_workers.setup"
    launcher.SingleControllerActor = AuditedController
    sys.argv.append(
        "policy.worker_extension_cls_fqn=validation.pr3.audit_policy.AuditedPolicyWorker"
    )
    launcher.main()
