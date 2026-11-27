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

"""Opt-in two-GPU regression for the first SingleController Megatron LoRA refit.

Run from the repository root with two visible GPUs:
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    uv run --locked -m tests.functional._sc_lora_refit_probe

The trace records fixed-token scores before training, with the new weights
withheld after the optimizer step, and after the ordinary merged refit.
"""

import asyncio
import math
import sys
import uuid
from typing import Any

import ray
import torch

from examples import run_grpo_single_controller
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.models.generation.interfaces import GenerationDatumSpec

SCORING_WORKER = "tests.functional._sc_lora_refit_vllm.ScoringVllmWorker"
ASYNC_WORKER = (
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
)
PROMPT = "Solve this problem step by step. A shop sells 4 notebooks for 12 dollars."
SUFFIX = (
    " Each notebook costs 12 divided by 4, which is 3 dollars."
    " Therefore, seven notebooks cost 7 times 3, or 21 dollars."
)


def _rms(scores: torch.Tensor) -> float:
    return scores.square().mean().sqrt().item()


@ray.remote(num_cpus=1, num_gpus=0)
class _RefitProbeController(SingleControllerActor.__ray_metadata__.modified_class):
    async def _run_pumps(self) -> dict[str, Any]:
        self._probe_passed = False
        self._scoring_worker_id = None
        assert self._rollout_trace.enabled, "The probe requires the rollout trace"
        result = await super()._run_pumps()
        assert self._probe_passed, "Did not observe an optimizer step and merged refit"
        return result

    async def _rollout_pump(self) -> None:
        # The inherited run path has completed the initial sync before either
        # pump starts. Training cannot start before this pump supplies groups.
        tokenizer = get_tokenizer(self._master_config.policy["tokenizer"])
        prompt = tokenizer.encode(PROMPT, add_special_tokens=False)
        suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
        self._token_ids = prompt + suffix
        self._suffix_start = len(prompt)
        (self._scoring_worker,) = self._gen.worker_group.workers
        self._actor_old, actor_repeat = await asyncio.to_thread(
            self._actor_scores, repeats=2
        )
        self._engine_old = await self._engine_scores("before")
        engine_repeat = await self._engine_scores("before_repeat")
        actor_noise = _rms(actor_repeat - self._actor_old)
        engine_noise = _rms(engine_repeat - self._engine_old)
        assert math.isfinite(actor_noise) and math.isfinite(engine_noise)
        self._noise = max(actor_noise, engine_noise, 1e-6)
        self._rollout_trace.emit(
            "lora_probe_baseline",
            token_ids=self._token_ids,
            suffix_start=self._suffix_start,
            actor_scores=self._actor_old.tolist(),
            actor_repeat=actor_repeat.tolist(),
            actor_repeat_error=actor_noise,
            engine_repeat_error=engine_noise,
            repeat_noise=self._noise,
        )
        await super()._rollout_pump()

    def _actor_scores(self, *, repeats: int) -> list[torch.Tensor]:
        data = BatchedDataDict[GenerationDatumSpec](
            input_ids=torch.tensor([self._token_ids]),
            input_lengths=torch.tensor([len(self._token_ids)]),
        )
        self._trainer.prepare_for_lp_inference(keep_train_buffers=True)
        return [
            self._trainer.get_logprobs(data)["logprobs"][
                0, self._suffix_start :
            ].double()
            for _ in range(repeats)
        ]

    async def _engine_scores(self, phase: str) -> torch.Tensor:
        request_id = f"sc-lora-{phase}-{uuid.uuid4().hex}"
        self._rollout_trace.emit(
            "lora_probe_submit", phase=phase, request_id=request_id
        )
        worker_id, scores = await self._scoring_worker.score_fixed_suffix.remote(
            self._token_ids, suffix_start=self._suffix_start, request_id=request_id
        )
        if self._scoring_worker_id is None:
            self._scoring_worker_id = worker_id
        assert worker_id == self._scoring_worker_id, "Scoring switched rollout workers"
        self._rollout_trace.emit(
            "lora_probe_result",
            phase=phase,
            request_id=request_id,
            worker_id=worker_id,
            scores=scores,
        )
        return torch.tensor(scores, dtype=torch.float64)

    async def _sync_weights(
        self, *, calibration_data: BatchedDataDict[Any] | None = None
    ) -> int:
        if self._trainer_version == 0:
            return await super()._sync_weights(calibration_data=calibration_data)

        assert self._trainer_version == self._train_steps == 1
        assert not self._inflight_by_group_id, "Probe must not span an active rollout"
        withheld = await self._engine_scores("withheld")
        aborted = await super()._sync_weights(calibration_data=calibration_data)
        # The inherited sync emitted weights_synced. No await has handed the
        # event loop back to dispatch since that method reopened the gate.
        self._rollout_permitted.clear()
        try:
            updated = await self._engine_scores("after_refit")
            (actor_new,) = await asyncio.to_thread(self._actor_scores, repeats=1)
            self._check_scores(actor_new=actor_new, withheld=withheld, updated=updated)
            self._probe_passed = True
        finally:
            self._rollout_permitted.set()
        return aborted

    def _check_scores(
        self, *, actor_new: torch.Tensor, withheld: torch.Tensor, updated: torch.Tensor
    ) -> None:
        for scores in (actor_new, self._actor_old, self._engine_old, withheld, updated):
            assert scores.shape == (len(self._token_ids) - self._suffix_start,)
            assert torch.isfinite(scores).all(), "Non-finite fixed suffix logprobs"
        actor_delta = actor_new - self._actor_old
        signal = _rms(actor_delta)
        withheld_change = _rms(withheld - self._engine_old)
        # Compare changes to remove the fixed Megatron/vLLM BF16 kernel offset.
        withheld_error = _rms(withheld - self._engine_old - actor_delta)
        updated_error = _rms(updated - self._engine_old - actor_delta)
        post_gap = _rms(updated - actor_new)
        base_gap = _rms(self._engine_old - self._actor_old)
        self._rollout_trace.emit(
            "lora_probe_comparison",
            actor_scores=actor_new.tolist(),
            actor_delta=signal,
            repeat_noise=self._noise,
            withheld_change=withheld_change,
            withheld_delta_error=withheld_error,
            refit_delta_error=updated_error,
            base_cross_backend_gap=base_gap,
            refit_cross_backend_gap=post_gap,
        )
        assert signal > 8 * self._noise, "Optimizer change not resolved above noise"
        assert withheld_change <= 3 * self._noise, "Withheld control did not stay old"
        assert withheld_error >= signal / 2, "Freshness gate accepted stale weights"
        assert updated_error < signal / 2, "Refit did not track the updated actor"
        assert post_gap <= max(2 * base_gap, 0.02), (
            "Actor/vLLM scores disagree after refit"
        )


def main() -> None:
    if "--config" in sys.argv[1:]:
        raise ValueError("The probe uses the standard Megatron SingleController config")
    ACTOR_ENVIRONMENT_REGISTRY[SCORING_WORKER] = ACTOR_ENVIRONMENT_REGISTRY[
        ASYNC_WORKER
    ]
    run_grpo_single_controller.SingleControllerActor = _RefitProbeController
    sys.argv[1:1] = [
        "policy.model_name=Qwen/Qwen2.5-0.5B",
        "policy.precision=bfloat16",
        "policy.megatron_cfg.peft.enabled=true",
        "policy.megatron_cfg.peft.dim=32",
        # A resolved signal after exactly one real update, despite BF16 rounding.
        "policy.megatron_cfg.optimizer.lr=0.0005",
        "policy.megatron_cfg.scheduler.lr_warmup_iters=0",
        "policy.train_global_batch_size=32",
        "policy.train_micro_batch_size=1",
        "policy.logprob_batch_size=32",
        "policy.generation.lora_refit_mode=merged",
        "++policy.generation.vllm_cfg.enable_prefix_caching=false",
        "policy.generation.vllm_cfg.logprobs_mode=raw_logprobs",
        f"policy.generation.worker_extension_cls_fqn={SCORING_WORKER}",
        "cluster.gpus_per_node=2",
        "grpo.num_prompts_per_step=8",
        "grpo.num_generations_per_prompt=4",
        "grpo.max_num_steps=1",
        "data.shuffle=false",
        "async_rl.sampler.max_lookahead_versions=0",
        "rollout_checkpointing.trace=true",
        "logger.log_dir=logs/sc-lora-refit-probe",
    ]
    run_grpo_single_controller.main()


if __name__ == "__main__":
    main()
