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
"""Native Megatron worker with a diffusion schedule and postprocessors."""

import math
from functools import partial
from typing import Any

import ray
import torch
from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
from megatron.core import parallel_state
from omegaconf import OmegaConf
from transformers import AutoConfig

from just_grpo.algorithms.block_just_grpo import (
    BlockJustGRPOSchedule,
    select_low_confidence_tokens,
)
from just_grpo.config import JustGRPOConfig
from just_grpo.diffusion.denoising_schedule import aggregate_diffusion_logprobs
from just_grpo.diffusion.diffusion_processors import (
    DiffusionLogprobsPostProcessor,
    DiffusionLossPostProcessor,
    prepare_diffusion_microbatch,
)
from just_grpo.generation.megatron_generation import generate_responses, pack_responses
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.data.interfaces import TokenizerType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.train import model_forward
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


class MegatronDiffusionPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    """Keep the native worker lifecycle; specialize only diffusion execution."""

    def __init__(
        self, config: PolicyConfig, tokenizer: TokenizerType, **kwargs: Any
    ) -> None:
        self.research_config = OmegaConf.create(config.pop("diffusion_config"))
        self.diffusion = JustGRPOConfig.model_validate(
            OmegaConf.to_container(self.research_config.just_grpo, resolve=True)
        )
        # Register the trusted configuration class before Bridge reads the checkpoint.
        hf_config = AutoConfig.from_pretrained(
            config["model_name"], trust_remote_code=True
        )
        if hf_config.model_type != "nemotron_labs_diffusion":
            raise ValueError("JustGRPO requires Nemotron Labs Diffusion")
        if (
            2 * config["max_total_sequence_length"]
            > hf_config.rope_parameters["original_max_position_embeddings"]
        ):
            raise ValueError(
                "Doubled diffusion context exceeds the unscaled position window"
            )
        register_allowed_target_prefix(type(hf_config).__module__ + ".")
        config["megatron_cfg"]["model_overrides"] = {
            "seq_length": config["max_total_sequence_length"],
            "block_size": self.diffusion.schedule.block_size,
        }
        super().__init__(
            config,
            tokenizer,
            prepare_microbatch_fn=partial(
                prepare_diffusion_microbatch,
                mask_token_id=self.diffusion.schedule.mask_token_id,
            ),
            loss_postprocessor_factory=DiffusionLossPostProcessor,
            logprobs_postprocessor_factory=DiffusionLogprobsPostProcessor,
            **kwargs,
        )
        self.generation_index = 0
        self.generation_vocab_size = hf_config.vocab_size

    def _schedule(
        self, data: BatchedDataDict[Any], *, selected: bool = True
    ) -> BlockJustGRPOSchedule:
        data.to("cuda")
        fraction = self.diffusion.training_token_fraction if selected else 1.0
        data["training_token_mask"] = select_low_confidence_tokens(
            data,
            fraction=fraction,
            block_size=self.diffusion.schedule.block_size,
        )
        block = self.diffusion.schedule.block_size
        return BlockJustGRPOSchedule(
            data,
            self.diffusion.schedule,
            pad_token_id=self.tokenizer.pad_token_id,
            padded_width=self.cfg["max_total_sequence_length"],
            selected_positions_per_block=math.ceil(fraction * block)
            if fraction < 1
            else None,
        )

    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: int | None = None,
        require_router_replay: bool = True,
    ) -> BatchedDataDict[Any]:
        # Reference scoring enters this method inside the native reference context.
        schedule = self._schedule(data, selected=not require_router_replay)

        def score(level: BatchedDataDict[Any]) -> torch.Tensor:
            return super(MegatronDiffusionPolicyWorkerImpl, self).get_logprobs(
                data=level,
                micro_batch_size=micro_batch_size,
                require_router_replay=require_router_replay,
            )["logprobs"]

        scores = aggregate_diffusion_logprobs(
            schedule.iter_levels(data),
            score,
            original_shape=data["input_ids"].shape,
            device=data["input_ids"].device,
        )
        return BatchedDataDict(logprobs=scores.cpu())

    def train(
        self,
        *,
        data: BatchedDataDict[Any],
        loss_fn: ClippedPGLossFn,
        eval_mode: bool = False,
        gbs: int | None = None,
        mbs: int | None = None,
        check_dim_skip_keys: Any = None,
    ) -> dict[str, Any]:
        if eval_mode:
            raise ValueError("Use get_logprobs for diffusion scoring")
        schedule = self._schedule(data)
        self.begin_train_step(loss_fn)
        try:
            # Preserve the full response mask in the schedule for conditioning.
            for level in schedule.iter_levels(data):
                self.train_microbatch(level)
            result = self.finish_train_step()
        except Exception:
            self.abort_train_step()
            raise
        # Per-view sample counts must not multiply the actual rollout count.
        result["all_mb_metrics"]["num_valid_samples"] = [
            float((data["sample_mask"] > 0).sum())
        ]
        result["all_mb_metrics"]["global_valid_seqs"] = [
            count / schedule.num_steps
            for count in result["all_mb_metrics"]["global_valid_seqs"]
        ]
        local_tokens = (
            data["training_token_mask"] * data["sample_mask"][:, None]
        ).sum()
        kl = (
            sum(result["all_mb_metrics"]["gen_kl_error"])
            * result["all_mb_metrics"]["global_valid_toks"][0]
            / local_tokens.clamp_min(1)
        )
        torch.distributed.all_reduce(
            kl,
            op=torch.distributed.ReduceOp.MAX,
            group=parallel_state.get_data_parallel_group(),
        )
        if not torch.isfinite(kl) or kl > self.diffusion.max_generation_kl:
            raise RuntimeError(f"Generation/training KL exceeds limit: {float(kl)}")
        return result

    @torch.no_grad()
    def generate(
        self, *, data: BatchedDataDict[Any], greedy: bool = False
    ) -> BatchedDataDict[Any]:
        if (
            self.diffusion.runtime != "megatron"
            or parallel_state.get_tensor_model_parallel_world_size() != 1
        ):
            raise ValueError(
                "In-process diffusion generation requires Megatron runtime with TP=1"
            )
        prompts = [
            row[: int(length)].tolist()
            for row, length in zip(data["input_ids"], data["input_lengths"])
        ]
        sampling = self.diffusion.sampling.model_copy()
        if greedy:
            sampling.temperature = 0.0
        generation = self.cfg["generation"]
        stop = generation["stop_token_ids"]

        def forward(
            input_ids: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            logits = model_forward(
                self.model,
                BatchedDataDict(),
                input_ids,
                position_ids,
                attention_mask=None,
                defer_fp32_logits=True,
            )
            # MCore can pad the vocabulary for tensor parallelism. These
            # additional IDs are not part of the tokenizer's distribution.
            return logits[..., : self.generation_vocab_size]

        responses = generate_responses(
            forward,
            prompts,
            batch_size=self.cfg["logprob_batch_size"],
            device=torch.device("cuda", torch.cuda.current_device()),
            sampling=sampling,
            max_new_tokens=generation["max_new_tokens"],
            max_sequence_length=self.cfg["max_total_sequence_length"],
            mask_token_id=self.diffusion.schedule.mask_token_id,
            stop_token_ids=stop,
            seed=self.research_config.grpo.seed
            + parallel_state.get_data_parallel_rank()
            + self.generation_index * parallel_state.get_data_parallel_world_size(),
        )
        self.generation_index += 1
        return pack_responses(
            prompts,
            responses,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=generation["max_new_tokens"],
            stop_token_ids=stop,
        )


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class MegatronDiffusionPolicyWorker(MegatronDiffusionPolicyWorkerImpl):
    pass
