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
"""Configuration types and validation; experiment defaults live in YAML."""

from typing import Literal, Self

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, model_validator


class DiffusionSamplingParams(BaseModel, extra="allow"):
    temperature: float = Field(ge=0)
    block_size: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    selection_policy: Literal["leftmost", "confidence_threshold"]
    threshold: float = Field(gt=0, lt=1)
    returns_reveal_steps: bool
    returns_entropy: bool


class ScheduleConfig(BaseModel, extra="allow"):
    mask_token_id: int
    block_size: int = Field(gt=0)
    reveal_tokens_per_step: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_width(self) -> Self:
        if self.reveal_tokens_per_step > self.block_size:
            raise ValueError("reveal_tokens_per_step cannot exceed block_size")
        return self


class JustGRPOConfig(BaseModel, extra="forbid"):
    """Diffusion-specific extension to the standard NeMo-RL configuration."""

    runtime: Literal["reference_vllm", "megatron", "upstream_vllm", "upstream_sglang"]
    generation_python: str | None
    max_generation_kl: float = Field(ge=0)
    training_token_fraction: float = Field(gt=0, le=1)
    distributed: bool
    schedule: ScheduleConfig
    sampling: DiffusionSamplingParams
    validation_sampling: DiffusionSamplingParams

    @model_validator(mode="after")
    def validate_training_distribution(self) -> Self:
        s, p = self.schedule, self.sampling
        if p.temperature <= 0:
            raise ValueError("Training temperature must be positive")
        if p.block_size != s.block_size or p.selection_policy != "leftmost":
            raise ValueError("Training requires matching blocks and leftmost sampling")
        if s.block_size % s.reveal_tokens_per_step:
            raise ValueError(
                "Reference vLLM needs block_size divisible by reveal width"
            )
        if p.max_steps != s.block_size // s.reveal_tokens_per_step:
            raise ValueError("Generation steps must match the training reveal schedule")
        if self.training_token_fraction < 1 and s.reveal_tokens_per_step != 1:
            raise ValueError("Fast partial views require one token per reveal")
        for params in (p, self.validation_sampling):
            if params.returns_entropy or params.returns_reveal_steps:
                raise ValueError(
                    "Generation does not yet transport entropy/reveal channels"
                )
        if self.runtime not in ("reference_vllm", "megatron"):
            raise ValueError(
                "Upstream generation cannot yet replay leftmost reveals; use megatron or reference_vllm"
            )
        return self


def validate_config(config: "DictConfig") -> JustGRPOConfig:
    """Validate the supported research subset of the upstream GRPO schema."""
    diffusion = JustGRPOConfig.model_validate(
        OmegaConf.to_container(config.just_grpo, resolve=True)
    )
    g, p, loss = config.grpo, config.policy, config.loss_fn
    if loss.force_on_policy_ratio:
        raise ValueError(
            "JustGRPO requires recomputed prev_logprobs; force_on_policy_ratio must be false"
        )
    if (
        min(
            g.max_num_steps,
            g.num_prompts_per_step,
            g.max_val_samples,
            g.val_batch_size,
            p.train_micro_batch_size,
            p.logprob_batch_size,
            p.generation.max_new_tokens,
        )
        <= 0
        or g.num_generations_per_prompt < 2
    ):
        raise ValueError(
            "Require positive batch/step sizes and at least two generations per prompt"
        )
    if (
        p.train_global_batch_size
        != g.num_prompts_per_step * g.num_generations_per_prompt
    ):
        raise ValueError(
            "The research driver requires one global optimizer update per rollout group"
        )
    if p.optimizer.name != "torch.optim.AdamW" or p.optimizer.kwargs.lr <= 0:
        raise ValueError(
            "The research driver supports AdamW with positive learning rate"
        )
    if p.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if (
        p.dtensor_cfg.enabled
        or p.sequence_packing.enabled
        or p.dynamic_batching.enabled
    ):
        raise ValueError(
            "DTensor, packing, and dynamic batching are not supported by this research driver"
        )
    if not p.megatron_cfg.enabled:
        raise ValueError("JustGRPO requires the Megatron training backend")
    m = p.megatron_cfg
    world = config.cluster.num_nodes * config.cluster.gpus_per_node
    if not p.generation.colocated.enabled:
        if not g.async_grpo.enabled or diffusion.runtime != "reference_vllm":
            raise ValueError("Dedicated generation requires async reference vLLM")
        resources = p.generation.colocated.resources
        if config.cluster.num_nodes == 1:
            if resources.num_nodes not in (None, 1):
                raise ValueError("Single-node inference must use one node")
            if (
                resources.gpus_per_node is None
                or not 0 < resources.gpus_per_node < world
            ):
                raise ValueError("Dedicated inference needs a positive GPU subset")
            world -= resources.gpus_per_node
        else:
            if (
                resources.num_nodes is None
                or not 0 < resources.num_nodes < config.cluster.num_nodes
            ):
                raise ValueError("Dedicated inference needs a positive node subset")
            if resources.gpus_per_node != config.cluster.gpus_per_node:
                raise ValueError("Dedicated inference must use complete nodes")
            world -= resources.num_nodes * resources.gpus_per_node
    tp = m.tensor_model_parallel_size
    if not diffusion.distributed or tp < 1 or world % tp:
        raise ValueError(
            "Megatron requires distributed training and a TP size dividing the GPU count"
        )
    if (
        m.pipeline_model_parallel_size != 1
        or m.context_parallel_size != 1
        or m.expert_model_parallel_size != 1
        or m.sequence_parallel
        or m.peft.enabled
        or m.fp8_cfg.enabled
        or m.fp4_cfg.enabled
        or m.model_overrides
        or m.use_fused_linear_logprobs
    ):
        raise ValueError(
            "Megatron JustGRPO supports dense BF16 DP/TP with PP=CP=EP=1, without sequence parallelism, PEFT, quantization, or model overrides"
        )
    if (
        m.distributed_data_parallel_config.overlap_grad_reduce
        or m.distributed_data_parallel_config.overlap_param_gather
    ):
        raise ValueError(
            "Megatron JustGRPO requires synchronous gradient and parameter collectives"
        )
    if m.activation_checkpointing and m.recompute_granularity != "full":
        raise ValueError(
            "Megatron JustGRPO currently requires full activation recomputation"
        )
    if m.optimizer.optimizer != "adam" or m.optimizer.use_precision_aware_optimizer:
        raise ValueError("Megatron JustGRPO requires the standard Adam optimizer")
    if (
        m.optimizer.lr != p.optimizer.kwargs.lr
        or m.optimizer.weight_decay != p.optimizer.kwargs.weight_decay
    ):
        raise ValueError(
            "Megatron and policy optimizer learning rate/weight decay must agree"
        )
    if p.max_total_sequence_length % diffusion.schedule.block_size:
        raise ValueError(
            "Megatron context length must align to the diffusion block size"
        )
    local_batch = p.train_global_batch_size // (world // tp)
    if local_batch % p.train_micro_batch_size or local_batch % p.logprob_batch_size:
        raise ValueError(
            "Megatron requires complete training and logprob microbatches per DP replica"
        )
    if p.precision != "bfloat16" or p.dtensor_cfg.lora_cfg.enabled or p.draft.enabled:
        raise ValueError(
            "The research driver requires a full-parameter bf16-compute policy"
        )
    if g.use_dynamic_sampling or g.overlong_filtering or g.reward_shaping.enabled:
        raise ValueError(
            "Dynamic sampling, overlong filtering, and reward shaping are unsupported"
        )
    if g.async_grpo.enabled:
        if diffusion.runtime == "reference_vllm" and p.generation.colocated.enabled:
            raise ValueError("Async vLLM requires dedicated generation GPUs")
        if g.async_grpo.in_flight_weight_updates:
            raise ValueError("Shared-policy inference must drain before weight updates")
        if g.async_grpo.recompute_kv_cache_after_weight_updates:
            raise ValueError(
                "Uncached diffusion inference has no KV cache to recompute"
            )
        if not loss.use_importance_sampling_correction:
            raise ValueError("Async GRPO requires importance sampling correction")
        if g.reward_scaling.enabled or (config.get("data_plane") or {}).get(
            "enabled", False
        ):
            raise ValueError(
                "Async JustGRPO does not support reward scaling or the data plane"
            )
    if g.max_rollout_turns != 1 or g.val_num_generations_per_prompt != 1:
        raise ValueError(
            "The research driver supports single-turn fixed-step runs with one validation generation per prompt"
        )
    if g.val_period != -1 and g.val_period <= 0:
        raise ValueError("val_period must be -1 (disabled) or a positive step interval")
    if g.adv_estimator.name != "grpo" or g.stop_at_validation_metric is not None:
        raise ValueError(
            "The research driver supports GRPO advantages without early stopping"
        )
    if (
        loss.use_kl_in_reward
        or not loss.token_level_loss
        or loss.sequence_level_importance_ratios
        or loss.positive_example_nll_weight != 0
    ):
        raise ValueError(
            "The research worker requires token-level loss without reward-side KL, sequence ratios, or auxiliary NLL"
        )
    for split in (config.data.train, config.data.validation):
        if split.size <= 0 or split.repeat <= 0:
            raise ValueError("Dataset size and repeat must be positive")
    if config.data.train.size * config.data.train.repeat < g.num_prompts_per_step:
        raise ValueError("Training dataset must contain a full prompt batch")
    if g.max_num_epochs <= 0:
        raise ValueError("max_num_epochs must be positive")
    if config.data.default.prompt_style not in (
        "question",
        "sudoku_answer_tag",
    ):
        raise ValueError("Unsupported Sudoku prompt style")
    if config.checkpointing.enabled:
        raise ValueError("Resumable checkpointing is unsupported")
    world_size = world
    if diffusion.distributed:
        data_parallel_size = world_size // p.megatron_cfg.tensor_model_parallel_size
        if world_size < 1 or g.num_prompts_per_step % data_parallel_size:
            raise ValueError(
                "Distributed runs require a whole prompt group per data-parallel rank"
            )
        if g.max_val_samples < data_parallel_size:
            raise ValueError(
                "Validation needs at least one puzzle per data-parallel rank"
            )
    dataset_name = config.data.train.dataset_name
    if (
        dataset_name != "sudoku6x6"
        or config.data.validation.dataset_name != dataset_name
        or config.data.default.env_name != dataset_name
        or dataset_name not in config.env
    ):
        raise ValueError(
            "Training, validation, and environment must all select sudoku6x6"
        )
    if diffusion.validation_sampling != diffusion.sampling:
        raise ValueError("Validation must use the training sampling settings")
    if g.max_val_samples < g.val_batch_size:
        raise ValueError("max_val_samples must include at least one val_batch_size")
    expected_backend = "megatron" if diffusion.runtime == "megatron" else "vllm"
    if p.generation.backend != expected_backend:
        raise ValueError("policy.generation.backend must match just_grpo.runtime")
    if diffusion.runtime == "megatron":
        if tp != 1:
            raise ValueError(
                "In-process Megatron diffusion generation currently requires TP=1"
            )
        if diffusion.generation_python is not None:
            raise ValueError(
                "Megatron diffusion generation uses the policy interpreter"
            )
        if p.generation.mcore_generation_config.expose_http_server:
            raise ValueError(
                "Research Megatron inference requires HTTP serving disabled"
            )
        if p.generation.refit_transport != "mcore":
            raise ValueError("Shared Megatron inference requires native mcore refit")
        if p.generation.mcore_generation_config.cuda_graph_impl != "none":
            raise ValueError(
                "Uncached Megatron diffusion generation requires cuda_graph_impl=none"
            )
    else:
        worker_name = (
            "ReferenceVllmAsyncWorker"
            if g.async_grpo.enabled
            else "ReferenceVllmWorker"
        )
        if p.generation.worker_extension_cls_fqn != (
            f"just_grpo.generation.reference_vllm.{worker_name}"
        ):
            raise ValueError("Reference vLLM requires its diffusion worker extension")
        params = diffusion.sampling
        if OmegaConf.to_container(
            p.generation.vllm_kwargs.diffusion_config, resolve=True
        ) != {
            "canvas_length": params.block_size,
            "max_denoising_steps": params.max_steps,
            "temperature": params.temperature,
            "selection_policy": params.selection_policy,
            "confidence_threshold": params.threshold,
            "return_entropy": params.returns_entropy,
            "return_reveal_steps": params.returns_reveal_steps,
        }:
            raise ValueError("vLLM diffusion_config must match just_grpo.sampling")
        vllm = p.generation.vllm_cfg
        if (
            vllm.tensor_parallel_size != 1
            or vllm.pipeline_parallel_size != 1
            or vllm.expert_parallel_size != 1
        ):
            raise ValueError("Reference vLLM requires single-GPU engine replicas")
        if bool(vllm.async_engine) != bool(g.async_grpo.enabled):
            raise ValueError("vLLM async_engine must match async GRPO")
        if vllm.get("expose_http_server", False):
            raise ValueError(
                "Reference diffusion generation does not expose HTTP serving"
            )
        if vllm.precision != "bfloat16" or vllm.logprobs_mode != "processed_logprobs":
            raise ValueError(
                "Reference generation requires bf16 and processed at-unmask logprobs"
            )
    if (
        p.generation.top_p != 1
        or p.generation.val_top_p != 1
        or p.generation.top_k not in (None, -1)
        or p.generation.val_top_k not in (None, -1)
        or p.generation.stop_strings is not None
    ):
        raise ValueError(
            "Diffusion generation supports unfiltered sampling with model EOS only"
        )
    for params in (diffusion.sampling, diffusion.validation_sampling):
        if p.generation.max_new_tokens % params.block_size:
            raise ValueError("max_new_tokens must be a multiple of each canvas size")
    if (
        diffusion.sampling.temperature != p.generation.temperature
        or diffusion.validation_sampling.temperature != p.generation.val_temperature
    ):
        raise ValueError("Diffusion temperatures must match policy.generation settings")
    return diffusion
