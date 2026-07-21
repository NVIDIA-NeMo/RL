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
from typing import Any, Callable, NotRequired, Protocol, TypedDict

import torch
from pydantic import BaseModel, Field


class DiffusionPipelineCfg(BaseModel, extra="allow"):
    """Per-rollout pipeline knobs aligned with verl-omni `DiffusionPipelineConfig`."""

    height: int = 512
    width: int = 512
    num_inference_steps: int = 16
    true_cfg_scale: float = 4.0
    max_sequence_length: int = 256
    guidance_scale: float | None = None


class DiffusionAlgoCfg(BaseModel, extra="allow"):
    """SDE rollout algorithm knobs aligned with verl-omni `DiffusionRolloutAlgoConfig`."""

    noise_level: float = 0.7
    sde_type: str = "sde"
    # None lets every denoising step participate in the SDE window.
    sde_window_size: int | None = None
    # [start, end) envelope the active window is sampled from; None = full range.
    sde_window_range: list[int] | None = None


class DiffusionLoraCfg(BaseModel, extra="allow"):
    """LoRA adapter configuration (NeMo Automodel `_peft` stack).

    `rank`/`alpha` map onto Automodel PeftConfig `dim`/`alpha`.
    `target_modules` entries are full-path wildcard patterns anchored to the
    transformer's module FQNs (e.g. '*.attn.to_q', '*.img_mlp.net.0.proj');
    peft-style bare suffixes like 'to_q' match nothing. LoRA weights follow
    the base dtype (bf16).
    """

    enabled: bool = True
    rank: int = 32
    alpha: int = 64
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "*.attn.to_q",
            "*.attn.to_k",
            "*.attn.to_v",
            "*.attn.to_out.0",
        ]
    )
    dropout: float = 0.0
    exclude_modules: list[str] | None = None


class DiffusionOptimizerCfg(BaseModel, extra="allow"):
    """AdamW hyperparameters for the diffusion policy."""

    lr: float = 1.0e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float = 1.0


class DiffusionPolicyConfig(BaseModel, extra="allow"):
    """Top-level configuration for the diffusion policy/worker.

    Defaults live on the fields (config-conventions v2); the exemplar YAMLs
    under `examples/configs/diffusion_grpo_qwen_image*.yaml` document them.
    Workers materialize a `model_dump()` dict view internally and read it
    with plain key access.
    """

    model_name: str
    precision: str = "bfloat16"
    # None trains the whole rollout batch in a single backward pass.
    train_micro_batch_size: int | None = None
    enable_gradient_checkpointing: bool = True
    # Keep per-(latent-element) logprobs instead of reducing per step; pairs
    # with loss_fn.aggregate_logprobs_per_sample (experimental, not verl-omni
    # semantics: verl-omni means over all non-batch dims, matching the default).
    per_element_logprob: bool = False
    # Required for multi-worker DP: all ranks must seed LoRA init identically.
    seed: int | None = None
    optimizer: DiffusionOptimizerCfg = Field(default_factory=DiffusionOptimizerCfg)
    pipeline: DiffusionPipelineCfg = Field(default_factory=DiffusionPipelineCfg)
    algo: DiffusionAlgoCfg = Field(default_factory=DiffusionAlgoCfg)
    lora_cfg: DiffusionLoraCfg = Field(default_factory=DiffusionLoraCfg)


class DiffusionValGenerationCfg(BaseModel, extra="allow"):
    """Validation-time generation overrides.

    Validation always samples with the deterministic ODE (no SDE window, no
    logprob collection).
    """

    num_inference_steps: int = 40


class DiffusionGRPOAlgoConfig(BaseModel, extra="allow"):
    """Top-level diffusion-GRPO training-loop config."""

    num_prompts_per_step: int = 8
    num_generations_per_prompt: int = 16
    max_num_steps: int = 5000
    # 0 disables periodic validation.
    val_period: int = 50
    seed: int = 42
    ppo_epochs: int = 1
    val_at_start: bool = False
    val_at_end: bool = False
    # 0 validates on the full val dataloader.
    max_val_samples: int = 0
    use_leave_one_out_baseline: bool = True
    # Normalize advantages by the whole-batch reward std (verl-omni
    # `global_std`) instead of per-group std, which explodes on
    # near-constant groups under sparse rewards.
    use_global_std: bool = True
    val_generation: DiffusionValGenerationCfg = Field(
        default_factory=DiffusionValGenerationCfg
    )


class DiffusionLossConfig(BaseModel, extra="allow"):
    """Diffusion-GRPO loss knobs aligned with verl-omni `FlowGRPOLoss` config."""

    ratio_clip_min: float = 0.2
    ratio_clip_max: float = 0.2
    # Advantages are clamped to ±adv_clip_max before the ratio computation
    # (verl-omni FlowGRPOLoss semantics); None disables the clamp.
    adv_clip_max: float | None = 5.0
    beta: float = 0.0
    # Default False keeps per-(sample, step) ratio elements, which is the
    # verl-omni/Flow-GRPO formulation (verl-omni's log_prob is [B]: a mean
    # over all non-batch dims, never a sum along T). True is an experimental
    # sum-aggregation over T that inflates the log-ratio scale by
    # ~window_size and is incompatible with 1e-4-scale ratio clips.
    aggregate_logprobs_per_sample: bool = False


class DiffusionDatumSpec(TypedDict):
    prompt: str
    negative_prompt: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
    idx: int
    loss_multiplier: float
    task_name: NotRequired[str]


class DiffusionTrajectorySpec(TypedDict):
    prompts: list[str]
    negative_prompts: list[str]
    metadata: list[dict[str, Any]]
    images: torch.Tensor
    latents: torch.Tensor
    timesteps: torch.Tensor
    generation_logprobs: torch.Tensor
    timestep_mask: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor


class DiffusionTrainDataSpec(TypedDict):
    latents: torch.Tensor
    timesteps: torch.Tensor
    generation_logprobs: torch.Tensor
    advantages: torch.Tensor
    timestep_mask: torch.Tensor
    sample_mask: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_embeds_mask: torch.Tensor
    reference_policy_mean: NotRequired[torch.Tensor]
    current_policy_mean: NotRequired[torch.Tensor]
    std_dev: NotRequired[torch.Tensor]


class DiffusionPipelineAdapter(Protocol):
    def encode_condition(
        self, prompts: list[str], negative_prompts: list[str]
    ) -> dict[str, torch.Tensor]: ...

    def sample_trajectory(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        K: int,
        seed: int | None = None,
    ) -> DiffusionTrajectorySpec: ...

    def compute_transition_logprob(
        self,
        data: DiffusionTrainDataSpec,
        *,
        use_reference: bool = False,
        reference_forward_fn: Callable[..., torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...
