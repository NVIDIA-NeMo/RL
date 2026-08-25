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
"""Qwen-Image pipeline adapter for flow-GRPO.

Owns the per-step denoising math used by both rollout (``sample_trajectory``)
and training-side log-prob recompute (``compute_transition_logprob``). Sharing
``_denoise_step`` between the two paths is a deliberate parity requirement;
verl-omni splits sampling and training into separate adapters, which has been
a recurring source of subtle logprob drift. By construction we cannot diverge.

Scheduler contract (informally): the scheduler must expose
``timesteps`` (1-D iterable), ``sigmas`` (1-D iterable indexed by
``index_for_timestep``), and ``index_for_timestep(t) -> int``. This matches
``diffusers`` >= 0.30 ``FlowMatchEulerDiscreteScheduler``.

CFG aggregation follows verl-omni
``pipelines/qwen_image_flow_grpo/common.py::apply_true_cfg`` — additive guidance
with norm rescale toward the conditional branch.
"""

from typing import Any, Callable

import torch

from nemo_rl.models.diffusion.interfaces import (
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)
from nemo_rl.models.diffusion.sde import (
    compute_window_mask,
    sde_step_with_logprob,
)


def apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Reference: verl-omni `apply_true_cfg`.

    Combines the conditional and unconditional noise predictions with the
    standard CFG formula, then rescales toward the norm of the conditional
    branch (the "true-CFG" trick used by Qwen-Image).
    """
    if true_cfg_scale == 1.0:
        return noise_pred
    combined = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    comb_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (cond_norm / comb_norm.clamp_min(1e-12))


class QwenImagePipelineAdapter:
    """Adapter wiring diffusers Qwen-Image components into flow-GRPO loops.

    The constructor takes already-loaded components rather than a model path so
    the worker (which manages FSDP/PEFT lifecycle) controls instantiation.
    Passing a callable for ``forward_transformer`` allows tests to swap in a
    pure-function mock while keeping the adapter unmodified.
    """

    def __init__(
        self,
        *,
        transformer: torch.nn.Module,
        scheduler: Any,
        # Dict views of DiffusionPipelineCfg / DiffusionAlgoCfg (the config
        # crosses the Ray boundary as model_dump() dicts).
        pipeline_cfg: dict[str, Any],
        algo_cfg: dict[str, Any],
        encode_condition_fn: Callable[[list[str], list[str]], dict[str, torch.Tensor]]
        | None = None,
        decode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prepare_initial_latents_fn: Callable[[int, int | None], torch.Tensor]
        | None = None,
        latent_channels: int = 16,
        vae_scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        forward_transformer_fn: Callable[..., torch.Tensor] | None = None,
        per_element_logprob: bool,
    ) -> None:
        self.transformer = transformer
        self.per_element_logprob = per_element_logprob
        self.scheduler = scheduler
        self.pipeline_cfg = pipeline_cfg
        self.algo_cfg = algo_cfg
        self._encode_condition_fn = encode_condition_fn
        self._decode_fn = decode_fn
        self._prepare_initial_latents_fn = prepare_initial_latents_fn
        self.latent_channels = latent_channels
        self.vae_scale_factor = vae_scale_factor
        # pyrefly: ignore  # read-only
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self._forward_transformer_fn = forward_transformer_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode_condition(
        self, prompts: list[str], negative_prompts: list[str]
    ) -> dict[str, torch.Tensor]:
        if self._encode_condition_fn is None:
            raise RuntimeError(
                "encode_condition_fn was not provided; the worker must inject "
                "a text-encoder callable."
            )
        return self._encode_condition_fn(prompts, negative_prompts)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self._decode_fn is None:
            raise RuntimeError("decode_fn was not provided; cannot VAE-decode")
        return self._decode_fn(latents)

    def sample_trajectory(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        K: int,
        seed: int | None = None,
    ) -> DiffusionTrajectorySpec:
        # Repeat each prompt K times for GRPO group sampling.
        rep_prompts = [p for p in prompts for _ in range(K)]
        rep_negs = [n for n in negative_prompts for _ in range(K)]
        rep_meta = [m for m in metadata for _ in range(K)]

        cond = self.encode_condition(rep_prompts, rep_negs)
        prompt_embeds = cond["prompt_embeds"]
        prompt_embeds_mask = cond["prompt_embeds_mask"]
        negative_prompt_embeds = cond["negative_prompt_embeds"]
        negative_prompt_embeds_mask = cond["negative_prompt_embeds_mask"]

        B = prompt_embeds.shape[0]
        latents = self._prepare_initial_latents(B, seed=seed)

        timesteps = self._scheduler_timesteps()  # [T]
        T = timesteps.shape[0]
        window_mask = compute_window_mask(
            T,
            window_start=self._sample_window_start(T, seed),
            window_size=self._effective_window_size(T),
            device=self.device,
        )  # [T]

        latents_history = [latents]
        logprobs: list[torch.Tensor] = []
        # Derive a distinct seed for the SDE noise: _prepare_initial_latents
        # already consumed a generator seeded with `seed`, and two fresh
        # generators with the same seed produce identical Philox streams —
        # the step-0 variance noise would be a permutation of the initial
        # latents, correlating the transition with its own input.
        generator = (
            torch.Generator(device=self.device).manual_seed(seed + 1)
            if seed is not None
            else None
        )

        for step in range(T):
            inside_window = bool(window_mask[step].item() > 0)
            latents_next, lp = self._denoise_step(
                latents=latents,
                timestep_value=timesteps[step],
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                prev_sample=None,
                stochastic=inside_window,
                generator=generator,
            )
            if not inside_window:
                lp = torch.zeros_like(lp)
            latents_history.append(latents_next)
            logprobs.append(lp)
            latents = latents_next

        # [B, T+1, ...]
        latents_stacked = torch.stack(latents_history, dim=1)
        # [B, T]
        logprobs_stacked = torch.stack(logprobs, dim=1)
        timesteps_stacked = timesteps.unsqueeze(0).expand(B, -1)

        images = self.decode(latents) if self._decode_fn is not None else latents

        spec: DiffusionTrajectorySpec = {
            "prompts": rep_prompts,
            "negative_prompts": rep_negs,
            "metadata": rep_meta,
            "images": images,
            "latents": latents_stacked,
            "timesteps": timesteps_stacked,
            "generation_logprobs": logprobs_stacked,
            "timestep_mask": window_mask.unsqueeze(0).repeat(B, 1),
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
        }
        return spec

    def compute_transition_logprob(
        self,
        data: DiffusionTrainDataSpec,
        *,
        use_reference: bool = False,
        reference_forward_fn: Callable[..., torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        latents = data["latents"]  # [B, T+1, ...]
        # [B, T] raw scheduler timestep values (e.g. 1000.0, 980.0, ...) as
        # stored by sample_trajectory; sde_step_with_logprob maps them back to
        # sigma indices via scheduler.index_for_timestep.
        timesteps = data["timesteps"]
        prompt_embeds = data["prompt_embeds"]
        prompt_embeds_mask = data["prompt_embeds_mask"]
        negative_prompt_embeds = data["negative_prompt_embeds"]
        negative_prompt_embeds_mask = data["negative_prompt_embeds_mask"]

        T = timesteps.shape[1]
        curr_logprobs = []
        current_means = []
        std_devs = []
        reference_means: list[torch.Tensor] = []

        for step in range(T):
            x_t = latents[:, step]
            x_next = latents[:, step + 1]
            timestep_value = timesteps[:, step]

            _, lp, mean, std = self._denoise_step(
                latents=x_t,
                timestep_value=timestep_value,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                prev_sample=x_next,
                stochastic=True,
                generator=None,
                return_mean_std=True,
            )
            curr_logprobs.append(lp)
            current_means.append(mean)
            std_devs.append(std)

            if use_reference:
                if reference_forward_fn is None:
                    raise RuntimeError(
                        "use_reference=True but reference_forward_fn is None"
                    )
                _, _, ref_mean, _ = self._denoise_step(
                    latents=x_t,
                    timestep_value=timestep_value,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                    prev_sample=x_next,
                    stochastic=True,
                    generator=None,
                    return_mean_std=True,
                    forward_override=reference_forward_fn,
                )
                reference_means.append(ref_mean.detach())

        curr = torch.stack(curr_logprobs, dim=1)
        means = torch.stack(current_means, dim=1)
        stds = torch.stack(std_devs, dim=1)
        refs = torch.stack(reference_means, dim=1) if reference_means else None
        return curr, means, stds, refs

    # ------------------------------------------------------------------
    # Internals shared by sampling and recompute
    # ------------------------------------------------------------------
    def _denoise_step(
        self,
        *,
        latents: torch.Tensor,
        timestep_value: torch.Tensor | float,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        prev_sample: torch.Tensor | None,
        stochastic: bool,
        generator: torch.Generator | None,
        return_mean_std: bool = False,
        forward_override: Callable[..., torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        forward = forward_override or self._forward_transformer
        noise_pred = forward(
            hidden_states=latents,
            timestep=self._normalize_timestep(timestep_value),
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
        )
        true_cfg_scale = float(self.pipeline_cfg["true_cfg_scale"])
        if true_cfg_scale > 1.0:
            neg_noise_pred = forward(
                hidden_states=latents,
                timestep=self._normalize_timestep(timestep_value),
                encoder_hidden_states=negative_prompt_embeds,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
            )
            noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

        noise_level = float(self.algo_cfg["noise_level"]) if stochastic else 0.0
        prev, lp, mean, std = sde_step_with_logprob(
            self.scheduler,
            noise_pred,
            timestep_value,
            latents,
            noise_level=noise_level,
            prev_sample=prev_sample,
            generator=generator,
            sde_type=self.algo_cfg["sde_type"],
            reduce_per_sample=not self.per_element_logprob,
        )
        if return_mean_std:
            return prev, lp, mean, std
        return prev, lp

    def _forward_transformer(self, **kwargs: Any) -> torch.Tensor:
        if self._forward_transformer_fn is not None:
            return self._forward_transformer_fn(self.transformer, **kwargs)
        out = self.transformer(return_dict=False, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def normalize_timestep(self, t: torch.Tensor | float) -> torch.Tensor:
        """Normalize a raw scheduler timestep into the transformer's expected range.

        Qwen-Image divides the timestep by 1000.
        """
        return self._normalize_timestep(t)

    def _prepare_initial_latents(
        self, batch_size: int, *, seed: int | None
    ) -> torch.Tensor:
        if self._prepare_initial_latents_fn is not None:
            return self._prepare_initial_latents_fn(batch_size, seed)
        h = self.pipeline_cfg["height"] // self.vae_scale_factor
        w = self.pipeline_cfg["width"] // self.vae_scale_factor
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        return torch.randn(
            (batch_size, self.latent_channels, h, w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    def _scheduler_timesteps(self) -> torch.Tensor:
        return torch.as_tensor(self.scheduler.timesteps, device=self.device)

    def _effective_window_size(self, T: int) -> int | None:
        size = self.algo_cfg.get("sde_window_size")
        return size if size is not None else T

    def _sample_window_start(self, T: int, seed: int | None) -> int:
        """Uniformly sample the SDE window start within `sde_window_range`.

        Matches verl-omni's per-rollout `torch.randint(range[0], range[1]-size+1)`.
        Seeded with `seed + 2`: `seed` feeds the initial latents and `seed + 1`
        the SDE noise, so this stream stays independent of both.
        """
        rng = self.algo_cfg.get("sde_window_range")
        if rng is None:
            return 0
        size = self._effective_window_size(T) or T
        lo = int(rng[0])
        hi = min(int(rng[1]), T) - size
        if hi <= lo:
            return lo
        gen = torch.Generator().manual_seed(seed + 2) if seed is not None else None
        return int(torch.randint(lo, hi + 1, (1,), generator=gen).item())

    @staticmethod
    def _normalize_timestep(t: torch.Tensor | float) -> torch.Tensor:
        # Qwen-Image normalizes timesteps to [0, 1] by dividing by 1000.
        if isinstance(t, torch.Tensor):
            return t.float() / 1000.0
        return torch.tensor([float(t) / 1000.0])
