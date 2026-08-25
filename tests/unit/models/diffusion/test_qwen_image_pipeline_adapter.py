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
"""End-to-end logprob parity tests for ``QwenImagePipelineAdapter``.

We deliberately avoid loading the real Qwen-Image checkpoint. The adapter's
``_denoise_step`` accepts a callable ``forward_transformer_fn`` so we can plug
in a deterministic, pure-function "transformer" that returns a tensor with the
same shape as the input ``hidden_states``. This isolates the parity-sensitive
math (transformer forward → CFG → SDE step → log-prob) from any model-loading
concerns.
"""

from typing import Any

import pytest
import torch

from nemo_rl.models.diffusion.pipeline import (
    QwenImagePipelineAdapter,
    apply_true_cfg,
)


class _FakeFlowMatchScheduler:
    """Minimal scheduler that satisfies the contract used by ``sde_step_with_logprob``."""

    def __init__(self) -> None:
        # 4 active steps + a sentinel terminal sigma.
        self.timesteps = torch.tensor([1000.0, 800.0, 600.0, 400.0])
        self.sigmas = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])

    def index_for_timestep(self, t: float | torch.Tensor) -> int:
        t = torch.as_tensor(t)
        matches = torch.nonzero(self.timesteps == t.cpu(), as_tuple=False)
        return int(matches[0].item())


def _make_fake_transformer_fn(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # A simple, deterministic, pure-function "model": noise_pred[b,c,h,w] =
    # sigmoid(W * latents) - 0.5 with a fixed kernel.
    W = torch.randn(1, generator=g).item()

    def forward(transformer: Any, **kwargs: Any) -> torch.Tensor:
        x = kwargs["hidden_states"]
        return torch.sigmoid(W * x) - 0.5

    return forward


def _build_adapter(true_cfg_scale: float = 1.0):
    pipeline_cfg = {
        "height": 16,
        "width": 16,
        "num_inference_steps": 4,
        "true_cfg_scale": true_cfg_scale,
        "max_sequence_length": 4,
    }
    algo_cfg = {
        "noise_level": 0.7,
        "sde_type": "sde",
        "sde_window_size": 4,
        "sde_window_range": [0, 4],
    }
    return QwenImagePipelineAdapter(
        transformer=torch.nn.Identity(),
        scheduler=_FakeFlowMatchScheduler(),
        pipeline_cfg=pipeline_cfg,
        algo_cfg=algo_cfg,
        encode_condition_fn=None,
        decode_fn=None,
        latent_channels=2,
        vae_scale_factor=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        forward_transformer_fn=_make_fake_transformer_fn(),
        per_element_logprob=False,
    )


def test_apply_true_cfg_passthrough_when_scale_one():
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    assert torch.equal(apply_true_cfg(a, b, 1.0), a)


def test_apply_true_cfg_preserves_norm_of_conditional():
    torch.manual_seed(0)
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    combined = apply_true_cfg(a, b, true_cfg_scale=3.0)
    # The last-dim norm of the result should equal the last-dim norm of the
    # conditional branch.
    assert torch.allclose(
        torch.norm(combined, dim=-1), torch.norm(a, dim=-1), atol=1e-5
    )


def test_sample_then_recompute_matches_in_fp32():
    adapter = _build_adapter()
    B, T = 2, 4
    timesteps_global = adapter._scheduler_timesteps()  # [T]

    # Drive sampling manually so we can store all intermediate state, mirroring
    # what `sample_trajectory` would do (we bypass encode_condition because it
    # was not wired into this fake adapter).
    latents_history = [torch.randn(B, 2, 4, 4)]
    sampling_logprobs: list[torch.Tensor] = []
    prompt_embeds = torch.zeros(B, 1, 4)
    prompt_embeds_mask = torch.ones(B, 1)
    neg_embeds = torch.zeros(B, 1, 4)
    neg_mask = torch.ones(B, 1)
    g = torch.Generator().manual_seed(123)

    for step in range(T):
        latents_next, lp = adapter._denoise_step(
            latents=latents_history[-1],
            timestep_value=timesteps_global[step],
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_embeds_mask=neg_mask,
            prev_sample=None,
            stochastic=True,
            generator=g,
        )
        latents_history.append(latents_next)
        sampling_logprobs.append(lp)

    latents_stacked = torch.stack(latents_history, dim=1)  # [B, T+1, ...]
    sampling_logprobs_stacked = torch.stack(sampling_logprobs, dim=1)  # [B, T]
    timesteps_per_sample = timesteps_global.unsqueeze(0).expand(B, -1)

    data: dict[str, Any] = {
        "latents": latents_stacked,
        "timesteps": timesteps_per_sample,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": neg_embeds,
        "negative_prompt_embeds_mask": neg_mask,
        "generation_logprobs": sampling_logprobs_stacked,
        "advantages": torch.zeros(B, T),
        "timestep_mask": torch.ones(B, T),
    }
    curr_lp, _, _, _ = adapter.compute_transition_logprob(data)  # type: ignore[arg-type]

    diff = (curr_lp - sampling_logprobs_stacked).abs().max().item()
    assert diff < 1e-4, f"fp32 logprob drift {diff:.2e} exceeds 1e-4"


def test_recompute_parity_holds_under_cfg():
    adapter = _build_adapter(true_cfg_scale=3.0)
    B, T = 1, 4
    timesteps_global = adapter._scheduler_timesteps()
    latents_history = [torch.randn(B, 2, 4, 4)]
    sampling_logprobs: list[torch.Tensor] = []
    prompt_embeds = torch.randn(B, 1, 4)
    prompt_embeds_mask = torch.ones(B, 1)
    neg_embeds = torch.randn(B, 1, 4)
    neg_mask = torch.ones(B, 1)
    g = torch.Generator().manual_seed(7)

    for step in range(T):
        latents_next, lp = adapter._denoise_step(
            latents=latents_history[-1],
            timestep_value=timesteps_global[step],
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_embeds_mask=neg_mask,
            prev_sample=None,
            stochastic=True,
            generator=g,
        )
        latents_history.append(latents_next)
        sampling_logprobs.append(lp)

    latents_stacked = torch.stack(latents_history, dim=1)
    sampling_logprobs_stacked = torch.stack(sampling_logprobs, dim=1)
    timesteps_per_sample = timesteps_global.unsqueeze(0).expand(B, -1)

    data: dict[str, Any] = {
        "latents": latents_stacked,
        "timesteps": timesteps_per_sample,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": neg_embeds,
        "negative_prompt_embeds_mask": neg_mask,
        "generation_logprobs": sampling_logprobs_stacked,
        "advantages": torch.zeros(B, T),
        "timestep_mask": torch.ones(B, T),
    }
    curr_lp, _, _, _ = adapter.compute_transition_logprob(data)  # type: ignore[arg-type]

    diff = (curr_lp - sampling_logprobs_stacked).abs().max().item()
    assert diff < 1e-4, f"fp32 CFG logprob drift {diff:.2e} exceeds 1e-4"


def test_window_outside_zeros_logprob_in_sample_trajectory_path():
    """SDE-window-aware behaviour: steps outside the active window get 0 logprob."""
    adapter = _build_adapter()
    # Restrict the window to steps [1, 2].
    adapter.algo_cfg = {
        **adapter.algo_cfg,
        "sde_window_size": 2,
        "sde_window_range": [1, 2],
    }

    # Mimic the relevant snippet from sample_trajectory:
    timesteps_global = adapter._scheduler_timesteps()
    T = timesteps_global.shape[0]
    from nemo_rl.models.diffusion.sde import compute_window_mask

    # range=[1,2) with size=2 pins the start at 1 (single legal start), so the
    # sampled window is deterministic regardless of seed.
    mask = compute_window_mask(
        T,
        window_start=adapter._sample_window_start(T, seed=None),
        window_size=adapter._effective_window_size(T),
    )
    expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parity_bf16_within_loose_tolerance(dtype: torch.dtype):
    adapter = _build_adapter()
    adapter.dtype = dtype
    B, T = 1, 4
    timesteps_global = adapter._scheduler_timesteps()
    latents = torch.randn(B, 2, 4, 4, dtype=dtype)
    sampling_logprobs: list[torch.Tensor] = []
    prompt_embeds = torch.zeros(B, 1, 4, dtype=dtype)
    prompt_embeds_mask = torch.ones(B, 1)
    neg_embeds = torch.zeros(B, 1, 4, dtype=dtype)
    neg_mask = torch.ones(B, 1)
    g = torch.Generator().manual_seed(42)

    latents_history = [latents]
    for step in range(T):
        nxt, lp = adapter._denoise_step(
            latents=latents_history[-1],
            timestep_value=timesteps_global[step],
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_embeds_mask=neg_mask,
            prev_sample=None,
            stochastic=True,
            generator=g,
        )
        latents_history.append(nxt)
        sampling_logprobs.append(lp)

    latents_stacked = torch.stack(latents_history, dim=1)
    sampling_logprobs_stacked = torch.stack(sampling_logprobs, dim=1)
    timesteps_per_sample = timesteps_global.unsqueeze(0).expand(B, -1)
    data: dict[str, Any] = {
        "latents": latents_stacked,
        "timesteps": timesteps_per_sample,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt_embeds": neg_embeds,
        "negative_prompt_embeds_mask": neg_mask,
        "generation_logprobs": sampling_logprobs_stacked,
        "advantages": torch.zeros(B, T),
        "timestep_mask": torch.ones(B, T),
    }
    curr_lp, _, _, _ = adapter.compute_transition_logprob(data)  # type: ignore[arg-type]
    tol = 1e-4 if dtype == torch.float32 else 1e-2
    diff = (curr_lp - sampling_logprobs_stacked).abs().max().item()
    assert diff < tol, f"{dtype} logprob drift {diff:.2e} exceeds {tol:.0e}"


def _make_window_adapter(num_steps=8, window_size=2, window_range=(0, 5)):
    pytest.importorskip("diffusers")

    from diffusers import FlowMatchEulerDiscreteScheduler

    from nemo_rl.models.diffusion.pipeline import QwenImagePipelineAdapter

    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(num_steps)
    dummy_embeds = lambda prompts, negs: {
        "prompt_embeds": torch.zeros(len(prompts), 4, 8),
        "prompt_embeds_mask": torch.ones(len(prompts), 4, dtype=torch.long),
        "negative_prompt_embeds": torch.zeros(len(prompts), 4, 8),
        "negative_prompt_embeds_mask": torch.ones(len(prompts), 4, dtype=torch.long),
    }
    return QwenImagePipelineAdapter(
        transformer=torch.nn.Identity(),
        scheduler=scheduler,
        pipeline_cfg={"height": 32, "width": 32, "true_cfg_scale": 1.0},
        algo_cfg={
            "noise_level": 0.7,
            "sde_type": "sde",
            "sde_window_size": window_size,
            "sde_window_range": list(window_range),
        },
        encode_condition_fn=dummy_embeds,
        prepare_initial_latents_fn=lambda b, seed: torch.randn(
            b,
            16,
            4,
            4,
            generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        ),
        device=torch.device("cpu"),
        dtype=torch.float32,
        forward_transformer_fn=lambda transformer, **kw: torch.zeros_like(
            kw["hidden_states"]
        ),
        per_element_logprob=False,
    )


def test_window_start_sampled_within_range_and_deterministic_per_seed():
    adapter = _make_window_adapter()
    starts = {adapter._sample_window_start(8, seed=s) for s in range(64)}
    # range=[0, 5), size=2 gives valid starts {0, 1, 2, 3}; 64 seeds should hit more than one
    assert starts <= {0, 1, 2, 3}
    assert len(starts) > 1
    assert adapter._sample_window_start(8, seed=7) == adapter._sample_window_start(
        8, seed=7
    )


def test_trajectory_carries_timestep_mask_matching_nonzero_logprobs():
    adapter = _make_window_adapter()
    traj = adapter.sample_trajectory(["a"], [" "], [{}], K=2, seed=3)
    mask = traj["timestep_mask"]
    assert mask.shape == traj["generation_logprobs"].shape  # [2, 8]
    assert float(mask.sum(dim=1)[0].item()) == 2.0  # window_size
    # Out-of-window logprobs are zeroed; in-window ones are nonzero.
    assert torch.all(traj["generation_logprobs"][mask == 0] == 0)
    assert torch.all(traj["generation_logprobs"][mask == 1] != 0)


def test_reference_path_yields_zero_kl_when_ref_equals_policy():
    adapter = _make_window_adapter()  # helper shared with the window tests
    traj = adapter.sample_trajectory(["a"], [" "], [{}], K=2, seed=3)
    data = {
        "latents": traj["latents"],
        "timesteps": traj["timesteps"],
        "prompt_embeds": traj["prompt_embeds"],
        "prompt_embeds_mask": traj["prompt_embeds_mask"],
        "negative_prompt_embeds": traj["negative_prompt_embeds"],
        "negative_prompt_embeds_mask": traj["negative_prompt_embeds_mask"],
    }
    # reference forward = the same forward (mimics the initial state where LoRA deltas are zero)
    curr, means, stds, refs = adapter.compute_transition_logprob(
        data,
        use_reference=True,
        reference_forward_fn=lambda **kw: torch.zeros_like(kw["hidden_states"]),
    )
    assert refs is not None and torch.allclose(refs, means)


def test_pipeline_adapter_raises_when_encode_fn_missing():
    adapter = _build_adapter()
    with pytest.raises(RuntimeError, match="encode_condition_fn"):
        adapter.encode_condition(["a"], [" "])


def test_pipeline_adapter_raises_when_decode_fn_missing():
    adapter = _build_adapter()
    with pytest.raises(RuntimeError, match="decode_fn"):
        adapter.decode(torch.zeros(1, 2, 4, 4))
