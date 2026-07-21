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
"""Tests for diffusion-GRPO resume helpers."""

import os

from nemo_rl.algorithms.diffusion_grpo import _latest_checkpoint


def _make_ckpt(root, name: str, complete: bool = True) -> str:
    """Mimic the Automodel Checkpointer layout: model/ + optim/.metadata last."""
    path = os.path.join(root, name)
    os.makedirs(os.path.join(path, "model"))
    if complete:
        os.makedirs(os.path.join(path, "optim"))
        with open(os.path.join(path, "optim", ".metadata"), "wb"):
            pass
    return path


def test_latest_checkpoint_missing_dir_returns_none(tmp_path):
    assert _latest_checkpoint(str(tmp_path / "does_not_exist")) is None


def test_latest_checkpoint_picks_highest_complete_step(tmp_path):
    _make_ckpt(tmp_path, "step_1")
    expected = _make_ckpt(tmp_path, "step_10")
    # Incomplete checkpoint (no optim/.metadata yet) must be skipped even
    # though its step number is the highest.
    _make_ckpt(tmp_path, "step_30", complete=False)
    _make_ckpt(tmp_path, "not_a_checkpoint")
    assert _latest_checkpoint(str(tmp_path)) == (expected, 10)


def test_latest_checkpoint_all_incomplete_returns_none(tmp_path):
    _make_ckpt(tmp_path, "step_5", complete=False)
    assert _latest_checkpoint(str(tmp_path)) is None


def test_master_config_rejects_kl_with_full_param():
    import pytest
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.diffusion_grpo import DiffusionMasterConfig
    from nemo_rl.utils.config import load_config

    cfg = OmegaConf.to_container(
        load_config("examples/configs/diffusion_grpo_qwen_image_ocr.yaml"),
        resolve=True,
    )
    cfg["loss_fn"]["beta"] = 0.04
    cfg["policy"]["lora_cfg"]["enabled"] = False
    with pytest.raises(Exception, match="beta"):
        DiffusionMasterConfig.model_validate(cfg)


def test_run_validation_passes_generation_overrides():
    import torch

    from nemo_rl.algorithms.diffusion_grpo import _run_validation

    seen = []

    class FakePolicy:
        def sample_trajectory(
            self,
            prompts,
            negative_prompts,
            metadata,
            *,
            K,
            seed,
            generation_overrides=None,
        ):
            seen.append(generation_overrides)
            B = len(prompts) * K
            return {
                "prompts": prompts * K,
                "negative_prompts": negative_prompts * K,
                "metadata": metadata * K,
                "images": torch.zeros(B, 3, 4, 4),
                "latents": torch.zeros(B, 2, 4),
                "timesteps": torch.zeros(B, 1),
                "generation_logprobs": torch.zeros(B, 1),
                "timestep_mask": torch.zeros(B, 1),
                "prompt_embeds": torch.zeros(B, 1, 1),
                "prompt_embeds_mask": torch.ones(B, 1),
                "negative_prompt_embeds": torch.zeros(B, 1, 1),
                "negative_prompt_embeds_mask": torch.ones(B, 1),
            }

    class FakeEnv:
        def score_images(self, images, prompts, metadata):
            return torch.zeros(images.shape[0]), {}

    class FakeLogger:
        def log_metrics(self, *a, **k):
            pass

    _run_validation(
        FakePolicy(),
        FakeEnv(),
        [{"prompts": ["p"], "negative_prompts": [" "], "metadata": [{}]}],
        step=0,
        logger=FakeLogger(),
        seed=42,
        generation_overrides={"num_inference_steps": 40},
    )
    assert seen == [{"num_inference_steps": 40}]


def test_build_train_data_slices_to_window_columns():
    import torch

    from nemo_rl.algorithms.diffusion_grpo import _build_train_data

    B, T, w = 2, 8, 3
    mask = torch.zeros(B, T)
    mask[0, 1 : 1 + w] = 1.0  # sample 0 window [1, 4)
    mask[1, 4 : 4 + w] = 1.0  # sample 1 window [4, 7)
    traj = {
        "latents": torch.arange(B * (T + 1) * 4, dtype=torch.float32).reshape(
            B, T + 1, 4
        ),
        "timesteps": torch.arange(T, dtype=torch.float32).repeat(B, 1),
        "generation_logprobs": torch.randn(B, T) * mask,
        "timestep_mask": mask,
        "prompt_embeds": torch.zeros(B, 4, 8),
        "prompt_embeds_mask": torch.ones(B, 4),
        "negative_prompt_embeds": torch.zeros(B, 4, 8),
        "negative_prompt_embeds_mask": torch.ones(B, 4),
        "prompts": ["a", "b"],
        "negative_prompts": [" ", " "],
        "metadata": [{}, {}],
        "images": torch.zeros(B, 3, 4, 4),
    }
    out = _build_train_data(
        traj, torch.tensor([1.0, -1.0]), loss_multiplier=torch.ones(B)
    )
    assert out["timesteps"].shape == (B, w)
    assert out["latents"].shape == (B, w + 1, 4)
    assert torch.all(out["timestep_mask"] == 1)
    # Sample 1's window starts at 4, so the sliced timesteps must be [4, 5, 6].
    assert out["timesteps"][1].tolist() == [4.0, 5.0, 6.0]
    # Latents keep one extra column (w + 1): sample 1 must equal the original latents[1, 4:8].
    assert torch.equal(out["latents"][1], traj["latents"][1, 4:8])


def _mini_batch_algo_cfg(**overrides):
    from nemo_rl.models.diffusion.interfaces import DiffusionGRPOAlgoConfig

    cfg = {
        "num_prompts_per_step": 4,
        "num_generations_per_prompt": 4,
        "max_num_steps": 1,
        "val_period": 0,
        "ppo_epochs": 1,
        "val_at_start": False,
        "val_at_end": False,
    }
    cfg.update(overrides)
    return DiffusionGRPOAlgoConfig.model_validate(cfg)


def test_ppo_mini_batch_size_must_keep_groups_whole():
    import pytest

    with pytest.raises(Exception, match="num_generations_per_prompt"):
        _mini_batch_algo_cfg(ppo_mini_batch_size=6)  # not a multiple of K=4


def test_ppo_mini_batch_size_must_divide_rollout_batch():
    import pytest

    with pytest.raises(Exception, match="divide"):
        _mini_batch_algo_cfg(ppo_mini_batch_size=12)  # 16 % 12 != 0


def test_ppo_mini_batch_size_valid_values_accepted():
    assert _mini_batch_algo_cfg(ppo_mini_batch_size=8).ppo_mini_batch_size == 8
    assert _mini_batch_algo_cfg().ppo_mini_batch_size is None


class _RecordingPolicy:
    """Fake DiffusionPolicy capturing each train() call's sample ids."""

    num_workers = 1

    def __init__(self, K: int):
        self.K = K
        self.train_calls: list[list[int]] = []

    def sample_trajectory(
        self, prompts, negative_prompts, metadata, *, K, seed, generation_overrides=None
    ):
        import torch

        # Mirror the real pipeline layout: each prompt's K generations are
        # contiguous (repeat_interleave).
        rep_prompts = [p for p in prompts for _ in range(K)]
        B, T = len(rep_prompts), 2
        latents = torch.zeros(B, T + 1, 4)
        # Stamp the global sample index into the latents so train() calls can
        # be checked for order and group integrity.
        latents[:, 0, 0] = torch.arange(B, dtype=torch.float32)
        return {
            "prompts": rep_prompts,
            "negative_prompts": [n for n in negative_prompts for _ in range(K)],
            "metadata": [m for m in metadata for _ in range(K)],
            "images": torch.zeros(B, 3, 4, 4),
            "latents": latents,
            "timesteps": torch.zeros(B, T),
            "generation_logprobs": torch.zeros(B, T),
            "timestep_mask": torch.ones(B, T),
            "prompt_embeds": torch.zeros(B, 1, 1),
            "prompt_embeds_mask": torch.ones(B, 1),
            "negative_prompt_embeds": torch.zeros(B, 1, 1),
            "negative_prompt_embeds_mask": torch.ones(B, 1),
        }

    def train(self, data, loss_cfg):
        ids = [int(v) for v in data["latents"][:, 0, 0].tolist()]
        self.train_calls.append(ids)
        # Distinct loss per call so the cross-mini mean is observable.
        return {"loss": float(len(self.train_calls)), "mean_ratio": 1.0}

    def save_checkpoint(self, path):
        raise AssertionError("checkpointing is disabled in this test")


def _run_one_train_step(algo_cfg):
    import torch

    from nemo_rl.algorithms.diffusion_grpo import diffusion_grpo_train
    from nemo_rl.models.diffusion.interfaces import DiffusionLossConfig

    K = algo_cfg.num_generations_per_prompt
    policy = _RecordingPolicy(K)

    class FakeEnv:
        def score_images(self, images, prompts, metadata):
            # Varying rewards so the advantage path sees non-constant groups.
            return torch.arange(images.shape[0], dtype=torch.float32), {}

    logged = []

    class FakeLogger:
        def log_metrics(self, metrics, step):
            logged.append(metrics)

    diffusion_grpo_train(
        policy,
        FakeEnv(),
        [
            {
                "prompts": ["a", "b", "c", "d"],
                "negative_prompts": [" "] * 4,
                "metadata": [{}] * 4,
            }
        ],
        None,
        algo_cfg=algo_cfg,
        loss_cfg=DiffusionLossConfig(),
        logger=FakeLogger(),
        checkpoint_dir=None,
    )
    return policy, logged


def test_mini_batches_preserve_order_and_groups():
    policy, logged = _run_one_train_step(_mini_batch_algo_cfg(ppo_mini_batch_size=8))

    # 16 samples / mini 8 → exactly two optimizer updates, in rollout order.
    assert policy.train_calls == [list(range(0, 8)), list(range(8, 16))]
    # Each mini holds only complete K=4 groups (group id = sample id // K).
    for ids in policy.train_calls:
        groups = [i // 4 for i in ids]
        assert all(groups.count(g) == 4 for g in set(groups))
    # Cross-mini metrics are averaged (verl-omni reduce_metrics semantics).
    assert logged[-1]["train/loss"] == 1.5


def test_mini_batch_disabled_trains_whole_batch_once():
    policy, logged = _run_one_train_step(_mini_batch_algo_cfg())

    assert policy.train_calls == [list(range(16))]
    assert logged[-1]["train/loss"] == 1.0


def test_mini_batches_repeat_across_ppo_epochs():
    policy, _ = _run_one_train_step(
        _mini_batch_algo_cfg(ppo_mini_batch_size=8, ppo_epochs=2)
    )

    assert policy.train_calls == [list(range(0, 8)), list(range(8, 16))] * 2


def test_global_std_tames_constant_group_amplification():
    import torch

    from nemo_rl.algorithms.diffusion_grpo import _compute_advantages

    # Group a is all-constant (common under OCR rewards); group b carries a
    # tiny signal.
    prompts = ["a"] * 4 + ["b"] * 4
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.02, 0.0, 0.0])
    adv_group = _compute_advantages(
        prompts, rewards, use_leave_one_out_baseline=True, use_global_std=False
    )
    adv_global = _compute_advantages(
        prompts, rewards, use_leave_one_out_baseline=True, use_global_std=True
    )
    # Per-group std normalization amplifies group b's tiny spread far beyond
    # the globally normalized magnitude.
    assert adv_group.abs().max() > 10 * adv_global.abs().max()
    # Under global normalization the constant group's advantage is 0 and the
    # overall magnitude stays bounded.
    assert torch.all(adv_global[:4].abs() < 1e-3)
    assert adv_global.abs().max() < 5.0
