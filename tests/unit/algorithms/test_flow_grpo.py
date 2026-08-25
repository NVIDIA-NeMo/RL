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
"""Tests for the flow-GRPO training loop."""

import json
import os


def test_master_config_rejects_kl_with_full_param():
    import pytest
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.flow_grpo import MasterConfig
    from nemo_rl.utils.config import load_config

    cfg = OmegaConf.to_container(
        load_config("examples/configs/flow_grpo_qwen_image_ocr.yaml"),
        resolve=True,
    )
    cfg["loss_fn"]["beta"] = 0.04
    cfg["policy"]["lora_cfg"]["enabled"] = False
    with pytest.raises(Exception, match="beta"):
        MasterConfig.model_validate(cfg)


def test_run_validation_passes_generation_overrides():
    import torch

    from nemo_rl.algorithms.flow_grpo import _run_validation

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

    from nemo_rl.algorithms.flow_grpo import _build_train_data

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
    out = _build_train_data(traj, torch.tensor([1.0, -1.0]))
    assert out["timesteps"].shape == (B, w)
    assert out["latents"].shape == (B, w + 1, 4)
    assert torch.all(out["timestep_mask"] == 1)
    # Sample 1's window starts at 4, so the sliced timesteps must be [4, 5, 6].
    assert out["timesteps"][1].tolist() == [4.0, 5.0, 6.0]
    # Latents keep one extra column (w + 1): sample 1 must equal the original latents[1, 4:8].
    assert torch.equal(out["latents"][1], traj["latents"][1, 4:8])


def _mini_batch_algo_cfg(**overrides):
    from nemo_rl.algorithms.flow_grpo import FlowGRPOAlgoConfig

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
    return FlowGRPOAlgoConfig.model_validate(cfg)


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
    """Fake FlowGRPOPolicy capturing each train() call's sample ids."""

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

    def save_checkpoint(self, path, *, save_optimizer=True):
        raise AssertionError("checkpointing is disabled in this test")


def _master_config(algo_cfg, checkpoint_dir=None):
    """Exemplar config with the algo block swapped for the test's `algo_cfg`."""
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.flow_grpo import MasterConfig
    from nemo_rl.utils.config import load_config

    cfg = OmegaConf.to_container(
        load_config("examples/configs/flow_grpo_qwen_image_ocr.yaml"),
        resolve=True,
    )
    cfg["checkpointing"]["enabled"] = checkpoint_dir is not None
    if checkpoint_dir is not None:
        cfg["checkpointing"]["checkpoint_dir"] = str(checkpoint_dir)
    master = MasterConfig.model_validate(cfg)
    master.flow_grpo = algo_cfg
    return master


def _run_one_train_step(algo_cfg, checkpoint_dir=None, policy=None):
    import torch

    from nemo_rl.algorithms.flow_grpo import flow_grpo_train
    from nemo_rl.utils.checkpoint import CheckpointManager

    K = algo_cfg.num_generations_per_prompt
    policy = policy if policy is not None else _RecordingPolicy(K)
    master = _master_config(algo_cfg, checkpoint_dir)

    class FakeEnv:
        def score_images(self, images, prompts, metadata):
            # Varying rewards so the advantage path sees non-constant groups.
            return torch.arange(images.shape[0], dtype=torch.float32), {}

    logged = []

    class FakeLogger:
        def log_metrics(self, metrics, step):
            logged.append(metrics)

    flow_grpo_train(
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
        master_config=master,
        logger=FakeLogger(),
        checkpointer=CheckpointManager(master.checkpointing),
    )
    return policy, logged


def test_resume_reads_step_from_training_info(tmp_path):
    """A complete step_N/ resumes at N and loads from its policy/ subdir."""
    step_dir = tmp_path / "step_5"
    step_dir.mkdir(parents=True)
    (step_dir / "training_info.json").write_text(json.dumps({"step": 5}))

    class ResumingPolicy(_RecordingPolicy):
        def __init__(self, K):
            super().__init__(K)
            self.loaded_from = None

        def load_checkpoint(self, path):
            self.loaded_from = path

    policy = ResumingPolicy(4)
    # max_num_steps == the resumed step, so the loop body never runs.
    algo_cfg = _mini_batch_algo_cfg(max_num_steps=5)
    _run_one_train_step(algo_cfg, checkpoint_dir=tmp_path, policy=policy)

    assert policy.loaded_from == os.path.join(str(step_dir), "policy")
    assert policy.train_calls == []


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

    from nemo_rl.algorithms.flow_grpo import _compute_advantages

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
