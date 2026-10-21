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
"""CPU tests for adaptive-ppo-clip filtering and its entropy controller."""

import pickle
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.grpo import (
    _get_grpo_save_state,
    _initial_grpo_save_state,
    _restore_adaptive_loss_state,
    _save_adaptive_loss_state,
    _update_adaptive_loss_state,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import MetricNormalizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _adaptive_loss(**overrides: Any) -> ClippedPGLossFn:
    config = {
        "reference_policy_kl_penalty": 0.0,
        "token_level_loss": True,
        "use_importance_sampling_correction": True,
        "force_on_policy_ratio": True,
        "truncated_importance_sampling_type": "adaptive-ppo-clip",
        "truncated_importance_sampling_ratio": 1.0,
        "truncated_importance_sampling_ratio_min": 0.5,
        "truncated_importance_sampling_adaptive_target_entropy": 0.3,
    }
    config.update(overrides)
    return ClippedPGLossFn(ClippedPGLossConfig(**config))


def _loss_inputs(weights: list[float], advantages: list[float]) -> dict[str, Any]:
    """Build on-policy optimization inputs with controlled behavior-policy ratios."""
    token_count = len(weights)
    previous = torch.full((1, token_count), -5.0, dtype=torch.float64)
    current = previous.clone().requires_grad_()
    generation = previous - torch.log(torch.tensor([weights], dtype=torch.float64))

    def prepend(value: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.zeros((1, 1), dtype=value.dtype), value), dim=-1)

    data = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, token_count + 1), dtype=torch.long),
            "token_mask": prepend(torch.ones_like(previous)),
            "sample_mask": torch.ones(1, dtype=torch.float64),
            "advantages": prepend(torch.tensor([advantages], dtype=torch.float64)),
            "prev_logprobs": prepend(previous),
            "generation_logprobs": prepend(generation),
            "reference_policy_logprobs": prepend(previous),
        }
    )
    return {
        "next_token_logprobs": current,
        "data": data,
        "global_valid_seqs": torch.tensor(1.0),
        "global_valid_toks": torch.tensor(float(token_count)),
    }


@pytest.mark.parametrize("token_level_loss", [False, True])
def test_filters_by_advantage_sign_and_hard_outer_bound(
    token_level_loss: bool,
) -> None:
    loss_fn = _adaptive_loss(token_level_loss=token_level_loss)
    inputs = _loss_inputs(
        weights=[0.2, 0.2, 3.0, 3.0, 51.0, 51.0, 1.2, 49.0, 51.0],
        advantages=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.0],
    )
    loss, metrics = loss_fn(**inputs)
    # Low-weight negative tokens, high-weight positive tokens, and every token
    # beyond the hard outer bound are removed. Zero advantage ignores inner bounds.
    expected_gradient = (
        torch.tensor(
            [[0.0, -0.2, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        / 9
    )
    torch.testing.assert_close(loss, expected_gradient.sum())
    loss.backward()
    torch.testing.assert_close(inputs["next_token_logprobs"].grad, expected_gradient)
    assert metrics["is_oob_ratio"] == pytest.approx(6.0 / 9.0)
    assert loss_fn.metric_normalizations["is_oob_ratio"] == MetricNormalizer.TOKENS


def test_filter_metric_respects_token_mask_and_global_denominator() -> None:
    loss_fn = _adaptive_loss()
    inputs = _loss_inputs([0.2, 0.2, 3.0], [-1.0, -1.0, 1.0])
    inputs["data"]["token_mask"][0, 2] = 0.0
    inputs["global_valid_toks"] = torch.tensor(8.0)
    _, metrics = loss_fn(**inputs)
    assert metrics["is_oob_ratio"] == pytest.approx(2.0 / 8.0)


def test_entropy_changes_next_step_filter_after_serialization() -> None:
    loss_fn = _adaptive_loss()
    initial_loss, _ = loss_fn(**_loss_inputs([1.2], [1.0]))
    torch.testing.assert_close(initial_loss, torch.tensor(0.0, dtype=torch.float64))

    # Workers receive the controller state with the pickled loss function.
    loss_fn.update_adaptive_loss_state(0.1)
    worker_loss = pickle.loads(pickle.dumps(loss_fn))
    next_loss, _ = worker_loss(**_loss_inputs([1.2], [1.0]))
    torch.testing.assert_close(next_loss, torch.tensor(-1.2, dtype=torch.float64))
    assert worker_loss.cur_adaptive_ppo_clip_k == 0.25


def test_entropy_controller_clamps_and_holds_at_target() -> None:
    loss_fn = _adaptive_loss()
    for _ in range(6):
        loss_fn.update_adaptive_loss_state(0.1)
    assert loss_fn.cur_adaptive_ppo_clip_k == 1.0
    loss_fn.update_adaptive_loss_state(0.3)
    assert loss_fn.cur_adaptive_ppo_clip_k == 1.0
    loss_fn.update_adaptive_loss_state(0.5)
    assert loss_fn.cur_adaptive_ppo_clip_k == 0.75
    for _ in range(6):
        loss_fn.update_adaptive_loss_state(0.5)
    assert loss_fn.cur_adaptive_ppo_clip_k == 0.0


def test_controller_and_outer_bound_follow_config() -> None:
    # Non-default values show that every knob comes from the config.
    loss_fn = _adaptive_loss(
        truncated_importance_sampling_adaptive_target_entropy=0.5,
        truncated_importance_sampling_adaptive_k_step=0.5,
        truncated_importance_sampling_adaptive_k_max=2.5,
        truncated_importance_sampling_adaptive_outer_bound=3.0,
    )
    loss_fn.update_adaptive_loss_state(0.4)
    assert loss_fn.cur_adaptive_ppo_clip_k == 0.5
    for _ in range(6):
        loss_fn.update_adaptive_loss_state(0.4)
    assert loss_fn.cur_adaptive_ppo_clip_k == 2.5
    # Upper bound is 1.0 + 2.5 = 3.5, but the outer bound of 3.0 drops 3.2.
    loss, metrics = loss_fn(**_loss_inputs([2.8, 3.2], [1.0, 1.0]))
    torch.testing.assert_close(loss, torch.tensor(-1.4, dtype=torch.float64))
    assert metrics["is_oob_ratio"] == pytest.approx(0.5)
    # Entropy above the target narrows the bound by the configured step.
    loss_fn.update_adaptive_loss_state(0.6)
    assert loss_fn.cur_adaptive_ppo_clip_k == 2.0


def test_set_k_clamps_to_configured_range() -> None:
    loss_fn = _adaptive_loss()
    loss_fn.set_adaptive_ppo_clip_k(0.5)
    assert loss_fn.cur_adaptive_ppo_clip_k == 0.5
    loss_fn.set_adaptive_ppo_clip_k(5.0)
    assert loss_fn.cur_adaptive_ppo_clip_k == 1.0
    loss_fn.set_adaptive_ppo_clip_k(-5.0)
    assert loss_fn.cur_adaptive_ppo_clip_k == 0.0
    wide = _adaptive_loss(truncated_importance_sampling_adaptive_k_max=2.5)
    wide.set_adaptive_ppo_clip_k(5.0)
    assert wide.cur_adaptive_ppo_clip_k == 2.5


def test_other_tis_types_do_not_acquire_adaptive_state() -> None:
    loss_fn = _adaptive_loss(truncated_importance_sampling_type="icepop")
    loss_fn.set_adaptive_ppo_clip_k(0.75)
    loss_fn.update_adaptive_loss_state(0.1)
    assert not loss_fn.has_adaptive_loss_state()
    assert loss_fn.cur_adaptive_ppo_clip_k is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_importance_sampling_correction": False},
        {"truncated_importance_sampling_ratio_min": None},
        {"truncated_importance_sampling_ratio_min": 2.0},
        {"sequence_level_importance_ratios": True, "token_level_loss": False},
        {"metrics_level": "minimal"},
    ],
)
def test_rejects_incompatible_configuration(overrides: dict) -> None:
    with pytest.raises(AssertionError):
        _adaptive_loss(**overrides)


def test_requires_force_on_policy_ratio() -> None:
    with pytest.raises(AssertionError, match="intended for async RL"):
        _adaptive_loss(force_on_policy_ratio=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"truncated_importance_sampling_adaptive_target_entropy": -0.1},
        {"truncated_importance_sampling_adaptive_k_step": 0.0},
        {"truncated_importance_sampling_adaptive_k_max": -0.1},
        {"truncated_importance_sampling_adaptive_outer_bound": 0.0},
    ],
)
def test_rejects_invalid_controller_settings(overrides: dict) -> None:
    with pytest.raises(ValueError):
        ClippedPGLossConfig(**overrides)


# ── GRPO save-state helpers shared by the legacy trainers and SingleController ──


def test_inactive_adaptive_state_is_saved_as_none() -> None:
    for loss_fn in (ClippedPGLossFn(ClippedPGLossConfig()), object()):
        # Start from a k left by an earlier adaptive-ppo-clip run.
        state = _initial_grpo_save_state()
        state.cur_adaptive_ppo_clip_k = 0.5
        _save_adaptive_loss_state(state, loss_fn)  # type: ignore[arg-type]
        assert state.cur_adaptive_ppo_clip_k is None


def test_step_state_survives_checkpoint_and_restore() -> None:
    state = _initial_grpo_save_state()
    loss_fn = _adaptive_loss()
    _save_adaptive_loss_state(state, loss_fn)
    assert state.cur_adaptive_ppo_clip_k == 0.0

    # The logged k is the one this step trained with; the update applies next step.
    metrics = {"approx_entropy": 0.1}
    _update_adaptive_loss_state(loss_fn, metrics)
    assert metrics == {"approx_entropy": 0.1, "cur_adaptive_ppo_clip_k": 0.0}
    assert loss_fn.cur_adaptive_ppo_clip_k == 0.25
    _save_adaptive_loss_state(state, loss_fn)
    resumed = _adaptive_loss()
    _restore_adaptive_loss_state(resumed, _get_grpo_save_state(vars(state)))
    assert resumed.cur_adaptive_ppo_clip_k == 0.25

    # Checkpoints written before adaptive-ppo-clip was enabled start from k=0.
    legacy = _adaptive_loss()
    _restore_adaptive_loss_state(legacy, _get_grpo_save_state({"total_steps": 8}))
    assert legacy.cur_adaptive_ppo_clip_k == 0.0


def test_missing_entropy_raises() -> None:
    loss_fn = _adaptive_loss()
    with pytest.raises(KeyError, match="approx_entropy"):
        _update_adaptive_loss_state(loss_fn, {"loss": 1.0})
    assert loss_fn.cur_adaptive_ppo_clip_k == 0.0
