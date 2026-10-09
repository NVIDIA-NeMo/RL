# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.algorithms.single_controller_utils.multi_trace import (
    logical_rollout_batch,
    trace_training_evidence,
    validate_nonempty_optimizer_metrics,
    validate_multitrace_resume_config,
    validate_multitrace_resume_progress,
)


def _tags(counts: list[int], group_id: str = "task-1") -> list[dict]:
    return [
        {
            "group_id": group_id,
            "rollout_index": rollout,
            "trace_index": trace,
            "num_traces": count,
            "num_logical_rollouts": len(counts),
            "segment_boundary_reason": ("initial", "compaction", "post_compaction")[
                trace % 3
            ],
            "is_compaction_summary": trace % 3 == 1,
        }
        for rollout, count in enumerate(counts)
        for trace in range(count)
    ]


def _estimator() -> GRPOAdvantageEstimator:
    return GRPOAdvantageEstimator(AdvEstimatorConfig(), None)


def test_single_trace_matches_baseline_advantages_and_gradient() -> None:
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    valid = torch.tensor([1.0, 1.0, 0.0, 1.0])
    mask = torch.tensor([[0, 0, 1, 1, 0]] * 4).float()
    logical = logical_rollout_batch(
        _tags([1, 1, 1, 1]),
        rewards=rewards,
        sample_mask=valid,
        token_mask=mask,
        expected_generations=4,
    )
    estimator = _estimator()
    baseline = estimator.compute_advantage(
        torch.tensor([[10, 11]] * 4),
        rewards,
        mask,
        valid_mask=valid,
    )
    collapsed = estimator.compute_advantage(
        logical.prompt_ids,
        logical.rewards,
        torch.ones(4, 1),
        valid_mask=logical.valid_mask,
    )[logical.trace_to_rollout].expand_as(mask)
    torch.testing.assert_close(collapsed, baseline, rtol=0, atol=0)
    logits = torch.zeros_like(mask, requires_grad=True)
    baseline_loss = -(
        logits[:, :-1] * baseline[:, 1:] * mask[:, 1:] * valid[:, None]
    ).sum()
    collapsed_loss = -(
        logits[:, :-1] * collapsed[:, 1:] * mask[:, 1:] * valid[:, None]
    ).sum()
    torch.testing.assert_close(
        torch.autograd.grad(baseline_loss, logits)[0],
        torch.autograd.grad(collapsed_loss, logits)[0],
        rtol=0,
        atol=0,
    )


def test_summary_and_post_compaction_share_one_outcome_and_train_targets() -> None:
    tags = _tags([1, 3, 1])
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0, 100.0])
    valid = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    mask = torch.tensor([[0, 0, 1, 1, 0]] * 5).float()
    logical = logical_rollout_batch(
        tags,
        rewards=rewards,
        sample_mask=valid,
        token_mask=mask,
        expected_generations=3,
    )
    assert logical.rewards.tolist() == [0.0, 1.0, 100.0]
    assert logical.valid_mask.tolist() == [1.0, 1.0, 0.0]
    adv = (
        _estimator()
        .compute_advantage(
            logical.prompt_ids,
            logical.rewards,
            torch.ones(3, 1),
            valid_mask=logical.valid_mask,
        )[logical.trace_to_rollout]
        .expand_as(mask)
    )
    # Invalid sibling cannot pollute valid siblings; trace count cannot weight votes.
    torch.testing.assert_close(adv[0], -adv[1])
    torch.testing.assert_close(adv[1], adv[2])
    torch.testing.assert_close(adv[2], adv[3])
    assert adv[2, 2] > 0
    logits = torch.zeros_like(mask, requires_grad=True)
    effective = mask * valid[:, None]
    loss = (
        -(logits[:, :-1] * adv[:, 1:] * effective[:, 1:]).sum() / effective[:, 1:].sum()
    )
    loss.backward()
    assert logits.grad[2, 1:3].ne(0).all()  # summary targets
    assert logits.grad[3, 1:3].ne(0).all()  # post-compaction targets
    assert logits.grad[:, 0].eq(0).all()  # repeated context
    assert logits.grad[:, 3:].eq(0).all()  # padding / causal tail
    assert logits.grad[4].eq(0).all()  # invalid logical sibling
    evidence = trace_training_evidence(
        tags,
        sample_ids=[str(i) for i in range(5)],
        token_mask=mask,
        effective_mask=effective,
        advantages=adv,
    )
    assert evidence[2]["is_compaction_summary"]
    assert evidence[2]["nonzero_advantage_targets"] == 2
    assert evidence[3]["nonzero_advantage_targets"] == 2
    assert evidence[4]["retained_targets"] == 0


def test_multiple_compactions_empty_traces_and_partial_trace_filtering() -> None:
    mask = torch.tensor([[0, 1, 1]] * 7).float()
    mask[2] = 0  # empty output cannot make the rollout valid on its own
    logical = logical_rollout_batch(
        _tags([5, 1, 1]),
        rewards=torch.tensor([1.0] * 5 + [0.0, 1.0]),
        sample_mask=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        token_mask=mask,
        expected_generations=3,
    )
    assert logical.valid_mask.tolist() == [0.0, 1.0, 0.0]


def test_original_group_identity_survives_identical_or_rewritten_prompts() -> None:
    tags = _tags([2, 1], "first") + _tags([1, 1], "second")
    logical = logical_rollout_batch(
        tags,
        rewards=torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]),
        sample_mask=torch.ones(5),
        token_mask=torch.tensor([[0, 1]] * 5),
        expected_generations=2,
    )
    assert logical.prompt_ids.tolist() == [[0], [0], [1], [1]]
    assert logical.trace_to_rollout.tolist() == [0, 0, 1, 2, 3]


@pytest.mark.parametrize(
    "defect",
    ["missing_sibling", "missing_trace", "duplicate_trace", "different_reward"],
)
def test_incomplete_or_inconsistent_ownership_fails(defect: str) -> None:
    tags = deepcopy(_tags([1, 3]))
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0])
    if defect == "missing_sibling":
        tags = tags[1:]
        rewards = rewards[1:]
    elif defect == "missing_trace":
        tags = tags[:-1]
        rewards = rewards[:-1]
    elif defect == "duplicate_trace":
        tags[-1]["trace_index"] = 1
    else:
        rewards[-1] = 0.0
    with pytest.raises(ValueError):
        logical_rollout_batch(
            tags,
            rewards=rewards,
            sample_mask=torch.ones(len(tags)),
            token_mask=torch.tensor([[0, 1]] * len(tags)),
            expected_generations=2,
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"loss": float("nan")},
        {"grad_norm": float("inf")},
        {"grad_norm": None},
        {"global_valid_toks": 0},
    ],
)
def test_empty_or_nonfinite_optimizer_metrics_cannot_advance(bad: dict) -> None:
    with pytest.raises(RuntimeError):
        validate_nonempty_optimizer_metrics(
            {"loss": 1.0, "grad_norm": 0.2, "global_valid_toks": 8, **bad}
        )


@pytest.mark.parametrize("split_chunks", [False, True])
def test_compaction_production_loss_gradients_with_candidate_tis(
    split_chunks: bool,
) -> None:
    """Exercise the actual loss, causal mask, TIS clamps and global token divisor."""
    # Rollout 1 owns ordinary work, summary and continuation rows. The final
    # sibling is an invalid high-reward outcome and cannot affect its baseline.
    tags = _tags([1, 3, 1])
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0, 100.0])
    sample_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    token_mask = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],  # summary: repeated context before t=4
            [0, 0, 0, 0, 0, 1, 1, 1, 0],  # continuation: summary is context-only
            [0, 0, 1, 1, 0, 0, 0, 0, 0],
        ]
    ).float()
    logical = logical_rollout_batch(
        tags,
        rewards=rewards,
        sample_mask=sample_mask,
        token_mask=token_mask,
        expected_generations=3,
    )
    advantages = (
        _estimator()
        .compute_advantage(
            logical.prompt_ids,
            logical.rewards,
            torch.ones(3, 1),
            valid_mask=logical.valid_mask,
        )[logical.trace_to_rollout]
        .expand_as(token_mask)
    )
    torch.testing.assert_close(advantages[2], advantages[3])
    assert advantages[2, 4] > 0

    # Match candidate.yaml: force the PPO ratio to 1 while the independent
    # behavior-policy TIS weight remains active and is clamped to [0.2, 5].
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            token_level_loss=True,
            sequence_level_importance_ratios=False,
            ratio_clip_min=0.2,
            ratio_clip_max=0.28,
            reference_policy_kl_penalty=0.0,
            kl_input_clamp_value=None,
            kl_output_clamp_value=None,
            force_on_policy_ratio=True,
            use_on_policy_kl_approximation=True,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio_min=0.2,
            truncated_importance_sampling_ratio=5.0,
        )
    )
    current_logprobs = torch.full((5, 8), -4.0, requires_grad=True)
    raw_weights = torch.ones_like(current_logprobs)
    raw_weights[2, 3:5] = torch.tensor([0.05, 10.0])
    raw_weights[3, 4:7] = torch.tensor([0.5, 1.0, 2.0])
    generation_logprobs = torch.zeros_like(token_mask)
    generation_logprobs[:, 1:] = current_logprobs.detach() - raw_weights.log()
    fields = {
        "input_ids": torch.zeros_like(token_mask, dtype=torch.long),
        "advantages": advantages,
        "generation_logprobs": generation_logprobs,
        # A deliberately different old-policy value verifies that force-on-policy
        # does not accidentally replace the independent behavior-policy TIS path.
        "prev_logprobs": torch.full_like(token_mask, -30.0),
        "token_mask": token_mask,
        "sample_mask": sample_mask,
    }
    causal_mask = token_mask[:, 1:] * sample_mask[:, None]
    global_valid_toks = causal_mask.sum()
    global_valid_seqs = (causal_mask.sum(-1) > 0).float().sum()
    assert global_valid_toks.item() == 8
    chunks = [slice(0, 2), slice(2, 5)] if split_chunks else [slice(None)]
    losses, metrics = [], []
    for rows in chunks:
        loss, chunk_metrics = loss_fn(
            next_token_logprobs=current_logprobs[rows],
            data=BatchedDataDict({name: value[rows] for name, value in fields.items()}),
            global_valid_toks=global_valid_toks,
            global_valid_seqs=global_valid_seqs,
        )
        losses.append(loss)
        metrics.append(chunk_metrics)
    actual_loss = torch.stack(losses).sum()
    actual_loss.backward()

    # With ratio=1, dL/dlogp=-A*TIS/N. The explicit expected weights exercise
    # both truncation bounds, rather than treating IS as a constant all-ones mask.
    expected_weights = raw_weights.clone()
    expected_weights[2, 3:5] = torch.tensor([0.2, 5.0])
    expected_grad = -advantages[:, 1:] * expected_weights * causal_mask / 8
    torch.testing.assert_close(actual_loss, expected_grad.sum())
    torch.testing.assert_close(current_logprobs.grad, expected_grad)
    assert current_logprobs.grad[2, 3:5].ne(0).all()
    assert current_logprobs.grad[3, 4:7].ne(0).all()
    assert current_logprobs.grad[~causal_mask.bool()].eq(0).all()
    assert current_logprobs.grad[4].eq(0).all()
    assert sum(m["probs_ratio"] for m in metrics) == pytest.approx(1.0)
    assert sum(m["is_oob_ratio"] for m in metrics) == pytest.approx(2 / 8)


def test_multitrace_resume_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="requires explicit"):
        validate_multitrace_resume_progress(
            recovery_enabled=False,
            current_step=2,
            total_steps=2,
            trainer_version=2,
            consumed_groups=64,
            groups_per_update=32,
            max_updates=4,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"total_steps": 0},
        {"trainer_version": 0},
        {"trainer_version": None},
        {"consumed_groups": 128},
        {"groups_per_update": 16},
        {"max_updates": 1},
    ],
)
def test_multitrace_resume_rejects_reset_or_inflated_progress(change):
    kwargs = {
        "recovery_enabled": True,
        "current_step": 2,
        "total_steps": 2,
        "trainer_version": 2,
        "consumed_groups": 64,
        "groups_per_update": 32,
        "max_updates": 4,
        **change,
    }
    with pytest.raises(ValueError):
        validate_multitrace_resume_progress(**kwargs)


def test_multitrace_resume_preserves_saved_progress_and_operational_changes():
    validate_multitrace_resume_progress(
        recovery_enabled=True,
        current_step=2,
        total_steps=2,
        trainer_version=2,
        consumed_groups=64,
        groups_per_update=32,
        max_updates=4,
    )
    before = {
        "data": {
            "train": [{"dataset_name": "pilot", "data_path": "/frozen/train.jsonl"}]
        },
        "grpo": {
            "num_prompts_per_step": 32,
            "num_generations_per_prompt": 16,
            "max_num_steps": 4,
        },
        "policy": {
            "train_global_batch_size": 512,
            "megatron_cfg": {"scheduler": {"lr_decay_iters": 4, "lr_warmup_iters": 0}},
        },
        "async_rl": {
            "sampler": {"name": "windowed", "max_staleness_versions": 2},
            "fault_injection_pause_before_update": 3,
        },
    }
    after = deepcopy(before)
    after["async_rl"]["fault_injection_pause_before_update"] = None
    after["async_rl"]["allocation_stop_after_step"] = None
    validate_multitrace_resume_config(before, after)
    for path, value in [
        (("data", "train"), [{"dataset_name": "new"}]),
        (("grpo", "max_num_steps"), 40),
        (("grpo", "num_prompts_per_step"), 16),
        (
            ("policy", "megatron_cfg", "scheduler"),
            {"lr_decay_iters": 40, "lr_warmup_iters": 10},
        ),
        (("async_rl", "sampler"), {"name": "windowed", "max_staleness_versions": 5}),
    ]:
        changed = deepcopy(after)
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        with pytest.raises(ValueError, match="changed training semantics"):
            validate_multitrace_resume_config(before, changed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("opencode_training_mode", "compaction"),
        ("opencode_subagents_enabled", False),
        ("opencode_subagent_max_depth", 2),
        ("opencode_subagent_max_active", 2),
        ("opencode_compaction_auto", True),
        ("opencode_pruning_enabled", True),
    ],
)
def test_native_resume_binds_subagent_mode_and_runtime_limits(key, value):
    before = {
        "env": {
            "nemo_gym": {
                "training_mode": "subagent",
                "results_dir": "/old-allocation",
                "swe": {
                    "responses_api_agents": {
                        "swe_agents": {
                            "opencode_training_mode": "subagent",
                            "opencode_subagents_enabled": True,
                            "opencode_subagent_max_depth": 1,
                            "opencode_subagent_max_active": 1,
                            "opencode_compaction_auto": False,
                            "opencode_pruning_enabled": False,
                        }
                    }
                },
            }
        }
    }
    after = deepcopy(before)
    after["env"]["nemo_gym"]["results_dir"] = "/new-allocation"
    validate_multitrace_resume_config(before, after)
    after["env"]["nemo_gym"]["swe"]["responses_api_agents"]["swe_agents"][key] = value
    with pytest.raises(ValueError, match="session training mode or limits"):
        validate_multitrace_resume_config(before, after)
    after = deepcopy(before)
    after["env"]["nemo_gym"]["training_mode"] = "plain"
    with pytest.raises(ValueError, match="session training mode or limits"):
        validate_multitrace_resume_config(before, after)
