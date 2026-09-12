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
"""CPU tests of the Gym wire contract and the actual GRPO loss boundary."""

from copy import deepcopy

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import (
    GRPOConfig,
    MasterConfig,
    RewardPenaltyConfig,
    _validate_gym_multi_trace_capability,
    add_grpo_token_loss_masks_and_generation_logprobs,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.gym_traces import (
    gym_trace_message_log,
    parse_gym_training_traces,
    prepare_gym_trace_batch,
)


def _row(trace_id, tokens, spans):
    mask = [0] * len(tokens)
    for _, start, end in spans:
        mask[start:end] = [1] * (end - start)
    return {
        "trace_id": trace_id,
        "model_call_ids": [call for call, _, _ in spans],
        "token_ids": tokens,
        "generation_logprobs": [-0.125 if bit else 0.0 for bit in mask],
        "loss_mask": mask,
        "sampled_spans": [
            {"model_call_id": call, "start": start, "end": end}
            for call, start, end in spans
        ],
    }


def _envelope(rollout, rows, builder="prefix_merging"):
    return {
        "schema_version": 1,
        "rollout_id": rollout,
        "builder": builder,
        "traces": rows,
    }


def _batch(envelopes, *, rewards, groups, masks=None):
    traces = [
        parse_gym_training_traces(envelope) if envelope else []
        for envelope in envelopes
    ]
    return BatchedDataDict(
        {
            "gym_training_traces": traces,
            "gym_rollout_id": [
                envelope["rollout_id"] if envelope else f"masked-{index}"
                for index, envelope in enumerate(envelopes)
            ],
            "gym_task_group_id": torch.tensor(groups),
            "total_reward": torch.tensor(rewards, dtype=torch.float32),
            "loss_multiplier": torch.tensor(
                masks if masks is not None else [1.0] * len(envelopes)
            ),
            "message_log": [
                gym_trace_message_log(rows[0]) if rows else [] for rows in traces
            ],
            "length": torch.ones(len(envelopes), dtype=torch.long),
        }
    )


def _prepare(batch, *, max_length=32, multiple=4):
    return prepare_gym_trace_batch(
        batch,
        estimator=GRPOAdvantageEstimator(
            AdvEstimatorConfig(normalize_rewards=False), ClippedPGLossConfig()
        ),
        pad_token_id=0,
        max_sequence_length=max_length,
        row_multiple=multiple,
    )


def _flatten(prepared):
    add_grpo_token_loss_masks_and_generation_logprobs(prepared.batch["message_log"])
    flat, lengths = batched_message_log_to_flat_message(
        prepared.batch["message_log"], pad_value_dict={"token_ids": 0}
    )
    return BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": lengths,
            "token_mask": flat["token_loss_mask"],
            "generation_logprobs": flat["generation_logprobs"],
            "sample_mask": prepared.batch["loss_multiplier"],
            "advantages": prepared.advantages.unsqueeze(-1).expand_as(
                flat["token_ids"]
            ),
        }
    )


def test_exact_tokens_logprobs_masks_and_context_edits():
    envelope = _envelope(
        "rollout",
        [
            _row("a", [1, 2, 3, 9, 4, 5], [("a1", 2, 3), ("a2", 4, 6)]),
            _row("b", [7, 8], [("b1", 1, 2)]),
        ],
    )
    prepared = _prepare(_batch([envelope], rewards=[1], groups=[17]))
    flat = _flatten(prepared)
    assert flat["input_ids"][0].tolist() == [1, 2, 3, 9, 4, 5]
    assert flat["input_lengths"][0].item() == 6
    assert prepared.batch["length"][0].item() == 2
    assert flat["token_mask"][0].tolist() == [0, 0, 1, 0, 1, 1]
    assert flat["generation_logprobs"][0].tolist() == [0, 0, -0.125, 0, -0.125, -0.125]
    assert flat["sample_mask"].tolist() == [1, 1, 0, 0]
    assert (
        prepared.logical_count == 1
        and prepared.physical_count == 2
        and prepared.padding_count == 2
    )


def test_task_groups_and_masked_rollouts_vote_before_expansion():
    envelopes = [
        _envelope(
            "a",
            [_row("a1", [1, 2], [("a1", 1, 2)]), _row("a2", [3, 4, 5], [("a2", 1, 3)])],
        ),
        _envelope("b", [_row("b1", [1, 3], [("b1", 1, 2)])]),
        _envelope("c", [_row("c1", [1, 4], [("c1", 1, 2)])]),
        _envelope("d", [_row("d1", [1, 5], [("d1", 1, 2)])]),
    ]
    prepared = _prepare(
        _batch(
            envelopes, rewards=[0, 100, 1, 0], groups=[7, 7, 8, 8], masks=[1, 0, 1, 1]
        )
    )
    assert prepared.logical_indices.tolist() == [0, 0, 1, 2, 3, 0, 0, 0]
    assert prepared.advantages[:5].tolist() == [0, 0, 0, 1, -1]
    assert prepared.batch["loss_multiplier"].tolist() == [1, 1, 0, 1, 1, 0, 0, 0]
    assert prepared.invalid_rollout_count == 1
    assert prepared.logical_valid_mask.tolist() == [1, 0, 1, 1]


def test_overlong_trace_masks_only_its_row_and_empty_rollout_is_inert():
    envelope = _envelope(
        "a",
        [
            _row("long", [1, 2, 3, 4, 5], [("long", 1, 5)]),
            _row("short", [1, 2], [("short", 1, 2)]),
        ],
    )
    prepared = _prepare(
        _batch([envelope, None], rewards=[1, 99], groups=[5, 5]), max_length=3
    )
    assert prepared.batch["loss_multiplier"].tolist() == [0, 1, 0, 0]
    assert prepared.overlong_trace_count == 1 and prepared.invalid_rollout_count == 1
    assert prepared.batch["gym_rollout_id"] == ["a", "a", "masked-1", ""]
    assert prepared.advantages[1].item() == 0
    assert _flatten(prepared)["input_lengths"].max().item() <= 3


@pytest.mark.parametrize("leave_one_out", [False, True])
def test_baseline_diagnostics_use_the_same_logical_validity_as_advantages(
    leave_one_out,
):
    envelopes = [
        _envelope("valid-a", [_row("short", [1, 2], [("call", 1, 2)])]),
        _envelope("valid-b", [_row("short", [1, 3], [("call", 1, 2)])]),
        _envelope("overlong", [_row("long", [1, 2, 3, 4], [("call", 1, 4)])]),
        None,
    ]
    logical_batch = _batch(envelopes, rewards=[1, 0, 100, 99], groups=[7, 7, 7, 7])
    prepared = prepare_gym_trace_batch(
        logical_batch,
        estimator=GRPOAdvantageEstimator(
            AdvEstimatorConfig(
                normalize_rewards=False, use_leave_one_out_baseline=leave_one_out
            ),
            ClippedPGLossConfig(),
        ),
        pad_token_id=0,
        max_sequence_length=3,
        row_multiple=4,
    )
    assert prepared.logical_valid_mask.tolist() == [1, 1, 0, 0]
    baseline, _ = calculate_baseline_and_std_per_prompt(
        logical_batch["gym_task_group_id"].unsqueeze(-1),
        logical_batch["total_reward"],
        prepared.logical_valid_mask,
        leave_one_out_baseline=leave_one_out,
    )
    # Invalid rewards cannot influence the surviving rollouts or their diagnostics.
    assert baseline[:2].tolist() == ([0.0, 1.0] if leave_one_out else [0.5, 0.5])
    torch.testing.assert_close(
        prepared.advantages[:2], logical_batch["total_reward"][:2] - baseline[:2]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda env: env.update(schema_version=2),
        lambda env: env["traces"][0]["token_ids"].append(4),
        lambda env: env["traces"][0]["generation_logprobs"].__setitem__(
            1, float("nan")
        ),
        lambda env: env["traces"][0]["loss_mask"].__setitem__(0, 1),
        lambda env: env["traces"][0]["sampled_spans"][0].update(end=99),
        lambda env: env["traces"].append(dict(env["traces"][0], trace_id="other")),
    ],
)
def test_bad_custody_is_rejected(mutation):
    envelope = _envelope("a", [_row("trace", [1, 2], [("call", 1, 2)])])
    mutation(envelope)
    with pytest.raises(ValueError):
        parse_gym_training_traces(envelope)


def test_duplicate_rollout_and_all_invalid_batch_are_rejected():
    envelope = _envelope("a", [_row("trace", [1, 2], [("call", 1, 2)])])
    with pytest.raises(ValueError, match="Duplicate Gym rollout_id"):
        _prepare(_batch([envelope, envelope], rewards=[1, 0], groups=[0, 0]))
    with pytest.raises(ValueError, match="no eligible training tokens"):
        _prepare(_batch([None], rewards=[1], groups=[0]))


def test_equivalent_split_and_merged_conditioning_has_identical_loss_and_gradient():
    merged = _envelope(
        "a", [_row("merged", [1, 2, 3, 4, 5, 6], [("a1", 2, 3), ("a2", 4, 6)])]
    )
    split = _envelope(
        "a",
        [
            _row("first", [1, 2, 3], [("a1", 2, 3)]),
            _row("second", [1, 2, 3, 4, 5, 6], [("a2", 4, 6)]),
        ],
        builder="per_request",
    )
    sibling = _envelope("b", [_row("other", [7, 8], [("b1", 1, 2)])])
    torch.manual_seed(11)
    base_model = torch.nn.Sequential(torch.nn.Embedding(16, 5), torch.nn.Linear(5, 16))
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0))
    outcomes = []
    for envelope in (merged, split):
        model = deepcopy(base_model)
        prepared = _prepare(
            _batch([envelope, sibling], rewards=[1, 0], groups=[42, 42])
        )
        data = _flatten(prepared)
        # A deterministic causal model: the same complete prefix gives the same logits.
        hidden = model[0](data["input_ids"][:, :-1]).cumsum(dim=1)
        logits = model[1](hidden)
        logprobs = (
            logits.log_softmax(dim=-1)
            .gather(-1, data["input_ids"][:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        data["prev_logprobs"] = torch.nn.functional.pad(logprobs.detach(), (1, 0))
        valid_tokens = (
            data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
        ).sum()
        loss, _ = loss_fn(
            logprobs,
            data,
            global_valid_seqs=data["sample_mask"].sum(),
            global_valid_toks=valid_tokens,
        )
        loss.backward()
        outcomes.append(
            (
                loss.detach(),
                torch.cat(
                    [parameter.grad.flatten() for parameter in model.parameters()]
                ),
                valid_tokens,
            )
        )
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0])
    torch.testing.assert_close(outcomes[0][1], outcomes[1][1])
    assert outcomes[0][2] == outcomes[1][2] == 4


def _config():
    return MasterConfig.model_construct(
        grpo=GRPOConfig(
            gym_multi_trace=True, num_prompts_per_step=2, num_generations_per_prompt=2
        ),
        policy={"megatron_cfg": {"enabled": True}, "train_global_batch_size": 4},
        env={
            "should_use_nemo_gym": True,
            "nemo_gym": {
                "token_id_capture": {
                    "enabled": True,
                    "all_agents": True,
                    "delivery": "all_traces",
                }
            },
        },
        loss_fn=ClippedPGLossConfig(),
        reward_penalties=RewardPenaltyConfig(),
    )


@pytest.mark.parametrize(
    "configure",
    [
        lambda cfg: setattr(cfg.grpo.async_grpo, "enabled", True),
        lambda cfg: setattr(cfg, "data_plane", {"enabled": True}),
        lambda cfg: cfg.policy.update(is_vlm=True),
        lambda cfg: setattr(cfg.loss_fn, "token_level_loss", False),
        lambda cfg: setattr(cfg.grpo, "use_dynamic_sampling", True),
        lambda cfg: setattr(cfg.grpo, "seq_logprob_error_threshold", 2.0),
        lambda cfg: setattr(cfg.grpo.adv_estimator, "name", "gdpo"),
        lambda cfg: cfg.env["nemo_gym"]["token_id_capture"].update(sink="custom:Sink"),
        lambda cfg: cfg.policy.update(train_global_batch_size=8),
    ],
)
def test_unsupported_configs_fail_before_worker_setup(configure):
    config = _config()
    _validate_gym_multi_trace_capability(config)
    configure(config)
    with pytest.raises((NotImplementedError, ValueError)):
        _validate_gym_multi_trace_capability(config)


def test_sequence_mask_tis_is_rejected_only_when_correction_is_enabled():
    config = _config()
    config.loss_fn.truncated_importance_sampling_type = "seq-mask-tis"
    config.loss_fn.truncated_importance_sampling_ratio = 1.1
    config.loss_fn.truncated_importance_sampling_ratio_min = 0.9
    config.loss_fn.use_importance_sampling_correction = True
    with pytest.raises(NotImplementedError, match="seq-mask-tis"):
        _validate_gym_multi_trace_capability(config)
    config.loss_fn.use_importance_sampling_correction = False
    _validate_gym_multi_trace_capability(config)


def test_both_sides_must_opt_in():
    config = _config()
    config.grpo.gym_multi_trace = False
    with pytest.raises(ValueError, match="requires grpo.gym_multi_trace"):
        _validate_gym_multi_trace_capability(config)
    config.env["nemo_gym"]["token_id_capture"]["delivery"] = "main_chain"
    _validate_gym_multi_trace_capability(config)


@pytest.mark.parametrize("logical_count", [None, 8])
def test_policy_forwards_logical_scheduler_count_only_when_requested(logical_count):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from nemo_rl.models.policy.lm_policy import Policy

    worker_group = SimpleNamespace(
        run_all_workers_sharded_data=Mock(return_value=[]),
        get_all_worker_results=Mock(
            return_value=[
                {
                    "global_loss": torch.tensor(0.0),
                    "grad_norm": torch.tensor(1.0),
                    "all_mb_metrics": {},
                }
            ]
        ),
    )
    policy = SimpleNamespace(
        cfg={
            "megatron_cfg": {"enabled": True},
            "train_global_batch_size": 8,
            "train_micro_batch_size": 1,
        },
        _shard_for_train=Mock(return_value=[]),
        _report_sharded_payload=Mock(),
        flops_tracker=None,
        worker_group=worker_group,
    )
    data = BatchedDataDict({"input_ids": torch.zeros(12, 2, dtype=torch.long)})
    Policy.train(policy, data, Mock(), gbs=12, scheduler_step_samples=logical_count)
    kwargs = worker_group.run_all_workers_sharded_data.call_args.kwargs["common_kwargs"]
    assert kwargs["gbs"] == 12
    if logical_count is None:
        assert "scheduler_step_samples" not in kwargs
    else:
        assert kwargs["scheduler_step_samples"] == 8
