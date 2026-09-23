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
"""The upstream-worker adapter preserves trajectory coverage and gradients."""

import pytest
import torch
from test_fast_selection import confidence_batch, make_schedule

from just_grpo.algorithms.block_just_grpo import (
    BlockJustGRPOSchedule,
    select_low_confidence_tokens,
)
from unittest.mock import Mock

pytest.importorskip("megatron.bridge")
from just_grpo.diffusion.megatron_diffusion_policy import (
    MegatronDiffusionPolicyWorkerImpl,
)

from megatron.core import parallel_state
from nemo_rl.models.megatron.data import ProcessedMicrobatch
from just_grpo.diffusion.diffusion_processors import (
    DiffusionLossPostProcessor,
    DiffusionLogprobsPostProcessor,
    prepare_diffusion_microbatch,
    selected_diffusion_logprobs,
)


@pytest.fixture(autouse=True)
def tensor_parallel_stubs(monkeypatch):
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: None)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: None)


def processed(trajectory):
    return ProcessedMicrobatch(
        data_dict=trajectory,
        input_ids=trajectory["input_ids"],
        input_ids_cp_sharded=trajectory["input_ids"],
        position_ids=trajectory["position_ids"],
        attention_mask=None,
        packed_seq_params=None,
        cu_seqlens_padded=None,
        original_seq_length=trajectory["input_ids"].shape[1],
    )


def make_diffusion_schedule(data, fraction):
    schedule = make_schedule(data)
    return BlockJustGRPOSchedule(
        data,
        schedule.config,
        pad_token_id=0,
        selected_positions_per_block=1 if fraction < 1 else None,
    )


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_schedule_yields_one_level_at_a_time_with_aligned_rollout_fields(
    monkeypatch, fraction
):
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    data["prev_logprobs"] = torch.randn_like(data["token_mask"].float())
    data["reference_policy_logprobs"] = data["prev_logprobs"] - 0.3
    schedule = make_diffusion_schedule(data, fraction)
    generated = []
    generate = schedule.generate_single_trajectory

    def track(step):
        generated.append(step)
        return generate(step)

    monkeypatch.setattr(schedule, "generate_single_trajectory", track)

    iterator = schedule.iter_levels(data)
    assert generated == []
    coverage = torch.zeros_like(data["token_mask"])
    for index in range(schedule.num_steps):
        micro = next(iterator)
        assert micro.size == data.size  # Never an expanded batch.
        assert generated == list(range(index + 1))
        for row, sample in enumerate(micro["sample_indices"]):
            active = micro["token_mask"][row]
            positions = micro["original_positions"][row][active]
            coverage[sample, positions] += 1
            for key in (
                "prev_logprobs",
                "reference_policy_logprobs",
                "generation_logprobs",
            ):
                torch.testing.assert_close(
                    micro[key][row][active], data[key][sample, positions]
                )
    with pytest.raises(StopIteration):
        next(iterator)
    torch.testing.assert_close(coverage.bool(), data["training_token_mask"])
    assert coverage.max() == 1


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_selected_same_position_scores_and_gradients_match_dense(fraction):
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    trajectory = next(make_diffusion_schedule(data, fraction).return_schedule(2))
    torch.manual_seed(17)
    clean = trajectory["input_ids"]
    logits = torch.randn(*clean.shape, 32, requires_grad=True)

    micro = prepare_diffusion_microbatch(processed(trajectory), mask_token_id=31)
    torch.testing.assert_close(
        micro.input_ids[:, : clean.shape[1]],
        clean.masked_fill(trajectory["masked_indices"], 31),
    )
    torch.testing.assert_close(micro.input_ids[:, clean.shape[1] :], clean)
    torch.testing.assert_close(
        micro.position_ids[:, : clean.shape[1]], micro.position_ids[:, clean.shape[1] :]
    )
    assert micro.data_dict is trajectory
    active = trajectory["token_mask"]
    actual = selected_diffusion_logprobs(
        torch.cat([logits, logits], dim=1) / 0.7, trajectory
    )
    expected = (logits / 0.7).log_softmax(-1).gather(-1, clean[..., None]).squeeze(-1)
    torch.testing.assert_close(actual[active], expected[active])
    actual.sum().backward()
    actual_grad = logits.grad.clone()
    logits.grad = None
    expected[active].sum().backward()
    torch.testing.assert_close(actual_grad, logits.grad)
    assert not actual_grad[~active].any()


def test_empty_reveal_retains_backward_graph():
    data = confidence_batch()
    trajectory = next(make_schedule(data).return_schedule(2))
    trajectory["token_mask"].zero_()
    logits = torch.randn(2, 24, 32, requires_grad=True)
    values = selected_diffusion_logprobs(logits, trajectory)
    values.sum().backward()
    assert logits.grad is not None
    assert not logits.grad.any()


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_standard_forward_and_diffusion_processors_match_core_loss(
    monkeypatch, fraction
):
    pytest.importorskip("megatron.bridge")
    from megatron.core import parallel_state
    from nemo_rl.models.megatron.train import forward_with_post_processing_fn
    from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: None)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: None)
    # LossPostProcessor imports the same functions directly.
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_rank", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_tensor_model_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_context_parallel_group", lambda: None
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.train.get_context_parallel_world_size", lambda: 1
    )
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    data["generation_logprobs"] = torch.zeros_like(data["token_mask"].float()) - 3
    data["prev_logprobs"] = data["generation_logprobs"].clone()
    data["reference_policy_logprobs"] = data["prev_logprobs"] - 0.1
    data["advantages"] = torch.ones_like(data["prev_logprobs"])
    schedule = make_diffusion_schedule(data, fraction)
    loss_fn = ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.01))
    cfg = {"sequence_packing": {"enabled": False}}
    params = TrainingSamplingParams(temperature=0.7, top_k=None, top_p=1.0)
    processor = DiffusionLossPostProcessor(
        loss_fn,
        cfg,
        num_microbatches=schedule.num_steps,
    )
    scoring = DiffusionLogprobsPostProcessor(cfg, sampling_params=params)
    torch.manual_seed(13)
    logits = torch.randn(2, 24, 32, requires_grad=True)
    actual_losses, expected_losses = [], []
    n_seqs = data["sample_mask"].sum()
    n_toks = data["training_token_mask"].sum()
    for index, trajectory in enumerate(schedule.return_schedule(2)):
        micro = schedule.prepare_trajectory(index, data)
        prepared = prepare_diffusion_microbatch(processed(micro), mask_token_id=31)
        output, callback = forward_with_post_processing_fn(
            iter([prepared]),
            lambda **kwargs: logits.clone(),
            processor,
            global_valid_seqs=n_seqs,
            global_valid_toks=n_toks,
            sampling_params=params,
        )
        score_output, score_callback = forward_with_post_processing_fn(
            iter([prepared]),
            lambda **kwargs: logits.clone(),
            scoring,
            sampling_params=params,
        )
        values = score_callback(score_output)[1]["logprobs"]
        torch.testing.assert_close(values, selected_diffusion_logprobs(output, micro))
        # Undo the scaling that MCore's schedule applies, exactly as a real update.
        actual_losses.append(callback(output)[0] / schedule.num_steps)
        expected_losses.append(
            loss_fn(
                next_token_logprobs=values,
                data=micro,
                shift_labels=False,
                global_valid_seqs=n_seqs,
                global_valid_toks=n_toks,
            )[0]
        )
    actual, expected = sum(actual_losses), sum(expected_losses)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_worker_accumulates_levels_before_one_optimizer_step(monkeypatch, fraction):
    data = confidence_batch()
    data["training_token_mask"] = select_low_confidence_tokens(
        data, fraction=fraction, block_size=4
    )
    schedule = make_diffusion_schedule(data, fraction)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    worker = Mock()
    worker._schedule.return_value = schedule
    worker.diffusion.max_generation_kl = 0.05
    counts = {"tokens": 0.0, "sequences": 0.0}

    def accumulate(level):
        # Mirror native streaming counts, including its AR first-column slice.
        counts["tokens"] += float(
            (level["token_mask"][:, 1:] * level["sample_mask"][:, None]).sum()
        )
        counts["sequences"] += float(level["sample_mask"].sum())

    def finish():
        return {
            "all_mb_metrics": {
                "gen_kl_error": [0.001],
                "global_valid_toks": [counts["tokens"]],
                "global_valid_seqs": [counts["sequences"]],
            }
        }

    worker.train_microbatch.side_effect = accumulate
    worker.finish_train_step.side_effect = finish
    result = MegatronDiffusionPolicyWorkerImpl.train(worker, data=data, loss_fn=Mock())
    worker.begin_train_step.assert_called_once()
    assert worker.train_microbatch.call_count == schedule.num_steps
    worker.finish_train_step.assert_called_once_with()
    worker.abort_train_step.assert_not_called()
    expected_tokens = float(
        (data["training_token_mask"] * data["sample_mask"][:, None]).sum()
    )
    assert counts["tokens"] == expected_tokens
    assert result["all_mb_metrics"]["global_valid_toks"] == [expected_tokens]
    assert result["all_mb_metrics"]["global_valid_seqs"] == [
        float(data["sample_mask"].sum())
    ]
    assert result["all_mb_metrics"]["num_valid_samples"] == [data.size]


@pytest.mark.parametrize("failure", ["train_microbatch", "finish_train_step"])
def test_training_failure_aborts_open_step(monkeypatch, failure):
    data = confidence_batch()
    data["training_token_mask"] = data["token_mask"].bool()
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda: None)
    worker = Mock()
    worker._schedule.return_value = make_schedule(data)
    getattr(worker, failure).side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        MegatronDiffusionPolicyWorkerImpl.train(worker, data=data, loss_fn=Mock())
    worker.abort_train_step.assert_called_once_with()
    if failure == "train_microbatch":
        worker.finish_train_step.assert_not_called()


@pytest.mark.parametrize("reference", [False, True])
def test_worker_scoring_uses_schedule_and_native_reference_context(
    monkeypatch, reference
):
    from contextlib import contextmanager
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    data = confidence_batch()
    schedule = make_schedule(data)
    worker = object.__new__(MegatronDiffusionPolicyWorkerImpl)
    worker._schedule = Mock(return_value=schedule)
    active = []
    calls = []

    @contextmanager
    def reference_weights():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    worker.use_reference_model = reference_weights

    def score(self, *, data, **kwargs):
        assert bool(active) == reference
        calls.append(data)
        return {"logprobs": data["target_ids"].float()}

    monkeypatch.setattr(MegatronPolicyWorkerImpl, "get_logprobs", score)
    result = (
        worker.get_reference_policy_logprobs(data=data)["reference_logprobs"]
        if reference
        else worker.get_logprobs(data=data)["logprobs"]
    )
    assert len(calls) == schedule.num_steps
    assert not active
    torch.testing.assert_close(result, data["input_ids"].float() * data["token_mask"])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Megatron worker schedules use CUDA"
)
def test_worker_preserves_fixed_model_canvas_for_short_rollouts():
    from types import SimpleNamespace

    data = confidence_batch()
    worker = object.__new__(MegatronDiffusionPolicyWorkerImpl)
    worker.diffusion = SimpleNamespace(
        training_token_fraction=0.25, schedule=make_schedule(data).config
    )
    worker.cfg = {"max_total_sequence_length": 32}
    worker.tokenizer = SimpleNamespace(pad_token_id=0)
    schedule = worker._schedule(data)
    assert schedule.base["input_ids"].shape == (data.size, 32)
    assert schedule.num_steps == 1
    assert not schedule.base["token_mask"][:, data["input_ids"].shape[1] :].any()
