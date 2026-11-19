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

import copy
import math
from collections import defaultdict

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.training_engine_adapter.config import apply_recipe_layout


def _policy_config() -> dict:
    """A policy config whose layout and token budgets are another recipe's."""
    return {
        "max_total_sequence_length": 16384,
        "make_sequence_length_divisible_by": 16,
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 2,
            "context_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "sequence_parallel": True,
        },
        "sequence_packing": {
            "enabled": True,
            "train_mb_tokens": 16384,
            "logprob_mb_tokens": 16384,
        },
        "training_engine": {
            "recipe": "some_recipe",
            "recipe_kwargs": {},
            "checkpoint_path": "/ckpt",
            "seq_length": 32768,
            "overrides": {},
        },
    }


def _recipe_layout() -> dict:
    """The layout of a recipe with context parallelism 8, which needs samples padded to 16."""
    return {
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
    }


def test_policy_config_takes_the_layout_and_token_budget_from_the_recipe():
    config = _policy_config()

    apply_recipe_layout(config, _recipe_layout(), sample_pad_multiple=16)

    assert config["megatron_cfg"] == {"enabled": True, **_recipe_layout()}
    assert config["sequence_packing"] == {
        "enabled": True,
        "train_mb_tokens": 32768,
        "logprob_mb_tokens": 32768,
    }


@pytest.mark.parametrize(
    "path, value, match",
    [
        (("megatron_cfg", "enabled"), False, "megatron_cfg.enabled"),
        (("sequence_packing", "enabled"), False, "sequence_packing.enabled"),
        (("training_engine", "recipe_kwargs"), {"seq_length": 1}, "recipe_kwargs"),
        (("max_total_sequence_length",), 65536, "max_total_sequence_length"),
        (("make_sequence_length_divisible_by",), 24, "multiple of 16"),
    ],
)
def test_policy_config_rejects_what_the_recipe_cannot_decide(path, value, match):
    config = copy.deepcopy(_policy_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        apply_recipe_layout(config, _recipe_layout(), sample_pad_multiple=16)


def test_policy_config_rejects_a_recipe_with_pipeline_parallelism():
    layout = {**_recipe_layout(), "pipeline_model_parallel_size": 2}
    with pytest.raises(NotImplementedError, match="pipeline"):
        apply_recipe_layout(_policy_config(), layout, sample_pad_multiple=16)


def _training_batch() -> BatchedDataDict:
    """Two samples of 6 and 4 tokens; the second row's last two columns are padding."""
    torch.manual_seed(0)
    token_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=torch.float32
    )
    return BatchedDataDict(
        {
            "input_ids": torch.arange(12).reshape(2, 6),
            "input_lengths": torch.tensor([6, 4]),
            "token_mask": token_mask,
            "sample_mask": torch.ones(2),
            "advantages": torch.randn(2, 6),
            "prev_logprobs": -torch.rand(2, 6),
            "generation_logprobs": -torch.rand(2, 6),
            "reference_policy_logprobs": -torch.rand(2, 6),
        }
    )


@pytest.mark.mcore
def test_clipped_pg_loss_sums_to_the_native_loss():
    """The engine's loss sum, divided by the token count, equals NeMo RL's loss."""
    from types import SimpleNamespace

    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.algorithms.loss.loss_functions import KL_PENALTY_KEY
    from nemo_rl.models.policy.training_engine_adapter.engine import (
        NeMoRLTrainingEngine,
    )

    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio=2,
        )
    )
    data = _training_batch()
    current = -torch.rand(2, 5)
    num_tokens = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).sum()
    native_loss, native_metrics = loss_fn(
        next_token_logprobs=current,
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=num_tokens,
    )

    # One packed row holding both samples, padded from 6 and 4 tokens to 8 and 4. Each
    # position predicts the next token of its sample; a sample's last position and the
    # padding predict nothing and their model output is arbitrary.
    junk = torch.tensor([-9.0])
    logprobs = torch.cat([current[0], junk, junk, junk, current[1, :3], junk])
    batch = SimpleNamespace(
        sample_inputs=[data],
        packed_seq_params=SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 8, 12])),
    )
    engine = SimpleNamespace(process_groups=SimpleNamespace(cp=None))
    extrema = defaultdict(list)
    loss_sum, sums = NeMoRLTrainingEngine.clipped_pg_loss(
        engine,
        batch,
        SimpleNamespace(logprobs=logprobs.unsqueeze(0)),
        loss_fn=loss_fn,
        extrema=extrema,
    )

    assert loss_sum / num_tokens == pytest.approx(float(native_loss), abs=1e-5)
    assert float(sums[KL_PENALTY_KEY]) / float(num_tokens) == pytest.approx(
        native_metrics[KL_PENALTY_KEY], abs=1e-5
    )
    # The extrema are not sums: they are handed out separately, one entry per microbatch.
    assert "probs_ratio_min" not in sums
    assert extrema["probs_ratio_min"] == [
        pytest.approx(native_metrics["probs_ratio_min"])
    ]
    assert extrema["probs_ratio_max"] == [
        pytest.approx(native_metrics["probs_ratio_max"])
    ]


def _engine_without_model(seq_length: int = 16, pad_multiple: int = 4):
    """The adapter engine with only what its data hooks read: no model, one rank."""
    from types import SimpleNamespace

    from nemo_rl.models.policy.training_engine_adapter.engine import (
        NeMoRLTrainingEngine,
    )

    engine = NeMoRLTrainingEngine.__new__(NeMoRLTrainingEngine)
    engine._recipe = SimpleNamespace(
        model=SimpleNamespace(seq_length=seq_length), pad_to_max_length=False
    )
    engine._runtime = SimpleNamespace(pg_collection=SimpleNamespace(cp=None))
    engine.sample_pad_multiple = pad_multiple
    return engine


def _scoring_batch() -> BatchedDataDict:
    """Three samples of 4, 6 and 3 tokens."""
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [
                    [10, 11, 12, 13, 0, 0],
                    [20, 21, 22, 23, 24, 25],
                    [30, 31, 32, 0, 0, 0],
                ]
            ),
            "input_lengths": torch.tensor([4, 6, 3]),
        }
    )


def _without_context_parallelism(monkeypatch):
    """NeMo RL's packing reads the context-parallel layout from Megatron's process state."""
    import nemo_rl.models.megatron.data as megatron_data

    monkeypatch.setattr(megatron_data, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(megatron_data, "get_context_parallel_rank", lambda: 0)


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine packs on the GPU")
def test_engine_packs_the_planned_microbatches_and_scores_map_back(monkeypatch):
    _without_context_parallelism(monkeypatch)
    engine = _engine_without_model()
    # The plan: samples 0 and 1 in one microbatch, sample 2 in the next.
    data = _with_plan(_scoring_batch(), [[[0, 2], [2, 3]]])

    first, second = engine.preprocess(data, training=False)

    # NeMo RL packs whole samples, each padded to a multiple of 4: 4 -> 4 and 6 -> 8
    # tokens, then 3 -> 4. A sample's last token and the padding have no target.
    assert first.input_ids.tolist() == [[10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 0, 0]]
    assert first.target_ids.tolist() == [[11, 12, 13, 0, 21, 22, 23, 24, 25, 0, 0, 0]]
    assert first.loss_mask.tolist() == [[1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]]
    assert first.position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 0]]
    assert first.padding_mask.tolist() == [[False] * 10 + [True] * 2]
    assert second.input_ids.tolist() == [[30, 31, 32, 0]]
    assert first.packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 4, 12]
    assert second.packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 4]

    # One score per packed position, equal to its target id. They come back in NeMo RL's
    # layout: token i's score at position i, 0 for the first token and the padding.
    scores = [mb.target_ids[0].float() for mb in (first, second)]
    output = engine.postprocess_forward_only(
        [first, second], scores, gather_output=True
    )
    assert output.tolist() == [
        [0.0, 11.0, 12.0, 13.0, 0.0, 0.0],
        [0.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        [0.0, 31.0, 32.0, 0.0, 0.0, 0.0],
    ]


@pytest.mark.mcore
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the engine packs on the GPU")
def test_engine_takes_the_training_loss_mask_from_the_batch(monkeypatch):
    _without_context_parallelism(monkeypatch)
    data = _with_plan(_training_batch(), [[[0, 2]]])

    (microbatch,) = _engine_without_model().preprocess(data, training=True)

    # Samples of 6 and 4 tokens, padded to 8 and 4. Position i carries the mask of
    # token i + 1: [0, 1, 1, 1, 1] of the first sample and [1, 1, 1] of the second.
    assert microbatch.loss_mask.tolist() == [[0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]]
    # The loss's own inputs travel unpacked, as one entry for the whole microbatch.
    (rows,) = microbatch.sample_inputs
    assert torch.equal(rows["advantages"].cpu(), data["advantages"])
    assert microbatch.mapping is None


@pytest.mark.mcore
def test_engine_rejects_an_empty_planned_microbatch():
    data = _with_plan(_scoring_batch(), [[[0, 0], [0, 3]]])
    with pytest.raises(ValueError, match="no samples"):
        _engine_without_model().preprocess(data, training=False)


def _with_plan(data: BatchedDataDict, global_batches: list[list[list[int]]]):
    """Attach a packing plan: per global batch, the [start, end) rows of each microbatch."""
    data.micro_batch_indices = global_batches
    data.micro_batch_lengths = [[8] * len(batch) for batch in global_batches]
    data.elem_counts_per_gb = [batch[-1][1] for batch in global_batches]
    return data


def _loss_fn():
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn

    return ClippedPGLossFn(ClippedPGLossConfig())


class _FakeEngine:
    """Stands in for the engine: 2.0 loss and 0.5 KL per loss token."""

    def __init__(self, dp_rank: int = 0):
        self.device = "cuda"
        self.trainable = True
        self.calls = []
        self.saved = None
        self.parallel_info = {"dp_rank": dp_rank, "dp_size": 1}

    def forward_backward(self, data, loss_fn, loss_kwargs):
        from types import SimpleNamespace

        loss_mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        tokens = int(loss_mask[:, 1:].sum())
        self.calls.append(
            ("forward_backward", len(data.micro_batch_indices[0]), tokens)
        )
        # Ratio extrema of this call, as the loss method hands them out.
        loss_kwargs["extrema"]["probs_ratio_min"].append(1.0 / tokens)
        loss_kwargs["extrema"]["probs_ratio_max"].append(float(tokens))
        return SimpleNamespace(
            loss_sum=2.0 * tokens,
            num_loss_tokens=tokens,
            metrics={"kl_penalty": 0.5 * tokens},
        )

    def optim_step(self, adam_params):
        self.calls.append(("optim_step", adam_params.learning_rate))
        return {"update_successful": True, "grad_norm": 0.25}

    def zero_grad(self):
        self.calls.append(("zero_grad",))

    def cuda(self):
        self.device = "cuda"

    def cpu(self):
        self.device = "cpu"

    def save_state(self, path, user_state, async_save):
        self.saved = (path, user_state, self.device)
        self.calls.append(("save_state", async_save))

    def wait_for_save(self):
        self.calls.append(("wait_for_save",))


def _worker(dp_rank: int = 0):
    """A worker around a fake engine, skipping the constructor that builds a real one."""
    from nemo_rl.models.policy.workers.training_engine_policy_worker import (
        TrainingEnginePolicyWorkerImpl,
    )

    worker = TrainingEnginePolicyWorkerImpl.__new__(TrainingEnginePolicyWorkerImpl)
    worker.engine = _FakeEngine(dp_rank)
    # One replica: its values are the totals.
    worker._sum_over_replicas = lambda values: values
    worker.cfg = {
        "train_global_batch_size": 2,
        "megatron_cfg": {
            "optimizer": {
                "lr": 1.0,
                "min_lr": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
            },
            "scheduler": {
                "lr_warmup_iters": 2,
                "lr_warmup_init": 0.0,
                "lr_decay_style": "constant",
                "start_weight_decay": 0.01,
                "end_weight_decay": 0.01,
                "weight_decay_incr_style": "constant",
            },
        },
    }
    worker._lr_decay_iters = None
    worker._weight_decay_iters = 10
    worker._train_calls = 0
    worker._train_state = None
    worker._async_save = False
    return worker


@pytest.mark.mcore
def test_streaming_train_step_adds_chunks_into_one_update():
    worker = _worker()
    data = _training_batch()  # 4 loss tokens in the first sample, 3 in the second

    worker.begin_train_step(loss_fn=_loss_fn())
    worker.train_microbatch(data=_with_plan(data.slice(0, 1), [[[0, 1]]]))
    worker.train_microbatch(data=_with_plan(data.slice(1, 2), [[[0, 1]]]))
    result = worker.finish_train_step()

    assert worker.engine.calls == [
        ("forward_backward", 1, 4),
        ("forward_backward", 1, 3),
        ("optim_step", 0.0),
    ]
    metrics = result["all_mb_metrics"]
    assert metrics["global_valid_toks"] == [7]
    assert metrics["global_valid_seqs"] == [2]
    assert metrics["loss"] == [pytest.approx(2.0)]
    assert metrics["kl_penalty"] == [pytest.approx(0.5)]
    assert float(result["global_loss"]) == pytest.approx(2.0)
    assert result["grad_norm"].tolist() == [0.25]
    assert worker._train_calls == 1
    assert worker._train_state is None


@pytest.mark.mcore
def test_streaming_train_step_guards_and_abort():
    worker = _worker()
    # Both samples in one update, each in its own microbatch.
    data = _with_plan(_training_batch(), [[[0, 1], [1, 2]]])
    loss_fn = _loss_fn()

    with pytest.raises(RuntimeError, match="open train step"):
        worker.train_microbatch(data=data)
    with pytest.raises(RuntimeError, match="open train step"):
        worker.finish_train_step()
    worker.abort_train_step()  # nothing open: no effect
    assert worker.engine.calls == []

    worker.begin_train_step(loss_fn=loss_fn)
    with pytest.raises(RuntimeError, match="already open"):
        worker.begin_train_step(loss_fn=loss_fn)
    worker.train_microbatch(data=data)
    assert worker.engine.calls[-1] == ("forward_backward", 2, 7)
    worker.abort_train_step()
    assert worker.engine.calls[-1] == ("zero_grad",)
    assert worker._train_calls == 0
    worker.begin_train_step(loss_fn=loss_fn)  # a new update can open after an abort


@pytest.mark.mcore
def test_train_step_rejects_an_unsupported_loss():
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn

    worker = _worker()
    with pytest.raises(NotImplementedError, match="ClippedPGLossFn only"):
        worker.begin_train_step(loss_fn=object())
    sequence_level = ClippedPGLossFn(ClippedPGLossConfig(token_level_loss=False))
    with pytest.raises(NotImplementedError, match="token_level_loss=false"):
        worker.begin_train_step(loss_fn=sequence_level)
    data = _with_plan(_training_batch(), [[[0, 2]]])
    with pytest.raises(NotImplementedError, match="token_level_loss=false"):
        worker.train(data, sequence_level, eval_mode=True)


@pytest.mark.mcore
def test_a_policy_without_an_optimizer_cannot_train():
    worker = _worker()
    worker.engine.trainable = False
    with pytest.raises(RuntimeError, match="without an optimizer"):
        worker.begin_train_step(loss_fn=_loss_fn())


@pytest.mark.mcore
def test_train_runs_one_update_per_global_batch_on_one_schedule_position():
    worker = _worker()
    worker._train_calls = 1  # halfway through the two-call warmup
    # Two global batches of one sample each.
    data = _with_plan(_training_batch(), [[[0, 1]], [[0, 1]]])

    result = worker.train(data, _loss_fn(), gbs=1)

    assert worker.engine.calls == [
        ("forward_backward", 1, 4),
        ("optim_step", 0.5),
        ("forward_backward", 1, 3),
        ("optim_step", 0.5),
    ]
    assert result["all_mb_metrics"]["global_valid_toks"] == [4, 3]
    assert result["all_mb_metrics"]["lr"] == [0.5, 0.5]
    assert result["grad_norm"].tolist() == [0.25, 0.25]
    assert worker._train_calls == 2


@pytest.mark.mcore
def test_only_the_first_replica_reports_the_metric_lists():
    """The metrics are totals over all ranks and the driver adds the replicas' lists."""
    data = _with_plan(_training_batch(), [[[0, 2]]])

    result = _worker(dp_rank=1).train(data, _loss_fn(), gbs=2)

    # Only the ratio extrema, which are this replica's own.
    assert result["all_mb_metrics"] == {
        "probs_ratio_min": [pytest.approx(1 / 7)],
        "probs_ratio_max": [7.0],
    }
    assert float(result["global_loss"]) == pytest.approx(2.0)
    assert result["grad_norm"].tolist() == [0.25]


@pytest.mark.mcore
def test_ratio_extrema_are_reduced_over_the_microbatches_of_an_update():
    worker = _worker()
    data = _training_batch()  # 4 loss tokens in the first sample, 3 in the second

    worker.begin_train_step(loss_fn=_loss_fn())
    worker.train_microbatch(data=_with_plan(data.slice(0, 1), [[[0, 1]]]))
    worker.train_microbatch(data=_with_plan(data.slice(1, 2), [[[0, 1]]]))
    metrics = worker.finish_train_step()["all_mb_metrics"]

    assert metrics["probs_ratio_min"] == [pytest.approx(1 / 4)]
    assert metrics["probs_ratio_max"] == [4.0]


@pytest.mark.mcore
def test_weight_decay_follows_the_schedule():
    worker = _worker()
    scheduler = worker.cfg["megatron_cfg"]["scheduler"]
    scheduler.update(
        start_weight_decay=0.0, end_weight_decay=0.1, weight_decay_incr_style="linear"
    )

    decay = {}
    for calls in (0, 5, 10, 11):
        worker._train_calls = calls
        decay[calls] = worker._adam_params().weight_decay
    assert decay == {
        0: 0.0,
        5: pytest.approx(0.05),
        10: pytest.approx(0.1),
        11: 0.1,
    }

    scheduler["weight_decay_incr_style"] = "cosine"
    worker._train_calls = 5
    assert worker._adam_params().weight_decay == pytest.approx(0.05)
    worker._train_calls = 2
    assert worker._adam_params().weight_decay == pytest.approx(
        0.05 * (math.cos(math.pi * 0.8) + 1.0)
    )


class _EvalEngine:
    """Scores every sample with fixed logprobs; training calls are not expected."""

    def __init__(self, logprobs: torch.Tensor):
        self.logprobs = logprobs
        self.parallel_info = {"dp_rank": 0, "dp_size": 1}

    def forward_only(self, data, gather_output):
        return self.logprobs[: data.size]


@pytest.mark.mcore
def test_eval_mode_returns_the_native_loss_and_updates_nothing():
    from nemo_rl.algorithms.loss.loss_functions import KL_PENALTY_KEY

    worker = _worker()
    data = _with_plan(_training_batch(), [[[0, 2]]])
    loss_fn = _loss_fn()
    # [B, S] scores as the engine returns them: the first token's is 0.
    logprobs = torch.cat([torch.zeros(2, 1), -torch.rand(2, 5)], dim=1)
    worker.engine = _EvalEngine(logprobs)
    num_tokens = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).sum()
    native_loss, native_metrics = loss_fn(
        next_token_logprobs=logprobs[:, 1:],
        data=data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=num_tokens,
    )

    result = worker.train(data, loss_fn, eval_mode=True, gbs=2)

    metrics = result["all_mb_metrics"]
    assert float(result["global_loss"]) == pytest.approx(float(native_loss), abs=1e-5)
    assert metrics["loss"] == [pytest.approx(float(native_loss), abs=1e-5)]
    assert metrics[KL_PENALTY_KEY] == [
        pytest.approx(native_metrics[KL_PENALTY_KEY], abs=1e-5)
    ]
    assert metrics["global_valid_toks"] == [7]
    assert metrics["global_valid_seqs"] == [2]
    assert metrics["probs_ratio_max"] == [
        pytest.approx(native_metrics["probs_ratio_max"])
    ]
    assert result["grad_norm"].tolist() == []
    assert worker._train_calls == 0


class _ScoringEngine:
    """Returns each token's id as its score and records the plan it was given."""

    def forward_only(self, data, gather_output):
        self.plan = [end - start for start, end in data.micro_batch_indices[0]]
        scores = data["input_ids"].float()
        scores[:, 0] = 0.0
        return scores


@pytest.mark.mcore
def test_scoring_follows_the_plan_and_keeps_the_row_order():
    worker = _worker()
    worker.engine = _ScoringEngine()
    worker.reference_engine = _ScoringEngine()
    data = _with_plan(_scoring_batch(), [[[0, 2], [2, 3]]])

    logprobs = worker.get_logprobs(data=data)["logprobs"]
    reference = worker.get_reference_policy_logprobs(data=data, micro_batch_size=None)

    assert worker.engine.plan == [2, 1]
    assert logprobs.tolist() == [
        [0.0, 11.0, 12.0, 13.0, 0.0, 0.0],
        [0.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        [0.0, 31.0, 32.0, 0.0, 0.0, 0.0],
    ]
    assert torch.equal(reference["reference_logprobs"], logprobs)


@pytest.mark.mcore
def test_save_checkpoint_keeps_the_schedule_position_and_residency():
    worker = _worker()
    worker._train_calls = 5
    worker.engine.cpu()

    worker.save_checkpoint("/ckpt/weights", optimizer_path="/ckpt/optimizer")

    assert worker.engine.saved == ("/ckpt/weights", {"train_calls": 5}, "cuda")
    assert worker.engine.device == "cpu"
    with pytest.raises(NotImplementedError, match="save_optimizer"):
        worker.save_checkpoint("/ckpt/weights", optimizer_path=None)


@pytest.mark.mcore
def test_async_save_is_passed_to_the_engine_and_finalized_by_waiting():
    worker = _worker()
    assert worker.poll_async_save()
    worker._async_save = True

    worker.save_checkpoint("/ckpt/weights", optimizer_path="/ckpt/optimizer")
    worker.finalize_async_save()

    assert worker.engine.calls == [("save_state", True), ("wait_for_save",)]
    with pytest.raises(NotImplementedError, match="defer_async_checkpoint_finalize"):
        worker.poll_async_save()


@pytest.mark.mcore
def test_refit_rejects_kv_scales_on_both_transports():
    worker = _worker()

    with pytest.raises(NotImplementedError, match="KV-cache scales"):
        worker.stream_weights_via_ipc_zmq(kv_scales={"scale": 1.0})
    with pytest.raises(NotImplementedError, match="KV-cache scales"):
        worker.broadcast_weights_for_collective(kv_scales={"scale": 1.0})
