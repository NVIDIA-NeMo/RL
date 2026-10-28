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

"""Unit tests for Automodel pipeline-parallel batch planning and the loss adapter."""

from typing import Any

import pytest
import torch
from torch import nn

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

try:
    from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
    from nemo_rl.models.automodel.pipeline_batch import (
        ExecutionBatch,
        plan_execution_batches,
    )
    from nemo_rl.models.automodel.pipeline_parallel import (
        PipelineCallContext,
        PipelineOutputAdapter,
    )
    from nemo_rl.models.automodel.train import (
        LogprobsPostProcessor,
        LossPostProcessor,
    )

    NEMO_AUTOMODEL_AVAILABLE = True
except ImportError:
    NEMO_AUTOMODEL_AVAILABLE = False

pytestmark = [
    pytest.mark.automodel,
    pytest.mark.skipif(
        not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available"
    ),
]

_SEQ_LEN = 6
_VOCAB = 11


class _SumLogitsLoss:
    """Loss = masked sum of logits / global token count, with a metric."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGIT

    def __call__(
        self,
        logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
        loss = (logits.sum(-1) * mask).sum() / global_valid_toks
        return loss, {
            "loss": loss.item(),
            "num_valid_samples": data["sample_mask"].sum().item(),
        }


def _microbatch(first_row: int, micro_batch_size: int) -> ProcessedMicrobatch:
    input_ids = (
        torch.arange(micro_batch_size * _SEQ_LEN).reshape(micro_batch_size, -1)
        + first_row
    ) % _VOCAB
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.full((micro_batch_size,), _SEQ_LEN),
            "token_mask": torch.ones(micro_batch_size, _SEQ_LEN),
            "sample_mask": torch.ones(micro_batch_size),
        }
    )
    return ProcessedMicrobatch(
        data_dict=data,
        processed_inputs=ProcessedInputs(input_ids=input_ids, seq_len=_SEQ_LEN),
        original_batch_size=micro_batch_size,
        original_seq_len=_SEQ_LEN,
    )


def _execution_batch(
    first_sample_id: int, *, num_microbatches: int, num_real: int, mbs: int = 2
) -> ExecutionBatch:
    return ExecutionBatch(
        microbatches=[
            _microbatch(first_sample_id + i * mbs, mbs) for i in range(num_microbatches)
        ],
        num_real_microbatches=num_real,
        sample_ids=torch.arange(
            first_sample_id, first_sample_id + num_microbatches * mbs
        ),
        model_inputs={},
        seq_len=_SEQ_LEN,
    )


def _context(execution_batch: ExecutionBatch, mode: str) -> PipelineCallContext:
    if mode == "logprobs":
        post_processing_fn = LogprobsPostProcessor(cfg={})
    else:
        post_processing_fn = LossPostProcessor(
            loss_fn=_SumLogitsLoss(), cfg={}, cp_mesh=None, cp_size=1, dp_size=4
        )
    return PipelineCallContext(
        execution_batch=execution_batch,
        mode=mode,
        post_processing_fn=post_processing_fn,
        sampling_params=None,
        sequence_dim=1,
        global_valid_seqs=torch.tensor(8.0),
        global_valid_toks=torch.tensor(48.0),
        loss_scale=4.0,
        results={},
    )


def test_microbatch_index_maps_sample_ids():
    context = _context(
        _execution_batch(8, num_microbatches=4, num_real=4), mode="train"
    )
    assert context.microbatch_index(torch.tensor([8, 9])) == 0
    assert context.microbatch_index(torch.tensor([14, 15])) == 3
    for bad in ([9, 10], [16, 17], [10, 12], [0, 1]):
        with pytest.raises(RuntimeError, match="not one microbatch"):
            context.microbatch_index(torch.tensor(bad))


def test_adapter_train_returns_backward_scale_and_records_results():
    context = _context(_execution_batch(0, num_microbatches=2, num_real=1), "train")
    adapter = PipelineOutputAdapter()
    logits = torch.randn(2, _SEQ_LEN, _VOCAB, requires_grad=True)

    returned = adapter(logits, torch.tensor([0, 1]), context=context)
    expected = logits.sum() / 48.0
    torch.testing.assert_close(returned, expected * 4.0)
    loss, metrics = context.results[0]
    assert not loss.requires_grad
    torch.testing.assert_close(loss, expected.detach())
    assert metrics["num_valid_samples"] == 2

    # A repeated call for the same microbatch overwrites instead of adding.
    adapter(logits, torch.tensor([0, 1]), context=context)
    assert set(context.results) == {0}

    # Padding microbatches still return a graph for the schedule's backward,
    # but contribute nothing.
    padding = adapter(logits, torch.tensor([2, 3]), context=context)
    assert padding.requires_grad
    assert padding.item() == 0.0
    assert context.results[1][0].item() == 0.0


def test_adapter_eval_frees_logits():
    context = _context(_execution_batch(0, num_microbatches=1, num_real=1), "eval")
    logits = torch.randn(2, _SEQ_LEN, _VOCAB)
    expected = logits.sum() / 48.0

    returned = PipelineOutputAdapter()(logits, torch.tensor([0, 1]), context=context)
    torch.testing.assert_close(returned, expected)
    torch.testing.assert_close(context.results[0][0], expected)
    assert logits.untyped_storage().size() == 0


def test_adapter_logprobs_records_token_logprobs_and_frees_logits():
    execution_batch = _execution_batch(4, num_microbatches=2, num_real=2)
    context = _context(execution_batch, "logprobs")
    logits = torch.randn(2, _SEQ_LEN, _VOCAB)
    input_ids = execution_batch.microbatches[1].processed_inputs.input_ids
    expected = torch.log_softmax(logits[:, :-1], dim=-1).gather(
        -1, input_ids[:, 1:].unsqueeze(-1)
    )
    expected = torch.cat([torch.zeros(2, 1), expected.squeeze(-1)], dim=1)

    PipelineOutputAdapter()(logits.clone(), torch.tensor([6, 7]), context=context)
    torch.testing.assert_close(context.results[1], expected)


def test_build_pipeline_config_sets_nemo_rl_runtime_fields():
    from types import SimpleNamespace

    from torch.distributed.fsdp import MixedPrecisionPolicy

    from nemo_rl.models.automodel.setup import build_pipeline_config

    fsdp2_config = SimpleNamespace(
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )
    )
    config = {
        "train_micro_batch_size": 2,
        "max_total_sequence_length": 128,
        "automodel_cfg": {
            "pipeline": {
                "pp_schedule": "interleaved_1f1b",
                "microbatches_per_step": 4,
                "layers_per_stage": 2,
            }
        },
    }

    pipeline_config = build_pipeline_config(config, fsdp2_config)

    assert pipeline_config.pp_schedule == "Interleaved1F1B"
    assert pipeline_config.pp_microbatch_size == 2
    assert pipeline_config.pp_batch_size == 8
    assert pipeline_config.layers_per_stage == 2
    assert pipeline_config.dtype == torch.float32
    assert pipeline_config.scale_grads_in_schedule is False
    assert pipeline_config.pp_seq_len == 128
    assert isinstance(pipeline_config.loss_fn, PipelineOutputAdapter)


class _Forward(nn.Module):
    def forward(self, input_ids, **kwargs):
        return input_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_plan_execution_batches_pads_to_the_busiest_dp_rank(monkeypatch):
    def fake_all_reduce(tensor, op=None, group=None):
        # Another DP rank needs three schedule calls.
        tensor.fill_(3)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    num_samples = 5
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(num_samples * _SEQ_LEN).reshape(num_samples, -1),
            "input_lengths": torch.full((num_samples,), _SEQ_LEN),
            "token_mask": torch.ones(num_samples, _SEQ_LEN),
            "sample_mask": torch.ones(num_samples),
        }
    ).to("cuda")

    execution_batches = plan_execution_batches(
        data,
        model=_Forward(),
        micro_batch_size=2,
        microbatches_per_step=2,
        dp_group=None,
        tokenizer=None,
        cfg={"automodel_cfg": {"sequence_parallel": False}},
        allow_flash_attn_args=True,
    )

    assert [b.num_real_microbatches for b in execution_batches] == [2, 1, 0]
    assert [b.sample_ids.tolist() for b in execution_batches] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    rows = torch.cat([b.model_inputs["input_ids"] for b in execution_batches])
    torch.testing.assert_close(rows[:num_samples], data["input_ids"])
    # Padding rows copy sample 0 with both masks zeroed.
    torch.testing.assert_close(rows[num_samples:], data["input_ids"][[0] * 7])
    sample_mask = torch.cat(
        [
            mb.data_dict["sample_mask"]
            for b in execution_batches
            for mb in b.microbatches
        ]
    )
    assert sample_mask.tolist() == [1.0] * num_samples + [0.0] * 7
    token_mask = torch.cat(
        [mb.data_dict["token_mask"] for b in execution_batches for mb in b.microbatches]
    )
    assert token_mask[num_samples:].sum().item() == 0
    first = execution_batches[0]
    assert first.model_inputs["position_ids"].shape == (4, _SEQ_LEN)
    assert first.model_inputs["use_cache"] is False
    # The caller's data is untouched.
    assert data["sample_mask"].tolist() == [1.0] * num_samples


def _check_tq_coordinates(rank: int, world_size: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_rl.models.policy.workers.automodel_policy_worker import (
        AutomodelPolicyWorkerImpl,
    )

    # The Automodel mesh order (pp, dp_replicate, dp_shard, cp, tp) with PP=2,
    # DP=1, TP=2: both pipeline stages of the single DP rank are replicas.
    worker = object.__new__(AutomodelPolicyWorkerImpl)
    worker.device_mesh = init_device_mesh(
        "cuda",
        (2, 1, 1, 1, 2),
        mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"),
    )
    coords = worker._local_coords()
    assert coords == {
        "tensor_parallel": rank % 2,
        "context_parallel": 0,
        "pipeline_parallel": rank // 2,
    }
    # Exactly one rank of the DP replica writes back to TQ.
    leaders = [None] * world_size
    torch.distributed.all_gather_object(leaders, worker._is_replica_leader())
    assert leaders == [True, False, False, False]
    replica_group = worker._get_replica_group()
    assert torch.distributed.get_process_group_ranks(replica_group) == [0, 1, 2, 3]


def test_tq_coordinates_include_pipeline_stage(distributed_test_runner):
    distributed_test_runner(_check_tq_coordinates, world_size=4)
