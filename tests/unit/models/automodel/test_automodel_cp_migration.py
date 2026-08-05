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

"""Contract tests for the Automodel context-parallel API migration."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.algorithms.loss.interfaces import LossInputType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    get_cp_sharded_next_token_logprobs,
    get_distillation_topk_logprobs_from_logits,
)
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.models.automodel.train import (
    FullLogitsPostProcessor,
    LossPostProcessor,
    PreparedModelForward,
    ScorePostProcessor,
    forward_with_post_processing_fn,
    prepare_model_forward,
)


class _PermutationTokenLayout:
    """Small non-identity layout implementing Automodel's token verbs."""

    def __init__(self, order: torch.Tensor) -> None:
        self.order = order
        self.inverse_order = torch.argsort(order)

    def shard_token_tensor(
        self, tensor: torch.Tensor, *, seq_dim: int, fill: int
    ) -> torch.Tensor:
        del fill
        return tensor.index_select(seq_dim, self.order)

    def gather_token_tensor(
        self, tensor: torch.Tensor, *, seq_dim: int, trim: bool
    ) -> torch.Tensor:
        assert trim
        return tensor.index_select(seq_dim, self.inverse_order)


@pytest.mark.automodel
class TestPrepareModelForward:
    def test_cp1_keeps_canonical_batch_and_skips_sharder(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=4,
            attention_mask=attention_mask,
            position_ids=None,
        )

        with patch(
            "nemo_rl.models.automodel.train.ContextParallelSharder"
        ) as sharder_cls:
            prepared = prepare_model_forward(
                torch.nn.Identity(),
                processed_inputs,
                device_mesh=None,
                cp_size=1,
                padding_token_id=0,
                is_reward_model=False,
                allow_flash_attn_args=True,
            )

        sharder_cls.assert_not_called()
        assert prepared.cp_size == 1
        assert prepared.cp_sharder is None
        assert prepared.model_batch["input_ids"] is input_ids
        assert prepared.model_batch["attention_mask"] is attention_mask
        assert "position_ids" not in prepared.model_batch
        assert "labels" not in prepared.model_batch
        with prepared.model_context_factory():
            pass

    def test_cp2_clones_model_tensors_and_delegates_layout(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones_like(input_ids)
        processed_inputs = ProcessedInputs(
            input_ids=input_ids,
            seq_len=4,
            attention_mask=attention_mask,
            position_ids=None,
        )
        model = torch.nn.Identity()
        device_mesh = MagicMock()
        sharder = MagicMock()
        observed_batch: dict[str, object] = {}

        def shard(model_batch: dict[str, object]):
            observed_batch.update(model_batch)
            assert torch.equal(model_batch["labels"], torch.full_like(input_ids, -100))
            return nullcontext, model_batch

        sharder.shard.side_effect = shard
        with patch(
            "nemo_rl.models.automodel.train.ContextParallelSharder",
            return_value=sharder,
        ) as sharder_cls:
            prepared = prepare_model_forward(
                model,
                processed_inputs,
                device_mesh=device_mesh,
                cp_size=2,
                padding_token_id=7,
                is_reward_model=False,
                allow_flash_attn_args=True,
            )

        constructor_args = sharder_cls.call_args
        assert constructor_args.args[:2] == (model, device_mesh)
        assert constructor_args.kwargs == {"padding_token_id": 7, "num_chunks": 1}
        assert prepared.cp_sharder is sharder
        assert prepared.model_context_factory is nullcontext
        assert prepared.model_batch["input_ids"] is not input_ids
        assert prepared.model_batch["attention_mask"] is not attention_mask
        assert torch.equal(prepared.model_batch["input_ids"], input_ids)
        assert torch.equal(prepared.model_batch["attention_mask"], attention_mask)
        assert "position_ids" not in prepared.model_batch
        assert "labels" not in prepared.model_batch
        assert "labels" in observed_batch
        assert processed_inputs.input_ids is input_ids


@pytest.mark.automodel
def test_cp_forward_requires_the_prepared_sharder() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    processed_inputs = ProcessedInputs(input_ids=input_ids, seq_len=4)
    processed_mb = ProcessedMicrobatch(
        data_dict=BatchedDataDict({"input_ids": input_ids}),
        processed_inputs=processed_inputs,
        original_batch_size=1,
        original_seq_len=4,
    )
    prepared = PreparedModelForward(
        model_batch={"input_ids": input_ids},
        cp_size=2,
        cp_sharder=None,
        model_context_factory=nullcontext,
    )
    model = MagicMock()

    with pytest.raises(RuntimeError, match="ContextParallelSharder is required"):
        forward_with_post_processing_fn(
            model=model,
            prepared=prepared,
            post_processing_fn=ScorePostProcessor(cfg={}),
            processed_mb=processed_mb,
        )

    model.assert_not_called()


@pytest.mark.automodel
def test_grpo_logprobs_follow_automodel_sequence_layout() -> None:
    torch.manual_seed(11)
    order = torch.tensor([0, 3, 1, 2])
    layout = _PermutationTokenLayout(order)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    canonical_logits = torch.randn(1, 4, 5)
    local_logits = canonical_logits.index_select(1, order).requires_grad_()

    actual = get_cp_sharded_next_token_logprobs(
        local_logits,
        input_ids,
        layout,
    )
    expected = (
        torch.log_softmax(canonical_logits.float(), dim=-1)[:, :-1]
        .gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert local_logits.grad is not None
    assert torch.isfinite(local_logits.grad).all()


@pytest.mark.automodel
def test_standard_distillation_statistics_follow_automodel_sequence_layout() -> None:
    torch.manual_seed(17)
    order = torch.tensor([0, 3, 1, 2])
    layout = _PermutationTokenLayout(order)
    teacher_topk_indices = torch.tensor(
        [[[0, 2], [1, 3], [2, 4], [0, 4]]], dtype=torch.long
    )
    teacher_topk_logits = torch.randn(1, 4, 2)
    canonical_student_logits = torch.randn(1, 4, 5)
    local_student_logits = canonical_student_logits.index_select(1, order)

    student_logprobs, teacher_logprobs, entropy = (
        get_distillation_topk_logprobs_from_logits(
            student_logits=local_student_logits,
            teacher_topk_logits=teacher_topk_logits,
            teacher_topk_indices=teacher_topk_indices,
            zero_outside_topk=True,
            calculate_entropy=True,
            cp_sharder=layout,
        )
    )

    canonical_student_logprobs = torch.log_softmax(
        canonical_student_logits.float(), dim=-1
    )
    expected_student = canonical_student_logprobs.gather(
        dim=-1, index=teacher_topk_indices
    )[:, :-1]
    expected_teacher = torch.log_softmax(teacher_topk_logits.float(), dim=-1)[:, :-1]
    expected_entropy = (
        canonical_student_logprobs.exp() * canonical_student_logprobs
    ).sum(dim=-1)[:, :-1]

    torch.testing.assert_close(student_logprobs, expected_student)
    torch.testing.assert_close(teacher_logprobs, expected_teacher)
    torch.testing.assert_close(entropy, expected_entropy)


@pytest.mark.automodel
def test_full_logits_postprocessor_emits_contiguous_cp_window() -> None:
    local_logits = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    full_logits = torch.randn(1, 6, 4)
    cp_group = object()
    cp_mesh = MagicMock()
    cp_mesh.get_group.return_value = cp_group
    cp_sharder = MagicMock()
    cp_sharder.gather_token_tensor.return_value = full_logits
    processor = FullLogitsPostProcessor(
        cfg={},
        cp_mesh=cp_mesh,
        cp_size=2,
    )

    with patch("torch.distributed.get_rank", return_value=1) as get_rank:
        actual = processor(
            logits=local_logits,
            data_dict=BatchedDataDict({}),
            processed_inputs=MagicMock(),
            original_batch_size=1,
            original_seq_len=6,
            cp_sharder=cp_sharder,
        )

    torch.testing.assert_close(actual, full_logits[:, 3:6].float())
    gathered_logits = cp_sharder.gather_token_tensor.call_args.args[0]
    torch.testing.assert_close(gathered_logits, local_logits.float())
    assert cp_sharder.gather_token_tensor.call_args.kwargs == {
        "seq_dim": 1,
        "trim": True,
    }
    get_rank.assert_called_once_with(cp_group)


@pytest.mark.automodel
@pytest.mark.parametrize(
    ("input_type", "expected_fanout"),
    [
        (LossInputType.LOGIT, 2),
        (LossInputType.LOGPROB, 2),
        (LossInputType.DISTILLATION, 2),
        (LossInputType.DISTILLATION_CROSS_TOKENIZER, 1),
        (LossInputType.DRAFT, 1),
    ],
)
def test_loss_cp_gradient_fanout_contract(
    input_type: LossInputType, expected_fanout: int
) -> None:
    loss_fn = MagicMock(input_type=input_type)
    processor = LossPostProcessor(
        loss_fn=loss_fn,
        cfg={},
        cp_mesh=None,
        cp_size=2,
        dp_size=1,
    )

    assert processor.cp_gradient_fanout == expected_fanout


@pytest.mark.automodel
def test_cp1_loss_gradient_fanout_is_identity() -> None:
    loss_fn = MagicMock(input_type=LossInputType.LOGPROB)
    processor = LossPostProcessor(
        loss_fn=loss_fn,
        cfg={},
        cp_mesh=None,
        cp_size=1,
        dp_size=1,
    )

    assert processor.cp_gradient_fanout == 1
