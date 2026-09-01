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

"""Contract tests for NeMo-RL's Automodel context-parallel integration."""

import os
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
    round_robin_local_indices,
    shard_batch_identity,
)

from nemo_rl.algorithms.loss.interfaces import LossInputType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    get_cp_sharded_next_token_logprobs,
    get_distillation_topk_logprobs_from_logits,
)
from nemo_rl.models.automodel.data import ProcessedInputs
from nemo_rl.models.automodel.train import (
    FullLogitsPostProcessor,
    LossPostProcessor,
    _cp_gather_logits,
    prepare_model_forward,
)


def _has_gloo() -> bool:
    return dist.is_available() and dist.is_gloo_available()


def _real_cp_sharder_worker(rank: int, world_size: int, init_file: str) -> None:
    """Exercise the real token gather and its autograd fanout on two ranks."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        device_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
        sharder = ContextParallelSharder(
            device_mesh=device_mesh,
            shard_batch=shard_batch_identity,
            local_token_global_indices=round_robin_local_indices,
            shard_layout=ShardLayout(original_seq_len=6, padded_seq_len=8),
        )

        full_logits = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)
        cp1_grad = torch.ones_like(full_logits)
        expected_local_grad = sharder.shard_token_tensor(cp1_grad, seq_dim=1, fill=0.0)

        local_logits = sharder.shard_token_tensor(
            full_logits, seq_dim=1, fill=0.0
        ).detach()
        local_logits.requires_grad_(True)
        gathered_logits = _cp_gather_logits(local_logits, sharder)

        # The sharder pads 6 -> 8 for the round-robin CP layout, then the real
        # collective restores canonical order and trims back to the CP1 shape.
        torch.testing.assert_close(gathered_logits, full_logits)
        gathered_logits.sum().backward()

        # Every CP rank consumes the gathered loss, so differentiable
        # all_gather sums one identical contribution per rank. Pad slots are
        # trimmed from the loss and therefore retain zero gradient.
        torch.testing.assert_close(local_logits.grad, expected_local_grad * world_size)

        normalized_local_logits = sharder.shard_token_tensor(
            full_logits, seq_dim=1, fill=0.0
        ).detach()
        normalized_local_logits.requires_grad_(True)
        normalized_gathered_logits = _cp_gather_logits(normalized_local_logits, sharder)
        processor = LossPostProcessor(
            loss_fn=MagicMock(input_type=LossInputType.LOGIT),
            cfg={},
            cp_mesh=device_mesh["cp"],
            cp_size=world_size,
            dp_size=1,
        )
        (normalized_gathered_logits.sum() / processor.cp_gradient_fanout).backward()
        torch.testing.assert_close(normalized_local_logits.grad, expected_local_grad)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.automodel
@pytest.mark.skipif(not _has_gloo(), reason="gloo backend unavailable")
def test_real_cp_sharder_forward_and_gradient_fanout_parity(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(
            target=_real_cp_sharder_worker,
            args=(rank, 2, str(tmp_path / "cp-sharder-init")),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            for child in processes:
                if child.is_alive():
                    child.terminate()
                    child.join()
            pytest.fail("two-rank Gloo CP sharder test timed out")
        assert process.exitcode == 0


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
    legacy_cp_group = object()

    with patch("torch.distributed.get_world_size") as get_world_size:
        student_logprobs, teacher_logprobs, entropy = (
            get_distillation_topk_logprobs_from_logits(
                student_logits=local_student_logits,
                teacher_topk_logits=teacher_topk_logits,
                teacher_topk_indices=teacher_topk_indices,
                zero_outside_topk=True,
                calculate_entropy=True,
                context_parallel_group=legacy_cp_group,
                cp_sharder=layout,
            )
        )

    get_world_size.assert_not_called()

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
def test_full_logits_postprocessor_rejects_non_divisible_cp_sequence() -> None:
    local_logits = torch.randn(1, 3, 4)
    cp_mesh = MagicMock()
    cp_sharder = MagicMock()
    cp_sharder.gather_token_tensor.return_value = torch.randn(1, 5, 4)
    processor = FullLogitsPostProcessor(
        cfg={},
        cp_mesh=cp_mesh,
        cp_size=2,
    )

    with pytest.raises(
        ValueError,
        match=(
            "X-token teacher sequence length must be divisible by the teacher "
            "context parallel size"
        ),
    ):
        processor(
            logits=local_logits,
            data_dict=BatchedDataDict({}),
            processed_inputs=MagicMock(),
            original_batch_size=1,
            original_seq_len=5,
            cp_sharder=cp_sharder,
        )

    cp_sharder.gather_token_tensor.assert_called_once_with(
        local_logits.float(), seq_dim=1, trim=True
    )
    cp_mesh.get_group.assert_not_called()


@pytest.mark.automodel
@pytest.mark.parametrize(
    ("input_type", "expected_fanout"),
    [
        (LossInputType.LOGIT, 2),
        (LossInputType.LOGPROB, 2),
        (LossInputType.DISTILLATION, 2),
        (LossInputType.DISTILLATION_CROSS_TOKENIZER, 1),
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
def test_loss_cp_rejects_unsupported_draft_input() -> None:
    loss_fn = MagicMock(input_type=LossInputType.DRAFT)
    processor = LossPostProcessor(
        loss_fn=loss_fn,
        cfg={},
        cp_mesh=None,
        cp_size=2,
        dp_size=1,
    )

    with pytest.raises(
        NotImplementedError,
        match=r"Loss input type LossInputType\.DRAFT is not supported",
    ):
        processor(
            logits=torch.empty(0),
            data_dict=BatchedDataDict({}),
            processed_inputs=MagicMock(),
            global_valid_seqs=torch.tensor(1),
            global_valid_toks=torch.tensor(1),
            cp_sharder=MagicMock(),
        )
