# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Packed score-centering shares one dense-vocabulary backward branch.

Use real Torch/Gloo groups and compare with an independent dense distribution.
The graph assertion covers the memory regression; numerical equality alone
would also accept two independent dense-logits backward consumers.
"""

from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from nemo_rl.algorithms.loss.interfaces import LossInputType
from nemo_rl.algorithms.loss.loss_input import prepare_packed_loss_input
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import get_policy_logprobs_at_sampler_topk

RTOL = 2e-5
ATOL = 2e-7
LENGTHS = (5, 3, 1)
LOGICAL = (0, 5, 8, 9)
PADDED = (0, 8, 12, 16)
WIDTH = 7
VOCAB = 32


def _groups(
    *, rank: int, cp_size: int, tp_size: int
) -> tuple[int, int, dist.ProcessGroup, dist.ProcessGroup]:
    # Every rank creates the same groups in the same order; no mocked collective.
    cp_groups = [
        dist.new_group([c * tp_size + t for c in range(cp_size)])
        for t in range(tp_size)
    ]
    tp_groups = [
        dist.new_group([c * tp_size + t for t in range(tp_size)])
        for c in range(cp_size)
    ]
    return (
        rank // tp_size,
        rank % tp_size,
        cp_groups[rank % tp_size],
        tp_groups[rank // tp_size],
    )


def _local_positions(*, cp_rank: int, cp_size: int) -> list[int]:
    if cp_size == 1:
        return list(range(PADDED[-1]))
    result = []
    for start, end in zip(PADDED, PADDED[1:]):
        chunk = (end - start) // (2 * cp_size)
        for part in (cp_rank, 2 * cp_size - cp_rank - 1):
            result.extend(range(start + part * chunk, start + (part + 1) * chunk))
    return result


def _data(
    *, score_centering: bool = True, nonfinite_case: bool = False
) -> tuple[BatchedDataDict[dict[str, torch.Tensor]], SimpleNamespace]:
    generator = torch.Generator().manual_seed(142)
    ids = torch.randint(0, VOCAB, (3, WIDTH), generator=generator)
    ids[0, 1] = 1
    ids[0, 3] = VOCAB - 1  # both TP ownership ranges appear on valid positions
    heads = torch.randint(0, VOCAB, (3, WIDTH, 4), generator=generator)
    heads[..., 0] = ids  # sampled target overlaps a captured head entry
    heads[..., 2] = heads[..., 1]  # independent weighted duplicate head indices
    if nonfinite_case:
        ids.fill_(VOCAB - 1)
        heads[..., 0] = 0  # legitimate -inf entry, inactive under finite-head mask
        heads[..., 1:] = VOCAB - 1
    token_mask = torch.zeros(3, WIDTH)
    for index, length in enumerate(LENGTHS):
        token_mask[index, 1:length] = 1
    token_mask[0, 2] = 0  # an interior prompt/tool/masked token
    captured_logprobs = torch.zeros_like(heads, dtype=torch.float32)
    if nonfinite_case:
        captured_logprobs[..., 0] = -float("inf")
    data = BatchedDataDict(
        {
            "input_ids": ids,
            "input_lengths": torch.tensor(LENGTHS),
            "token_mask": token_mask,
            "sample_mask": torch.tensor([1.0, 0.0, 1.0]),
            "sampler_topk_token_ids": heads,
            "sampler_topk_logprobs": captured_logprobs,
        }
    )
    loss = SimpleNamespace(
        input_type=LossInputType.LOGPROB, score_centering=score_centering
    )
    return data, loss


def _reference_outputs(
    logits: torch.Tensor, *, data: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    # Independently roll each padded original row, without calling the packer or
    # either old/new logprob helper. Compare every returned value, including pads.
    logprobs = logits.log_softmax(-1)
    sampled_rows, head_rows = [], []
    for row, (start, end) in enumerate(zip(PADDED, PADDED[1:])):
        span = end - start
        target = torch.zeros(span, dtype=torch.long)
        heads = torch.zeros(span, 4, dtype=torch.long)
        target[: LENGTHS[row]] = data["input_ids"][row, : LENGTHS[row]]
        heads[: LENGTHS[row]] = data["sampler_topk_token_ids"][row, : LENGTHS[row]]
        target = torch.cat((target[1:], target[:1]))
        heads = torch.cat((heads[1:], heads[:1]))
        width = min(span - 1, WIDTH - 1)
        values = logprobs[:, start : start + width]
        sampled = values.gather(-1, target[:width].view(1, width, 1)).squeeze(-1)
        head = values.gather(-1, heads[:width].unsqueeze(0))
        sampled_rows.append(F.pad(sampled, (0, WIDTH - 1 - width)))
        head_rows.append(F.pad(head, (0, 0, 0, WIDTH - 1 - width)))
    return {
        "next_token_logprobs": torch.cat(sampled_rows, 0),
        "policy_logprobs_at_sampler_topk": torch.cat(head_rows, 0),
    }


def _objective(
    prepared: Mapping[str, torch.Tensor], *, data: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    sampled = prepared["next_token_logprobs"]
    mask = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    sampled_weights = torch.linspace(-0.7, 0.9, sampled.numel()).reshape_as(sampled)
    safe_sampled = torch.where(mask != 0, sampled, torch.zeros_like(sampled))
    value = (safe_sampled * sampled_weights * mask).sum()
    head = prepared.get("policy_logprobs_at_sampler_topk")
    if head is not None:
        weights = torch.linspace(0.8, -0.4, head.numel()).reshape_as(head)
        finite = torch.isfinite(head)
        # Same inactive-q=-inf treatment used by score-centering's finite head.
        safe_head = torch.where(finite, head, torch.zeros_like(head))
        value = value + (safe_head * weights * mask.unsqueeze(-1)).sum()
    return value


def _node_counts(loss: torch.Tensor) -> dict[str, int]:
    seen, todo, counts = set(), [loss.grad_fn], {}
    while todo:
        node = todo.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
        todo.extend(next_node for next_node, _ in node.next_functions)
    return counts


def _packed_dense_reference_case(
    *,
    cp_rank: int,
    tp_rank: int,
    cp_size: int,
    tp_size: int,
    cp_group: dist.ProcessGroup,
    tp_group: dist.ProcessGroup,
    score_centering: bool,
) -> None:
    data, loss_fn = _data(score_centering=score_centering)
    before = {key: value.clone() for key, value in data.items()}
    positions = torch.tensor(
        _local_positions(cp_rank=cp_rank, cp_size=cp_size),
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(271)
    hidden_full = torch.randn(1, PADDED[-1], 7, generator=generator) / 3
    weight_full = torch.randn(VOCAB, 7, generator=generator) / 3
    vocab_slice = slice(
        tp_rank * (VOCAB // tp_size), (tp_rank + 1) * (VOCAB // tp_size)
    )
    hidden = hidden_full.index_select(1, positions).clone().requires_grad_()
    weight = weight_full[vocab_slice].clone().requires_grad_()
    logits = F.linear(
        hidden, weight
    )  # Real non-leaf activation logits, as in training.
    logits.retain_grad()
    original_logits = logits.detach().clone()
    prepared, returned = prepare_packed_loss_input(
        logits=logits,
        data=data,
        loss_fn=loss_fn,
        cu_seqlens_q=LOGICAL,
        cu_seqlens_q_padded=PADDED,
        vocab_parallel_rank=tp_rank,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
        chunk_size=3,
    )
    assert returned is data
    loss = _objective(prepared, data=data)
    nodes = _node_counts(loss)
    # One joint gather replaces simultaneous CE and head dense-logits gradients.
    assert nodes.get("ChunkedDistributedGatherLogprobBackward", 0) == int(
        score_centering
    )
    assert nodes.get("ChunkedDistributedLogprobBackward", 0) == int(not score_centering)
    # CP reconstruction repeats the scalar objective on each CP rank.
    (loss / cp_size).backward()
    torch.testing.assert_close(logits.detach(), original_logits, rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(data[key], value, rtol=0, atol=0)
    for value in (logits.grad, hidden.grad, weight.grad):
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
    hidden_grad = hidden.grad.clone()
    weight_grad = weight.grad.clone()
    dist.all_reduce(hidden_grad, group=tp_group)
    dist.all_reduce(weight_grad, group=cp_group)

    ref_hidden = hidden_full.clone().requires_grad_()
    ref_weight = weight_full.clone().requires_grad_()
    reference = _reference_outputs(F.linear(ref_hidden, ref_weight), data=data)
    if not score_centering:
        reference.pop("policy_logprobs_at_sampler_topk")
    _objective(reference, data=data).backward()
    assert set(prepared) == set(reference)
    for key in reference:
        torch.testing.assert_close(prepared[key], reference[key], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        hidden_grad, ref_hidden.grad.index_select(1, positions), rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(
        weight_grad, ref_weight.grad[vocab_slice], rtol=RTOL, atol=ATOL
    )


def _nonfinite_tp_shard_case(tp_rank: int, *, tp_group: dist.ProcessGroup) -> None:
    """An out-of-shard selection must contribute zero, including local -inf."""
    data, loss_fn = _data(nonfinite_case=True)
    full = torch.linspace(-0.8, 1.1, PADDED[-1] * VOCAB).reshape(1, PADDED[-1], VOCAB)
    full[..., : VOCAB // 2] = -float("inf")
    shard = slice(tp_rank * (VOCAB // 2), (tp_rank + 1) * (VOCAB // 2))
    local = full[..., shard].clone().requires_grad_()
    prepared, _ = prepare_packed_loss_input(
        logits=local,
        data=data,
        loss_fn=loss_fn,
        cu_seqlens_q=LOGICAL,
        cu_seqlens_q_padded=PADDED,
        vocab_parallel_rank=tp_rank,
        vocab_parallel_group=tp_group,
        chunk_size=3,
    )
    reference_logits = full.clone().requires_grad_()
    reference = _reference_outputs(reference_logits, data=data)
    for key in reference:
        assert not torch.isnan(prepared[key]).any()
        torch.testing.assert_close(
            prepared[key], reference[key], rtol=RTOL, atol=ATOL, equal_nan=False
        )
    _objective(prepared, data=data).backward()
    _objective(reference, data=data).backward()
    torch.testing.assert_close(
        local.grad,
        reference_logits.grad[..., shard],
        rtol=RTOL,
        atol=ATOL,
        equal_nan=False,
    )


# torch.multiprocessing.spawn passes rank and its args tuple positionally.
def _gloo_worker(
    rank: int, world_size: int, cp_size: int, tp_size: int, rendezvous: str
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        cp_rank, tp_rank, cp_group, tp_group = _groups(
            rank=rank, cp_size=cp_size, tp_size=tp_size
        )
        _packed_dense_reference_case(
            cp_rank=cp_rank,
            tp_rank=tp_rank,
            cp_size=cp_size,
            tp_size=tp_size,
            cp_group=cp_group,
            tp_group=tp_group,
            score_centering=True,
        )
        if cp_size == 1:
            # The no-score-centering path keeps its ordinary CE branch.
            _packed_dense_reference_case(
                cp_rank=cp_rank,
                tp_rank=tp_rank,
                cp_size=cp_size,
                tp_size=tp_size,
                cp_group=cp_group,
                tp_group=tp_group,
                score_centering=False,
            )
            if tp_size == 2:
                _nonfinite_tp_shard_case(tp_rank, tp_group=tp_group)
        assert not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "tp_size,cp_size", [(1, 1), (2, 1), (2, 2)], ids=["tp1", "tp2", "tp2-cp2"]
)
def test_packed_score_centering_joint_gather_matches_dense_backward(
    tmp_path: Path, *, tp_size: int, cp_size: int
) -> None:
    """Unequal/padded rows, length-one rows, masked samples, duplicate head IDs."""
    world_size = cp_size * tp_size
    mp.spawn(
        _gloo_worker,
        args=(world_size, cp_size, tp_size, str(tmp_path / "gloo-init")),
        nprocs=world_size,
        join=True,
    )


def test_sampler_head_without_tp_preserves_negative_inf_and_finite_gradients() -> None:
    """The direct path also keeps legitimate -inf entries without introducing NaN."""
    base = torch.linspace(-0.8, 1.1, 5 * 8).reshape(1, 5, 8)
    base[..., :4] = -float("inf")
    logits = base.clone().requires_grad_()
    ids = torch.tensor([[[0, 7, 7]] * 5])
    actual = get_policy_logprobs_at_sampler_topk(ids, logits, chunk_size=2)
    reference_logits = base.clone().requires_grad_()
    expected = reference_logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:])
    assert torch.isneginf(actual[..., 0]).all() and not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)
    weights = torch.linspace(-0.7, 0.9, actual.numel()).reshape_as(actual)
    torch.where(torch.isfinite(actual), actual, torch.zeros_like(actual)).mul(
        weights
    ).sum().backward()
    torch.where(torch.isfinite(expected), expected, torch.zeros_like(expected)).mul(
        weights
    ).sum().backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(
        logits.grad, reference_logits.grad, rtol=RTOL, atol=ATOL, equal_nan=False
    )
