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
"""Exactness oracle for the logprobs and the loss of shared-prefix packs.

A tree is one prompt followed by the completions of the rows that start with that
prompt. A pack is one model forward that holds several trees and runs each prompt
once.

The model is a table lookup. The logits at a token are the row of a trainable table
that a hash of the token's causal history selects. For a prompt token the history is
the prompt up to that token. For a completion token it is the prompt followed by the
row's own completion up to that token. Every token of a pack therefore gets the
logits that a dense forward gives the same token of the same row. Correct pack code
must reproduce the dense logprobs of every row, the dense loss and the dense table
gradient. Both paths are also checked against an fp64 single-process computation of
the same loss, so a scale error common to both fails too.

Batches take the production path. ``plan_shared_prefix_shards`` plans the forwards as
the driver does. The rows get the two columns that the Megatron worker attaches.
``get_microbatch_iterator`` then yields packs and dense bins, and
``forward_with_post_processing_fn`` runs them with ``LossPostProcessor`` and
``LogprobsPostProcessor``.

TP1/CP1 runs in-process on one GPU. The TP/CP variants spawn one process per GPU.
A test skips when it lacks its GPUs. With ``NRL_REQUIRE_SHARED_PREFIX_TESTS=1``, which
the Megatron lanes set, it fails instead, except the TP2/CP2 variant: it needs four
GPUs, and a lane guarantees only two. Run it with four visible GPUs, e.g.
``CUDA_VISIBLE_DEVICES=0,1,2,3 pytest --mcore-only <this file> -k distributed``.
"""

import functools
import hashlib
import os
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.datasets.shared_prefix.layout import SharedPrefixPack  # noqa: E402

from nemo_rl.algorithms.logits_sampling_utils import (  # noqa: E402
    TrainingSamplingParams,
)
from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType  # noqa: E402
from nemo_rl.data.packing.shared_prefix.plan import (  # noqa: E402
    SharedPrefixShard,
    plan_shared_prefix_shards,
)
from nemo_rl.data.packing.shared_prefix.row_metadata import (  # noqa: E402
    SHARED_PREFIX_PROMPT_LENGTHS,
    SHARED_PREFIX_TREE_IDS,
    UNSHARED_ROW,
    prompt_hash,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.megatron.data import (  # noqa: E402
    ProcessedMicrobatch,
    get_microbatch_iterator,
)
from nemo_rl.models.megatron.train import (  # noqa: E402
    LogprobsPostProcessor,
    LossPostProcessor,
    forward_with_post_processing_fn,
)
from nemo_rl.utils.sequence_lengths import to_cpu_int_tuple  # noqa: E402
from nemo_rl.utils.tensor_ops import pad_and_concat  # noqa: E402

pytestmark = pytest.mark.mcore

_VOCAB = 128
_FEATURES = 4099
_WIDTH = 40
_TABLE = (
    torch.randn(
        _FEATURES,
        _VOCAB,
        generator=torch.Generator().manual_seed(1234),
        dtype=torch.float64,
    )
    * 2.0
)
# The logprob error against the fp64 reference is fp32 rounding: the model has no
# matmul, so TF32 cannot enter. A misrouted logprob or a CP/TP scale error is orders
# of magnitude larger than these tolerances.
_LOGPROB_ATOL = 1e-5
_LOSS_RTOL = 1e-5
_GRAD_RTOL = 1e-6


@dataclass(frozen=True)
class _Case:
    # (group id, prompt length, completions); rows are interleaved at random.
    groups: tuple[tuple[str, int, int], ...]
    seed: int
    # Token budget of one forward.
    max_tokens: int
    # Whether the driver counts a training tree at its full length, with the prompt
    # in every row. It does when the model runs MTP.
    bound_by_full_length: bool = False
    # Groups that get one invalid placeholder row (prompt length 0, masked).
    placeholders: tuple[str, ...] = ()
    # Whether the batch also holds one group with the rows of forest training.
    forest_training: bool = False
    mtp_loss_mask: bool = False
    # What the training plan must contain besides one pack: "several_trees" (a pack
    # of two or more trees), "several_packs", "split_group" (one group in two or
    # more trees) or "dense_rows" (exactly the rows without a prompt and the rows
    # whose prompt no other row of their group has run in dense bins).
    expect: Optional[str] = None


_SPEC_A = (("g0", 5, 3), ("g1", 7, 4), ("g2", 9, 2), ("g3", 3, 5))
# A one-row group, a group with more rows than one tree holds, and a one-token prompt.
_SPEC_B = (("g0", 5, 3), ("big", 6, 17), ("one", 4, 1), ("p1", 1, 4), ("g4", 8, 2))
_CASES = {
    "one_pack": _Case(_SPEC_A, 11, 512, mtp_loss_mask=True, expect="several_trees"),
    "small_budget": _Case(_SPEC_A, 13, 160, expect="several_packs"),
    "full_length_bound": _Case(
        _SPEC_A,
        14,
        160,
        bound_by_full_length=True,
        mtp_loss_mask=True,
        expect="several_packs",
    ),
    "single_tree": _Case((("g0", 6, 5),), 15, 512),
    # The planner divides the 17-row group into trees of 9 and 8 rows.
    "split_group": _Case(_SPEC_B, 21, 1024, expect="split_group"),
    "placeholder": _Case(_SPEC_B, 22, 1024, placeholders=("g4",), expect="dense_rows"),
    "forest_training": _Case(
        _SPEC_A,
        23,
        512,
        forest_training=True,
        mtp_loss_mask=True,
        expect="dense_rows",
    ),
}


def _feature(history: Sequence[Any]) -> int:
    digest = hashlib.blake2b(str(list(history)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % _FEATURES


def _causal_features(tokens: Sequence[int]) -> list[int]:
    return [_feature(tokens[: position + 1]) for position in range(len(tokens))]


def _local_positions(
    segments: Sequence[tuple[int, int]], cp_rank: int, cp_size: int
) -> list[int]:
    """Global positions a CP rank holds: chunks r and 2*CP-1-r of each segment."""
    positions: list[int] = []
    for start, end in segments:
        if cp_size == 1:
            positions.extend(range(start, end))
            continue
        chunk = (end - start) // (2 * cp_size)
        assert chunk * 2 * cp_size == end - start, (start, end, cp_size)
        for index in (cp_rank, 2 * cp_size - 1 - cp_rank):
            positions.extend(range(start + index * chunk, start + (index + 1) * chunk))
    return positions


def _pack_features(ids: Sequence[int], layout: SharedPrefixPack) -> list[int]:
    """Causal-history features of a packed sequence: each prompt, then its rows."""
    features = [_feature(("pad", position)) for position in range(len(ids))]
    for offset, tree in layout.iter_roots():
        prompt = list(ids[offset : offset + tree.prefix_len])
        features[offset : offset + tree.prefix_len] = _causal_features(prompt)
        start = offset + tree.prefix_len
        for length in tree.completion_lens:
            row = prompt + list(ids[start : start + length])
            features[start : start + length] = _causal_features(row)[tree.prefix_len :]
            start += length
    return features


def _table_shard(tp_rank: int, tp_size: int) -> torch.Tensor:
    width = _VOCAB // tp_size
    shard = _TABLE[:, tp_rank * width : (tp_rank + 1) * width]
    return shard.to(device=torch.cuda.current_device(), dtype=torch.float32)


class _CausalHistoryModel(torch.nn.Module):
    """Logits from a table row keyed by each token's causal history.

    It rebuilds the whole token sequence from the CP-local inputs and the layout it
    is given, as attention would see it: a pack as each prompt followed by its rows,
    a dense bin as packed sequences.
    """

    def __init__(self, tp_rank: int, tp_size: int, cp_rank: int, cp_size: int):
        super().__init__()
        self.table = torch.nn.Parameter(_table_shard(tp_rank, tp_size))
        self.cp_rank = cp_rank
        self.cp_size = cp_size

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        shared_prefix_layout: Optional[SharedPrefixPack] = None,
        packed_seq_params: Any = None,
        **unused: Any,
    ) -> torch.Tensor:
        del position_ids, attention_mask, unused
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1, input_ids.shape
        if shared_prefix_layout is not None:
            assert packed_seq_params is None
            segments = [(0, input_ids.shape[1] * self.cp_size)]
        else:
            assert packed_seq_params is not None, "a planned forward ran unpacked"
            boundaries = to_cpu_int_tuple(packed_seq_params.cu_seqlens_q_padded)
            segments = list(zip(boundaries[:-1], boundaries[1:]))
        local = _local_positions(segments, self.cp_rank, self.cp_size)
        assert len(local) == input_ids.shape[1], (len(local), input_ids.shape)
        ids = self._global_ids(input_ids[0], segments)
        if shared_prefix_layout is not None:
            features = _pack_features(ids, shared_prefix_layout)
        else:
            features = [
                feature
                for start, end in segments
                for feature in _causal_features(ids[start:end])
            ]
        index = torch.tensor([features[position] for position in local])
        return self.table[index.to(self.table.device)].unsqueeze(0)

    def _global_ids(
        self, local_ids: torch.Tensor, segments: Sequence[tuple[int, int]]
    ) -> list[int]:
        if self.cp_size == 1:
            return local_ids.tolist()
        gathered = [torch.empty_like(local_ids) for _ in range(self.cp_size)]
        dist.all_gather(
            gathered,
            local_ids.contiguous(),
            group=parallel_state.get_context_parallel_group(),
        )
        ids: list[Optional[int]] = [None] * segments[-1][1]
        for rank, rank_ids in enumerate(gathered):
            for position, token in zip(
                _local_positions(segments, rank, self.cp_size), rank_ids.tolist()
            ):
                ids[position] = token
        assert None not in ids
        return ids  # type: ignore[return-value]


class _PolicyGradientLoss:
    """Token-level ``-sum(logprob * advantage * mask) / global_valid_toks``."""

    loss_type = LossType.TOKEN_LEVEL
    input_type = LossInputType.LOGPROB

    def __call__(
        self,
        next_token_logprobs: torch.Tensor,
        data: BatchedDataDict[Any],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del global_valid_seqs, kwargs
        mask = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
        assert next_token_logprobs.shape == mask.shape, (
            next_token_logprobs.shape,
            mask.shape,
        )
        weighted = next_token_logprobs * data["advantages"][:, 1:] * mask
        return -weighted.sum() / global_valid_toks, {}


@dataclass(frozen=True)
class _Rows:
    data: BatchedDataDict[Any]
    # What the rollout reassembler stamps on every row for the driver.
    group_ids: list[str]
    prompt_lengths: list[int]


# Group, prompt length, tokens and the column spans that carry loss. A row without a
# span is invalid: its sample mask is 0.
_Row = tuple[str, int, list[int], tuple[tuple[int, int], ...]]


def _forest_training_rows(rng: np.random.Generator) -> list[_Row]:
    """Rows of one forest-training group: two rollouts and two execution-padding rows.

    In forest training a rollout owns one row per selected path of its model calls.
    The first rollout owns rows 0 to 2. Row 0 holds its first call and a second call
    that continues it. Row 1 branches after the first call, whose loss row 0 owns.
    Row 2 starts from a rewritten prompt, as after a compaction. Row 3 is the second
    rollout. The padding rows copy row 0 and carry no loss.
    """

    def tokens(count: int) -> list[int]:
        return rng.integers(1, _VOCAB, count).tolist()

    prompt = tokens(5)
    # Same length as the prompt, and different in every token.
    rewritten_prompt = [token % (_VOCAB - 1) + 1 for token in prompt]
    # The prompt, the 4 tokens of the first call and 3 tokens that carry no loss.
    shared_calls = prompt + tokens(4) + tokens(3)
    continuation = shared_calls + tokens(6)
    branch = shared_calls + tokens(3)
    return [
        ("forest", 5, continuation, ((5, 9), (12, 18))),
        ("forest", 5, branch, ((12, 15),)),
        ("forest", 5, rewritten_prompt + tokens(7), ((5, 12),)),
        ("forest", 5, prompt + tokens(8), ((5, 13),)),
        ("forest", 0, continuation, ()),
        ("forest", 0, continuation, ()),
    ]


def _build_rows(case: _Case) -> _Rows:
    rng = np.random.default_rng(case.seed)
    rows: list[_Row] = []
    for group, prompt_length, completions in case.groups:
        prompt = rng.integers(1, _VOCAB, prompt_length).tolist()
        for _ in range(completions):
            completion_length = int(rng.integers(1, _WIDTH - prompt_length + 1))
            completion = rng.integers(0, _VOCAB, completion_length).tolist()
            tokens = prompt + completion
            rows.append((group, prompt_length, tokens, ((prompt_length, len(tokens)),)))
        if group in case.placeholders:
            # Reassembler placeholder: one pad token, prompt length 0, masked.
            rows.append((group, 0, [0], ()))
    if case.forest_training:
        rows.extend(_forest_training_rows(rng))
    rows = [rows[index] for index in rng.permutation(len(rows))]
    size = len(rows)
    input_ids = torch.zeros(size, _WIDTH, dtype=torch.long)
    token_mask = torch.zeros(size, _WIDTH)
    for index, (_, _, tokens, loss_spans) in enumerate(rows):
        input_ids[index, : len(tokens)] = torch.tensor(tokens)
        for start, end in loss_spans:
            token_mask[index, start:end] = 1.0
    sample_mask = torch.tensor([1.0 if loss_spans else 0.0 for *_, loss_spans in rows])
    data: dict[str, Any] = {
        "input_ids": input_ids,
        "input_lengths": torch.tensor([len(tokens) for _, _, tokens, _ in rows]),
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "advantages": torch.randn(
            size, _WIDTH, generator=torch.Generator().manual_seed(case.seed)
        ),
    }
    if case.mtp_loss_mask:
        data["mtp_loss_mask"] = token_mask * sample_mask.unsqueeze(-1)
    return _Rows(
        data=BatchedDataDict(data),
        group_ids=[group for group, *_ in rows],
        prompt_lengths=[length for _, length, *_ in rows],
    )


def _policy_cfg(chunk_size: Optional[int], tp_size: int, cp_size: int) -> dict:
    multiple = 2 * tp_size * cp_size if tp_size > 1 or cp_size > 1 else 1
    return {
        "make_sequence_length_divisible_by": multiple,
        "sequence_packing": {
            "enabled": True,
            "algorithm": "modified_first_fit_decreasing",
        },
        "dynamic_batching": {"enabled": False},
        "megatron_cfg": {
            "tensor_model_parallel_size": tp_size,
            "context_parallel_size": cp_size,
            "pipeline_model_parallel_size": 1,
            "sequence_parallel": tp_size > 1,
        },
        "logprob_chunk_size": chunk_size,
    }


def _plan_rank_batch(
    rows: _Rows, case: _Case, cfg: dict, *, bound_by_full_length: bool
) -> tuple[BatchedDataDict[Any], SharedPrefixShard]:
    """Plan the forwards on the driver and attach the plan as the worker does."""
    input_ids = rows.data["input_ids"].tolist()
    (shard,) = plan_shared_prefix_shards(
        sequence_lengths=rows.data["input_lengths"].tolist(),
        group_ids=rows.group_ids,
        prompt_lengths=rows.prompt_lengths,
        prompt_hashes=[
            prompt_hash(tokens[:length])
            for tokens, length in zip(input_ids, rows.prompt_lengths)
        ],
        num_shards=1,
        batch_size=None,
        max_tokens_per_microbatch=case.max_tokens,
        sequence_length_pad_multiple=cfg["make_sequence_length_divisible_by"],
        algorithm=cfg["sequence_packing"]["algorithm"],
        bound_by_full_length=bound_by_full_length,
        largest_first=False,
    )
    assert sorted(shard.row_indices) == list(range(rows.data.size))
    data = rows.data.select_indices(shard.row_indices)
    data.micro_batch_indices = shard.micro_batch_indices
    data.micro_batch_lengths = shard.micro_batch_lengths
    data.elem_counts_per_gb = shard.elem_counts_per_gb
    data[SHARED_PREFIX_TREE_IDS] = torch.tensor(shard.tree_ids)
    data[SHARED_PREFIX_PROMPT_LENGTHS] = torch.tensor(
        [rows.prompt_lengths[row] for row in shard.row_indices]
    )
    return data, shard


def _check_plan(case: _Case, rows: _Rows, shard: SharedPrefixShard) -> None:
    """Check that the plan has the forwards that the case is meant to exercise."""
    tree_ids = shard.tree_ids
    forwards = [
        (shard.row_indices[start:end], tree_ids[start:end])
        for start, end in shard.micro_batch_indices[0]
    ]
    packs = [forward for forward in forwards if forward[1][0] != UNSHARED_ROW]
    # A plan of dense bins only would pass every numeric check trivially.
    assert packs
    if case.expect == "several_trees":
        assert any(max(forward_tree_ids) > 0 for _, forward_tree_ids in packs)
    elif case.expect == "several_packs":
        assert len(packs) > 1
    elif case.expect == "split_group":
        trees = {
            (pack, tree_id, rows.group_ids[row])
            for pack, (forward_rows, forward_tree_ids) in enumerate(packs)
            for row, tree_id in zip(forward_rows, forward_tree_ids)
        }
        trees_per_group = Counter(group for _, _, group in trees)
        assert max(trees_per_group.values()) > 1, trees_per_group
    elif case.expect == "dense_rows":
        prompts = [
            (group, tuple(tokens[:length]))
            for group, length, tokens in zip(
                rows.group_ids, rows.prompt_lengths, rows.data["input_ids"].tolist()
            )
        ]
        rows_with_prompt = Counter(
            prompt for prompt, length in zip(prompts, rows.prompt_lengths) if length
        )
        dense_rows = sorted(
            row
            for row, tree_id in zip(shard.row_indices, tree_ids)
            if tree_id == UNSHARED_ROW
        )
        assert 0 in rows.prompt_lengths
        assert dense_rows == [
            row
            for row, (prompt, length) in enumerate(zip(prompts, rows.prompt_lengths))
            if length == 0 or rows_with_prompt[prompt] == 1
        ]


@dataclass
class _Truth:
    features: torch.Tensor  # [batch, width] causal-history feature per position
    logprobs: torch.Tensor  # [batch, width - 1] fp64
    loss: float
    # Sum of absolute loss terms: the scale for loss tolerances, since random
    # advantages can make the loss itself cancel to nearly zero.
    loss_scale: float
    grad: torch.Tensor  # this rank's vocabulary shard of d(loss)/d(table), fp64


def _truth(
    batch: BatchedDataDict[Any], temperature: float, tp_rank: int, tp_size: int
) -> _Truth:
    input_ids = batch["input_ids"]
    features = torch.tensor([_causal_features(row) for row in input_ids.tolist()])
    table = _TABLE.clone().requires_grad_(True)
    logits = table[features] / temperature
    logprobs = (
        torch.log_softmax(logits[:, :-1], dim=-1)
        .gather(-1, input_ids[:, 1:, None])
        .squeeze(-1)
    )
    mask = (batch["token_mask"][:, 1:] * batch["sample_mask"].unsqueeze(-1)).double()
    advantages = batch["advantages"][:, 1:].double()
    terms = logprobs * advantages * mask
    loss = -terms.sum() / mask.sum()
    loss.backward()
    width = _VOCAB // tp_size
    return _Truth(
        features=features,
        logprobs=logprobs.detach(),
        loss=float(loss),
        loss_scale=float(terms.detach().abs().sum() / mask.sum()),
        grad=table.grad[:, tp_rank * width : (tp_rank + 1) * width],
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.detach().double().cpu()
    expected = expected.detach().double().cpu()
    return float((actual - expected).norm() / expected.norm())


def _cp_summed(grad: torch.Tensor, cp_size: int) -> torch.Tensor:
    grad = grad.detach().clone()
    if cp_size > 1:
        dist.all_reduce(grad, group=parallel_state.get_context_parallel_group())
    return grad


def _dense_reference(
    batch: BatchedDataDict[Any],
    cfg: dict,
    truth: _Truth,
    temperature: float,
    global_valid: tuple[torch.Tensor, torch.Tensor],
    topology: tuple[int, int, int, int],
) -> tuple[float, torch.Tensor]:
    """NeMo-RL's unpacked loss on CP-zigzag logits, as one microbatch."""
    tp_rank, tp_size, cp_rank, cp_size = topology
    table = torch.nn.Parameter(_table_shard(tp_rank, tp_size))
    positions = _local_positions([(0, _WIDTH)], cp_rank, cp_size)
    index = truth.features[:, positions].to(table.device)
    logits = table[index] / temperature
    data = BatchedDataDict(
        {key: value.cuda() for key, value in batch.items() if torch.is_tensor(value)}
    )
    processor = LossPostProcessor(_PolicyGradientLoss(), cfg, num_microbatches=1)
    loss, _ = processor(data, None, *global_valid)(logits)
    # Megatron's schedule scales each microbatch loss by cp_size / num_microbatches.
    scaled = loss * cp_size
    scaled.backward()
    return float(scaled) * cp_size, _cp_summed(table.grad, cp_size)


def _recording(
    iterator: Iterator[ProcessedMicrobatch], seen: list[ProcessedMicrobatch]
) -> Iterator[ProcessedMicrobatch]:
    for microbatch in iterator:
        seen.append(microbatch)
        yield microbatch


def _check_pack_inputs(
    microbatch: ProcessedMicrobatch,
    row_indices: Sequence[int],
    rows: _Rows,
    cfg: dict,
    cp_rank: int,
    cp_size: int,
) -> None:
    """Check the model inputs of one pack against the rows it was built from."""
    params = microbatch.shared_prefix_params
    assert params is not None and microbatch.packed_seq_params is None
    multiple = cfg["make_sequence_length_divisible_by"]
    batch = rows.data
    tree_ids = microbatch.data_dict[SHARED_PREFIX_TREE_IDS].tolist()
    # The packed sequence: each tree's prompt once, then every row's completion,
    # padded with zeros until prompt plus completion is a multiple of M.
    tokens: list[int] = []
    positions: list[int] = []
    mtp_loss_mask: list[float] = []
    for position, row in enumerate(row_indices):
        ids = batch["input_ids"][row].tolist()
        length = int(batch["input_lengths"][row])
        prompt_length = rows.prompt_lengths[row]
        padded_length = -(-length // multiple) * multiple
        if position == 0 or tree_ids[position] != tree_ids[position - 1]:
            tokens += ids[:prompt_length]
            positions += range(prompt_length)
            mtp_loss_mask += [0.0] * prompt_length
        tokens += ids[prompt_length:length] + [0] * (padded_length - length)
        positions += range(prompt_length, padded_length)
        if "mtp_loss_mask" in batch:
            mtp_loss_mask += batch["mtp_loss_mask"][row, prompt_length:length].tolist()
            mtp_loss_mask += [0.0] * (padded_length - length)
    assert isinstance(params.inputs.layout, SharedPrefixPack)
    assert params.inputs.layout.mtp_loss_group_root_counts == (max(tree_ids) + 1,)
    assert params.inputs.packed_input_ids.tolist() == tokens

    # The model gets this rank's zigzag part of the pack, padded at its end to M.
    padded_length = params.cp_shard.padded_total_length
    assert padded_length == -(-len(tokens) // multiple) * multiple
    padding = padded_length - len(tokens)
    local = _local_positions([(0, padded_length)], cp_rank, cp_size)

    def local_part(values: list) -> list:
        padded = values + [0] * padding
        return [padded[position] for position in local]

    assert microbatch.input_ids_cp_sharded.tolist() == [local_part(tokens)]
    assert microbatch.position_ids.tolist() == [local_part(positions)]
    if "mtp_loss_mask" in batch:
        assert microbatch.mtp_loss_mask is not None
        assert microbatch.mtp_loss_mask.tolist() == [local_part(mtp_loss_mask)]
    else:
        assert microbatch.mtp_loss_mask is None


def _check_pack_logprobs(
    microbatch: ProcessedMicrobatch, logprobs: torch.Tensor
) -> None:
    """Check that every row of a tree holds the same copy of the prompt logprobs."""
    tree_ids = microbatch.data_dict[SHARED_PREFIX_TREE_IDS].tolist()
    prompt_lengths = microbatch.data_dict[SHARED_PREFIX_PROMPT_LENGTHS].tolist()
    lengths = microbatch.data_dict["input_lengths"].tolist()
    first_row = 0
    for position, tree_id in enumerate(tree_ids):
        if tree_id != tree_ids[first_row]:
            first_row = position
        prompt_length = prompt_lengths[position]
        assert torch.equal(
            logprobs[position, 1:prompt_length], logprobs[first_row, 1:prompt_length]
        )
        # Columns past the last token of a row stay 0.
        assert not logprobs[position, lengths[position] :].any()


def _check_case(
    case: _Case, *, chunk_size: Optional[int], temperature: float = 1.0
) -> None:
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    topology = (tp_rank, tp_size, cp_rank, cp_size)
    rows = _build_rows(case)
    batch = rows.data
    cfg = _policy_cfg(chunk_size, tp_size, cp_size)
    sampling = TrainingSamplingParams(temperature=temperature)
    truth = _truth(batch, temperature, tp_rank, tp_size)
    valid = batch["token_mask"][:, 1:] * batch["sample_mask"].unsqueeze(-1)
    device = torch.cuda.current_device()
    global_valid = (
        batch["sample_mask"].sum().to(device),
        valid.sum().to(device),
    )

    dense_loss, dense_grad = _dense_reference(
        batch, cfg, truth, temperature, global_valid, topology
    )
    assert abs(dense_loss - truth.loss) <= _LOSS_RTOL * truth.loss_scale
    assert _relative_error(dense_grad, truth.grad) < _GRAD_RTOL

    # Training: the driver's plan, run forward by forward as the worker does.
    data, shard = _plan_rank_batch(
        rows, case, cfg, bound_by_full_length=case.bound_by_full_length
    )
    _check_plan(case, rows, shard)
    model = _CausalHistoryModel(tp_rank, tp_size, cp_rank, cp_size)
    iterator, num_microbatches, micro_batch_size, width, _ = get_microbatch_iterator(
        data,
        cfg,
        mbs=data.size,
        straggler_timer=None,
        mtp_enabled=case.mtp_loss_mask,
    )
    forwards = list(zip(shard.micro_batch_indices[0], shard.micro_batch_lengths[0]))
    assert (num_microbatches, micro_batch_size, width) == (len(forwards), 1, _WIDTH)
    seen: list[ProcessedMicrobatch] = []
    stream = _recording(iterator, seen)
    processor = LossPostProcessor(
        _PolicyGradientLoss(), cfg, num_microbatches=num_microbatches
    )
    planned_loss = 0.0
    for (start, end), planned_length in forwards:
        output, loss_fn = forward_with_post_processing_fn(
            data_iterator=stream,
            model=model,
            post_processing_fn=processor,
            global_valid_seqs=global_valid[0],
            global_valid_toks=global_valid[1],
            sampling_params=sampling,
        )
        loss, _ = loss_fn(output)
        scaled = loss * cp_size / num_microbatches
        scaled.backward()
        planned_loss += float(scaled)
        microbatch = seen[-1]
        row_indices = shard.row_indices[start:end]
        # The rows of a forward keep the driver's order.
        assert torch.equal(
            microbatch.data_dict["input_ids"].cpu(), batch["input_ids"][row_indices]
        )
        # The worker builds a forward of the length that the driver planned.
        assert microbatch.input_ids_cp_sharded.shape[1] * cp_size == planned_length
        if shard.tree_ids[start] == UNSHARED_ROW:
            # A dense bin takes the ordinary packed path.
            assert microbatch.shared_prefix_params is None
            assert microbatch.packed_seq_params is not None
        else:
            _check_pack_inputs(microbatch, row_indices, rows, cfg, cp_rank, cp_size)
    assert next(stream, None) is None

    # The CP/TP-summed gradient must match too: a dropped CP normalization or a
    # non-differentiable CP reduction changes its scale without changing any logprob.
    planned_loss *= cp_size
    assert abs(planned_loss - dense_loss) <= _LOSS_RTOL * truth.loss_scale
    assert _relative_error(_cp_summed(model.table.grad, cp_size), dense_grad) < (
        _GRAD_RTOL
    )

    # Logprobs: forward-only, planned by packed length. The worker concatenates the
    # forwards in order, so row r of the result is row r of the rank's batch.
    data, shard = _plan_rank_batch(rows, case, cfg, bound_by_full_length=False)
    with torch.no_grad():
        iterator, num_microbatches, *_ = get_microbatch_iterator(
            data, cfg, mbs=data.size, straggler_timer=None
        )
        seen = []
        stream = _recording(iterator, seen)
        processor = LogprobsPostProcessor(cfg, sampling_params=sampling)
        outputs = []
        for _ in range(num_microbatches):
            output, post_fn = forward_with_post_processing_fn(
                data_iterator=stream,
                model=model,
                post_processing_fn=processor,
                sampling_params=sampling,
            )
            outputs.append(post_fn(output)[1]["logprobs"])
            if seen[-1].shared_prefix_params is not None:
                _check_pack_logprobs(seen[-1], outputs[-1])
        assert next(stream, None) is None
    logprobs = pad_and_concat(outputs, target_len=_WIDTH)
    assert logprobs.shape == (batch.size, _WIDTH)
    assert not logprobs[:, 0].any()
    for position, row in enumerate(shard.row_indices):
        length = int(batch["input_lengths"][row])
        torch.testing.assert_close(
            logprobs[position, 1:length].double().cpu(),
            truth.logprobs[row, : length - 1],
            atol=_LOGPROB_ATOL,
            rtol=0.0,
        )


def _unavailable(reason: str) -> None:
    """Skip a test that lacks its GPUs, or fail it where the lane must run it."""
    if os.environ.get("NRL_REQUIRE_SHARED_PREFIX_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def single_rank_model_parallel(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """TP1/CP1 Megatron parallel state in this process on one GPU."""
    if not torch.cuda.is_available():
        _unavailable("the microbatch iterator moves batches to CUDA")
    created = False
    if not dist.is_initialized():
        init_file = tmp_path_factory.mktemp("shared_prefix_oracle") / "pg_init"
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl", rank=0, world_size=1, init_method=f"file://{init_file}"
        )
        created = True
    elif dist.get_world_size() != 1:
        _unavailable("needs a single-rank default process group")
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=1
    )
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        if created:
            dist.destroy_process_group()


@pytest.mark.parametrize("chunk_size", [None, 7], ids=["unchunked", "chunk7"])
@pytest.mark.parametrize("case_name", list(_CASES))
def test_packs_match_dense_oracle(
    single_rank_model_parallel: None, case_name: str, chunk_size: Optional[int]
) -> None:
    _check_case(_CASES[case_name], chunk_size=chunk_size)


def test_packs_match_dense_oracle_with_temperature(
    single_rank_model_parallel: None,
) -> None:
    _check_case(_CASES["placeholder"], chunk_size=7, temperature=0.7)


def _distributed_oracle(
    rank: int, world_size: int, *, tp_size: int, cp_size: int
) -> None:
    del rank, world_size
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        for case in _CASES.values():
            for chunk_size in (None, 7):
                _check_case(case, chunk_size=chunk_size)
        _check_case(_CASES["placeholder"], chunk_size=7, temperature=0.7)
    finally:
        parallel_state.destroy_model_parallel()


@pytest.mark.parametrize(
    "tp_size, cp_size", [(1, 2), (2, 1), (2, 2)], ids=["tp1cp2", "tp2cp1", "tp2cp2"]
)
def test_packs_match_dense_oracle_distributed(
    distributed_test_runner: Callable[..., None], tp_size: int, cp_size: int
) -> None:
    world_size = tp_size * cp_size
    # run_unit.sh guarantees two GPUs. The runner skips a variant that needs more.
    if world_size <= 2 and torch.cuda.device_count() < world_size:
        _unavailable(f"needs {world_size} GPUs")
    distributed_test_runner(
        functools.partial(_distributed_oracle, tp_size=tp_size, cp_size=cp_size),
        world_size=world_size,
    )
