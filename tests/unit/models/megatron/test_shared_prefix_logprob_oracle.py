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
"""Exactness oracle for NeMo-RL's shared-prefix logprob and loss consumer.

The model is a table lookup: the logits at a token are the row of a trainable
table selected by a hash of that token's causal history. For a prompt token the
history is the prompt prefix; for a completion token it is the prompt plus the
branch's own completion prefix. Every physical token of a shared star or forest
therefore gets the logits that the dense row gets at the same logical position.
A correct consumer must reproduce the dense per-row logprobs, the dense loss and
the dense table gradient. Both paths are also checked against an fp64
single-process computation of the same loss, so a scale error common to both
fails too.

Shared batches run through the worker's entry points: ``get_microbatch_iterator``,
``forward_with_post_processing_fn`` (MCore layout lowering, temperature, chunked
extraction), ``LossPostProcessor`` / ``LogprobsPostProcessor`` and the worker's
source-order restore. Dense fallback units take the conventional packed loss path.

TP1/CP1 runs in-process on one GPU. The TP/CP variants spawn one process per GPU
and skip when fewer GPUs are visible. Run them with four visible GPUs, e.g.
``CUDA_VISIBLE_DEVICES=0,1,2,3 pytest --mcore-only <this file> -k distributed``.

Cross-repo and cross-module API assumptions. Other review fixes edit these
modules concurrently; update this file if one of them changes:

- ``nemo_rl.models.megatron.data``: ``get_microbatch_iterator(data, cfg, mbs,
  straggler_timer, shared_prefix_bin_capacity=, shared_prefix_execution_units=,
  shared_prefix_stage=)`` returns ``(iterator, num_units, 1, width,
  padded_seq_length)`` in shared mode, stamps ``SHARED_PREFIX_SOURCE_ROW_INDEX``
  on every unit, and keeps unit rows in ``layout.row_indices`` order. The units
  come from ``plan_shared_prefix_execution_units(data, cfg=, bin_capacity=,
  forward_only=)``.
- ``nemo_rl.models.megatron.shared_prefix_dense_bins``: ``plan_dense_training_bins``
  and ``share_prefixes_in_dense_training_bins(data, units, cfg=, bin_capacity=)``.
- ``nemo_rl.models.megatron.train``: a shared unit calls the model with a
  ``shared_prefix_layout`` keyword (MCore ``SharedPrefixLayout`` with
  ``prefix_len``/``completion_lens``, or ``SharedPrefixForestLayout.roots``) and
  without ``packed_seq_params``, and ``forward_with_post_processing_fn`` returns
  the restored ``[rows, width - 1]`` logprobs for it.
- ``nemo_rl.models.policy.workers.megatron_policy_worker._restore_logprobs_in_source_order(
  outputs, sequence_length=, expected_rows=)``.
- ``megatron.rl`` (MLM #7914): ``SharedPrefixTensorBin.layout`` with
  ``row_indices``, ``physical_total_length`` and ``iter_roots()`` yielding roots
  with ``prompt_length``, ``row_indices``, ``branch_starts`` (root-relative) and
  ``completion_lengths``; ``packed_input_ids``.
- ``nemo_rl.models.policy.SharedPrefixTrainingConfig`` accepts the flag sets in
  ``_CASES`` (valid under the stricter flag-combination validator as well).
"""

import functools
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")
pytest.importorskip("megatron.rl.shared_prefix_tensors")

from megatron.core import parallel_state  # noqa: E402

from nemo_rl.algorithms.logits_sampling_utils import (  # noqa: E402
    TrainingSamplingParams,
)
from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType  # noqa: E402
from nemo_rl.data.packing.shared_prefix_metadata import (  # noqa: E402
    SHARED_PREFIX_EXECUTION_SLOT,
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.megatron.data import (  # noqa: E402
    SHARED_PREFIX_SOURCE_ROW_INDEX,
    ProcessedMicrobatch,
    get_microbatch_iterator,
    plan_shared_prefix_execution_units,
)
from nemo_rl.models.megatron.train import (  # noqa: E402
    LogprobsPostProcessor,
    LossPostProcessor,
    forward_with_post_processing_fn,
)
from nemo_rl.utils.sequence_lengths import to_cpu_int_tuple  # noqa: E402

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
# Observed maxima on GB200 at TP{1,2} x CP{1,2}: logprob error 1.3e-6 against
# the fp64 reference (fp32 rounding; the model has no matmul, so TF32 cannot
# enter), gradient relative error 7e-8. A misrouted logprob or a CP/TP scale
# error is orders of magnitude larger.
_LOGPROB_ATOL = 1e-5
_LOSS_RTOL = 1e-5
_GRAD_RTOL = 1e-6


@dataclass(frozen=True)
class _Case:
    # (group id, prompt length, completions); rows are interleaved at random.
    groups: tuple[tuple[str, int, int], ...]
    seed: int
    shared_prefix: dict[str, Any]
    bin_capacity: int
    # Alternate siblings between two execution slots (prompt fragments).
    split_slots: bool = False
    # Groups that get one invalid placeholder row (prompt length 0, masked).
    placeholders: tuple[str, ...] = ()
    # TransferQueue delivers group IDs as a numpy object array.
    numpy_group_ids: bool = False
    mtp_loss_mask: bool = False
    # Structure the case must exercise: "stars", "forest" or "fallback".
    expect: Optional[str] = None


_PACK = {"mode": "train", "pack_groups": True, "repack_groups": True}
_DENSE_BINS = {**_PACK, "align_data_parallel": True, "training_dense_bins": True}
_SPEC_A = (("g0", 5, 3), ("g1", 7, 4), ("g2", 9, 2), ("g3", 3, 5))
# A singleton group (G=1), more siblings than one star holds, and a one-token prompt.
_SPEC_B = (("g0", 5, 3), ("big", 6, 17), ("one", 4, 1), ("p1", 1, 4), ("g4", 8, 2))
_CASES = {
    "stars": _Case(
        _SPEC_A, 11, {"mode": "train"}, 512, mtp_loss_mask=True, expect="stars"
    ),
    "forest": _Case(_SPEC_A, 12, _PACK, 512, numpy_group_ids=True, expect="forest"),
    "forest_small_bins": _Case(_SPEC_A, 13, _PACK, 160, expect="forest"),
    "split_slots": _Case(_SPEC_A, 21, _PACK, 512, split_slots=True),
    "dense_bins": _Case(_SPEC_A, 14, _DENSE_BINS, 160),
    "fallbacks": _Case(
        _SPEC_B, 22, _PACK, 1024, placeholders=("g4",), expect="fallback"
    ),
    "fallbacks_dense_bins": _Case(_SPEC_B, 23, _DENSE_BINS, 256),
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


def _star_features(ids: Sequence[int], layout: Any) -> list[int]:
    """Causal-history features of the MCore lowering ``[prefix, branch, ...]``."""
    features = [_feature(("pad", position)) for position in range(len(ids))]
    offset = 0
    for root in getattr(layout, "roots", (layout,)):
        prompt = list(ids[offset : offset + root.prefix_len])
        features[offset : offset + root.prefix_len] = _causal_features(prompt)
        start = offset + root.prefix_len
        for length in root.completion_lens:
            branch = prompt + list(ids[start : start + length])
            features[start : start + length] = _causal_features(branch)[
                root.prefix_len :
            ]
            start += length
        offset = start
    assert offset <= len(ids)
    return features


def _table_shard(tp_rank: int, tp_size: int) -> torch.Tensor:
    width = _VOCAB // tp_size
    shard = _TABLE[:, tp_rank * width : (tp_rank + 1) * width]
    return shard.to(device=torch.cuda.current_device(), dtype=torch.float32)


class _CausalHistoryModel(torch.nn.Module):
    """Logits from a table row keyed by each token's causal history.

    It reconstructs the global token sequence from the CP-local inputs and the
    layout it is given, exactly as attention would see it: a shared unit as the
    MCore star/forest lowering, a fallback unit as THD sequences.
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
        shared_prefix_layout: Any = None,
        packed_seq_params: Any = None,
        **unused: Any,
    ) -> torch.Tensor:
        del position_ids, attention_mask, unused
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1, input_ids.shape
        if shared_prefix_layout is not None:
            assert packed_seq_params is None
            segments = [(0, input_ids.shape[1] * self.cp_size)]
        else:
            assert packed_seq_params is not None, "shared mode ran an unpacked unit"
            boundaries = to_cpu_int_tuple(packed_seq_params.cu_seqlens_q_padded)
            segments = list(zip(boundaries[:-1], boundaries[1:]))
        local = _local_positions(segments, self.cp_rank, self.cp_size)
        assert len(local) == input_ids.shape[1], (len(local), input_ids.shape)
        ids = self._global_ids(input_ids[0], segments)
        if shared_prefix_layout is not None:
            features = _star_features(ids, shared_prefix_layout)
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


def _build_batch(case: _Case) -> BatchedDataDict[Any]:
    rng = np.random.default_rng(case.seed)
    rows: list[tuple[str, int, list[int], bool]] = []
    for group, prompt_length, completions in case.groups:
        prompt = rng.integers(1, _VOCAB, prompt_length).tolist()
        for _ in range(completions):
            completion_length = int(rng.integers(1, _WIDTH - prompt_length + 1))
            completion = rng.integers(0, _VOCAB, completion_length).tolist()
            rows.append((group, prompt_length, prompt + completion, True))
        if group in case.placeholders:
            # Reassembler placeholder: one pad token, prompt length 0, masked.
            rows.append((group, 0, [0], False))
    rows = [rows[index] for index in rng.permutation(len(rows))]
    size = len(rows)
    input_ids = torch.zeros(size, _WIDTH, dtype=torch.long)
    token_mask = torch.zeros(size, _WIDTH)
    for index, (_, prompt_length, tokens, valid) in enumerate(rows):
        input_ids[index, : len(tokens)] = torch.tensor(tokens)
        if valid:
            token_mask[index, prompt_length : len(tokens)] = 1.0
    group_ids = [group for group, *_ in rows]
    slots = [
        group_ids[:index].count(group) % 2 if case.split_slots else 0
        for index, group in enumerate(group_ids)
    ]
    sample_mask = torch.tensor([1.0 if valid else 0.0 for *_, valid in rows])
    data: dict[str, Any] = {
        "input_ids": input_ids,
        "input_lengths": torch.tensor([len(tokens) for _, _, tokens, _ in rows]),
        SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor([length for _, length, *_ in rows]),
        SHARED_PREFIX_GROUP_ID: (
            np.asarray(group_ids, dtype=object) if case.numpy_group_ids else group_ids
        ),
        SHARED_PREFIX_EXECUTION_SLOT: slots,
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "advantages": torch.randn(
            size, _WIDTH, generator=torch.Generator().manual_seed(case.seed)
        ),
    }
    if case.mtp_loss_mask:
        data["mtp_loss_mask"] = token_mask * sample_mask.unsqueeze(-1)
    return BatchedDataDict(data)


def _policy_cfg(
    case: _Case, chunk_size: Optional[int], tp_size: int, cp_size: int
) -> dict[str, Any]:
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
        "shared_prefix_training": dict(case.shared_prefix),
        "logprob_chunk_size": chunk_size,
    }


def _plan_training_units(
    case: _Case, batch: BatchedDataDict[Any], cfg: dict[str, Any]
) -> tuple[Any, ...]:
    if not case.shared_prefix.get("training_dense_bins"):
        return plan_shared_prefix_execution_units(
            batch, cfg=cfg, bin_capacity=case.bin_capacity
        )
    from nemo_rl.models.megatron.shared_prefix_dense_bins import (
        plan_dense_training_bins,
        share_prefixes_in_dense_training_bins,
    )

    units = plan_dense_training_bins(batch, cfg=cfg, bin_capacity=case.bin_capacity)
    return share_prefixes_in_dense_training_bins(
        batch, units, cfg=cfg, bin_capacity=case.bin_capacity
    )


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
    cfg: dict[str, Any],
    truth: _Truth,
    temperature: float,
    global_valid: tuple[torch.Tensor, torch.Tensor],
    topology: tuple[int, int, int, int],
) -> tuple[float, torch.Tensor]:
    """NeMo's conventional unpacked loss on CP-zigzag logits, as one microbatch."""
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


def _check_shared_unit(
    microbatch: ProcessedMicrobatch,
    logprobs: torch.Tensor,
    batch: BatchedDataDict[Any],
    truth: _Truth,
    cp_rank: int,
    cp_size: int,
) -> None:
    shared = microbatch.shared_prefix
    assert shared is not None
    layout = shared.tensor_bin.layout
    rows = list(layout.row_indices)
    # Unit rows are stamped with their source index and follow layout order.
    assert microbatch.data_dict[SHARED_PREFIX_SOURCE_ROW_INDEX].tolist() == rows
    assert torch.equal(
        microbatch.data_dict["input_ids"].cpu(), batch["input_ids"][rows]
    )

    # Model input: the minimally padded packed star in zigzag CP order.
    padded_length = shared.padded_total_length
    assert padded_length is not None
    assert padded_length % shared.padding_multiple == 0
    assert 0 <= padded_length - layout.physical_total_length < shared.padding_multiple
    padded_ids = torch.nn.functional.pad(
        shared.tensor_bin.packed_input_ids.cpu(),
        (0, padded_length - layout.physical_total_length),
    )
    local = _local_positions([(0, padded_length)], cp_rank, cp_size)
    assert microbatch.input_ids_cp_sharded[0].tolist() == padded_ids[local].tolist()

    if "mtp_loss_mask" in batch:
        # The MTP mask follows each completion token; prompts and padding are 0.
        expected = torch.zeros(padded_length)
        for offset, root in layout.iter_roots():
            for branch, row in enumerate(root.row_indices):
                start = offset + root.branch_starts[branch]
                length = root.completion_lengths[branch]
                prompt = root.prompt_length
                expected[start : start + length] = batch["mtp_loss_mask"][
                    row, prompt : prompt + length
                ]
        assert microbatch.mtp_loss_mask is not None
        assert torch.equal(
            microbatch.mtp_loss_mask[0].float().cpu(), expected[local].float()
        )

    # Restored logprobs equal the dense per-row reference; padding columns are 0.
    assert logprobs.shape == (len(rows), _WIDTH - 1)
    for local_row, row in enumerate(rows):
        length = int(batch["input_lengths"][row])
        torch.testing.assert_close(
            logprobs[local_row, : length - 1].detach().double().cpu(),
            truth.logprobs[row, : length - 1],
            atol=_LOGPROB_ATOL,
            rtol=0.0,
        )
        assert not logprobs[local_row, length - 1 :].detach().any()


def _check_case(
    case: _Case, *, chunk_size: Optional[int], temperature: float = 1.0
) -> None:
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    topology = (tp_rank, tp_size, cp_rank, cp_size)
    batch = _build_batch(case)
    cfg = _policy_cfg(case, chunk_size, tp_size, cp_size)
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

    # Training stage, with units planned once and handed to the iterator as the
    # worker does.
    units = _plan_training_units(case, batch, cfg)
    model = _CausalHistoryModel(tp_rank, tp_size, cp_rank, cp_size)
    iterator, num_units, micro_batch_size, width, padded_seq_length = (
        get_microbatch_iterator(
            batch,
            cfg,
            mbs=batch.size,
            straggler_timer=None,
            shared_prefix_bin_capacity=case.bin_capacity,
            shared_prefix_execution_units=units,
            shared_prefix_stage="train",
        )
    )
    assert (num_units, micro_batch_size, width) == (len(units), 1, _WIDTH)
    assert padded_seq_length % cfg["make_sequence_length_divisible_by"] == 0
    seen: list[ProcessedMicrobatch] = []
    stream = _recording(iterator, seen)
    processor = LossPostProcessor(
        _PolicyGradientLoss(), cfg, num_microbatches=num_units
    )
    shared_loss = 0.0
    for _ in range(num_units):
        output, loss_fn = forward_with_post_processing_fn(
            data_iterator=stream,
            model=model,
            post_processing_fn=processor,
            global_valid_seqs=global_valid[0],
            global_valid_toks=global_valid[1],
            sampling_params=sampling,
        )
        loss, _ = loss_fn(output)
        scaled = loss * cp_size / num_units
        scaled.backward()
        shared_loss += float(scaled)
        microbatch = seen[-1]
        assert microbatch.input_ids_cp_sharded.shape[1] * cp_size <= padded_seq_length
        if microbatch.shared_prefix is not None:
            _check_shared_unit(microbatch, output, batch, truth, cp_rank, cp_size)
        else:
            assert microbatch.packed_seq_params is not None
    assert next(stream, None) is None

    source_rows = sorted(
        row
        for microbatch in seen
        for row in microbatch.data_dict[SHARED_PREFIX_SOURCE_ROW_INDEX].tolist()
    )
    assert source_rows == list(range(batch.size))
    roots = [
        None
        if microbatch.shared_prefix is None
        else len(tuple(microbatch.shared_prefix.tensor_bin.layout.iter_roots()))
        for microbatch in seen
    ]
    if case.expect == "stars":
        assert roots and all(count == 1 for count in roots), roots
    elif case.expect == "forest":
        assert any(count is not None and count > 1 for count in roots), roots
    elif case.expect == "fallback":
        assert None in roots and any(count is not None for count in roots), roots

    # Summed unit losses equal the dense loss, and the CP/TP-summed gradient
    # matches it too: a dropped CP normalization or a non-differentiable CP
    # reduction changes the gradient scale without changing any logprob.
    shared_loss *= cp_size
    assert abs(shared_loss - dense_loss) <= _LOSS_RTOL * truth.loss_scale
    assert _relative_error(_cp_summed(model.table.grad, cp_size), dense_grad) < (
        _GRAD_RTOL
    )

    # Logprob stage: forward-only units, and the worker's restore returns the
    # conventional [batch, width] layout.
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _restore_logprobs_in_source_order,
    )

    units = plan_shared_prefix_execution_units(
        batch, cfg=cfg, bin_capacity=case.bin_capacity, forward_only=True
    )
    with torch.no_grad():
        iterator, num_units, *_ = get_microbatch_iterator(
            batch,
            cfg,
            mbs=batch.size,
            straggler_timer=None,
            shared_prefix_bin_capacity=case.bin_capacity,
            shared_prefix_execution_units=units,
            shared_prefix_stage="logprobs",
        )
        processor = LogprobsPostProcessor(cfg, sampling_params=sampling)
        outputs = []
        for _ in range(num_units):
            output, post_fn = forward_with_post_processing_fn(
                data_iterator=iterator,
                model=model,
                post_processing_fn=processor,
                sampling_params=sampling,
            )
            outputs.append(post_fn(output)[1])
        assert next(iterator, None) is None
    restored = _restore_logprobs_in_source_order(
        outputs, sequence_length=_WIDTH, expected_rows=batch.size
    )
    assert not restored[:, 0].any()
    for row in range(batch.size):
        length = int(batch["input_lengths"][row])
        torch.testing.assert_close(
            restored[row, 1:length].double().cpu(),
            truth.logprobs[row, : length - 1],
            atol=_LOGPROB_ATOL,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    "unit_rows, error",
    [
        (((2, 0), (1,)), None),
        (((0, 0), (2,)), "did not cover every source row exactly once"),
        (((0,), (2,)), "unexpected row count"),
    ],
    ids=["permuted", "duplicate_row", "missing_row"],
)
def test_logprob_restore_returns_source_order_or_fails(unit_rows, error):
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _restore_logprobs_in_source_order,
    )

    # Each row's logprobs hold its source index; a fallback unit is wider.
    outputs = [
        {
            "logprobs": torch.tensor(rows, dtype=torch.float32)[:, None].expand(
                -1, width
            ),
            SHARED_PREFIX_SOURCE_ROW_INDEX: torch.tensor(rows),
        }
        for rows, width in zip(unit_rows, (3, 4))
    ]
    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            _restore_logprobs_in_source_order(
                outputs, sequence_length=4, expected_rows=3
            )
        return
    restored = _restore_logprobs_in_source_order(
        outputs, sequence_length=4, expected_rows=3
    )
    assert restored.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 0]]


@pytest.fixture(scope="module")
def single_rank_model_parallel(tmp_path_factory):
    """TP1/CP1 Megatron parallel state in this process on one GPU."""
    if not torch.cuda.is_available():
        pytest.skip("the shared-prefix iterator moves batches to CUDA")
    created = False
    if not dist.is_initialized():
        init_file = tmp_path_factory.mktemp("shared_prefix_oracle") / "pg_init"
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl", rank=0, world_size=1, init_method=f"file://{init_file}"
        )
        created = True
    elif dist.get_world_size() != 1:
        pytest.skip("needs a single-rank default process group")
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
def test_shared_prefix_consumer_matches_dense_oracle(
    single_rank_model_parallel, case_name, chunk_size
):
    _check_case(_CASES[case_name], chunk_size=chunk_size)


def test_shared_prefix_consumer_matches_dense_oracle_with_temperature(
    single_rank_model_parallel,
):
    _check_case(_CASES["fallbacks"], chunk_size=7, temperature=0.7)


@pytest.mark.parametrize(
    "mode, stage",
    [(None, "train"), ("disabled", "train"), ("dense", "train"), ("logprobs", "train")],
    ids=["absent", "disabled", "dense", "logprobs_at_train_stage"],
)
def test_inactive_shared_prefix_modes_keep_the_conventional_iterator(
    single_rank_model_parallel, mode, stage
):
    case = _CASES["split_slots"]
    shared_batch = _build_batch(case)
    conventional = BatchedDataDict(
        {
            key: value
            for key, value in shared_batch.items()
            if key
            not in (
                SHARED_PREFIX_PROMPT_LENGTHS,
                SHARED_PREFIX_GROUP_ID,
                SHARED_PREFIX_EXECUTION_SLOT,
            )
        }
    )
    baseline_cfg = _policy_cfg(case, None, 1, 1)
    baseline_cfg["sequence_packing"]["enabled"] = False
    del baseline_cfg["shared_prefix_training"]
    cfg = _policy_cfg(case, None, 1, 1)
    cfg["sequence_packing"]["enabled"] = False
    if mode is None:
        del cfg["shared_prefix_training"]
    else:
        cfg["shared_prefix_training"] = {"mode": mode}

    def processed(data, policy_cfg):
        iterator, *metadata = get_microbatch_iterator(
            data.select_indices(list(range(data.size))),
            policy_cfg,
            mbs=7,
            straggler_timer=None,
            shared_prefix_stage=stage,
        )
        return list(iterator), metadata

    expected, expected_metadata = processed(conventional, baseline_cfg)
    actual, actual_metadata = processed(shared_batch, cfg)
    assert actual_metadata == expected_metadata
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.shared_prefix is None
        assert SHARED_PREFIX_SOURCE_ROW_INDEX not in got.data_dict
        for field in ("input_ids_cp_sharded", "attention_mask", "position_ids"):
            assert torch.equal(getattr(got, field), getattr(want, field)), field
    with pytest.raises(ValueError, match="only meaningful in shared-prefix"):
        get_microbatch_iterator(
            shared_batch,
            cfg,
            mbs=7,
            straggler_timer=None,
            shared_prefix_execution_units=(),
            shared_prefix_stage=stage,
        )


def _distributed_oracle(rank: int, world_size: int, *, tp_size: int, cp_size: int):
    del rank, world_size
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        for case in _CASES.values():
            for chunk_size in (None, 7):
                _check_case(case, chunk_size=chunk_size)
        _check_case(_CASES["fallbacks"], chunk_size=7, temperature=0.7)
    finally:
        parallel_state.destroy_model_parallel()


@pytest.mark.parametrize(
    "tp_size, cp_size", [(1, 2), (2, 1), (2, 2)], ids=["tp1cp2", "tp2cp1", "tp2cp2"]
)
def test_shared_prefix_consumer_matches_dense_oracle_distributed(
    distributed_test_runner, tp_size, cp_size
):
    distributed_test_runner(
        functools.partial(_distributed_oracle, tp_size=tp_size, cp_size=cp_size),
        world_size=tp_size * cp_size,
    )
