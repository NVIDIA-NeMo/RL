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
"""Score centering sees the same captured top-k whether it is fetched padded or as CP blocks.

The trainer's fetch keeps context-parallel rank ``r`` block ``r`` of every padded top-k row
(``TQWorkerMixin._topk_split``) and each microbatch gathers its rows back
(``gather_topk_cp_blocks``). At K = 32 with every head full (top_p 1.0: no capture floor),
the packed loss input, the score-centering term, the clipped PG loss with token-level TIS, its
metrics and the logits' gradient must be bit-identical to those of the padded fetch. Rows of
very different lengths, segmented by tool spans that carry the zero head and stay out of
token_mask; the forward pad target is no multiple of 2 * CP.
"""

import itertools
from datetime import timedelta

import pytest
import torch

_K, _V = 32, 48
_LENGTHS = (3, 33, 9, 2, 17, 26)
_PAD_TO = 37
_MICROBATCHES = ((0, 2), (2, 5), (5, 6))


def _wire(seed: int):
    """The rows as the data plane hands them to the trainer (jagged)."""
    from tensordict import TensorDict

    generator = torch.Generator().manual_seed(seed)
    names = (
        "input_ids",
        "token_mask",
        "generation_topk_ids",
        "generation_topk_logprobs",
        "advantages",
        "prev_logprobs",
        "generation_logprobs",
    )
    columns = {name: [] for name in names + ("reference_policy_logprobs",)}
    for length in _LENGTHS:
        input_ids = torch.randint(0, _V, (length,), generator=generator)
        token_mask = torch.zeros(length)
        ids = torch.zeros(length, _K, dtype=torch.int32)
        logprobs = torch.zeros(length, _K)
        for s in range(min(2, length - 1), length):
            if s % 7 in (5, 6):
                continue
            full = torch.randn(
                _V, generator=generator, dtype=torch.float64
            ).log_softmax(-1)
            head = full.topk(_K)
            ids[s] = head.indices.to(torch.int32)
            logprobs[s] = head.values.float()
            input_ids[s] = torch.multinomial(full.exp(), 1, generator=generator)
            token_mask[s] = 1.0
        generated = token_mask.bool()
        sampled = torch.randn(length, generator=generator) * 0.1 - 1.0
        rows = (
            input_ids,
            token_mask,
            ids,
            logprobs,
            torch.randn(length, generator=generator),
            torch.where(
                generated,
                sampled + torch.randn(length, generator=generator) * 0.05,
                0.0,
            ),
            torch.where(generated, sampled, 0.0),
        )
        for name, row in zip(names, rows):
            columns[name].append(row)
        columns["reference_policy_logprobs"].append(torch.zeros(length))
    return TensorDict(
        {
            **{
                name: torch.nested.as_nested_tensor(rows, layout=torch.jagged)
                for name, rows in columns.items()
            },
            "input_lengths": torch.tensor(_LENGTHS),
            "sample_mask": torch.ones(len(_LENGTHS)),
        },
        batch_size=len(_LENGTHS),
    )


def _consumers(data, row_logits, *, cu_seqlens, cu_seqlens_padded, groups, layout):
    """Everything score centering and the PG loss compute from one microbatch, gradients included."""
    from nemo_rl.algorithms.loss import (
        ClippedPGLossConfig,
        ClippedPGLossFn,
        prepare_packed_loss_input,
    )
    from nemo_rl.algorithms.loss.score_centering import score_centering_term
    from nemo_rl.algorithms.loss.utils import _pack_input_ids

    tp_rank, tp_size, tp_group, cp_rank, cp_size, cp_group = groups
    local_vocab = _V // tp_size
    shard = slice(tp_rank * local_vocab, (tp_rank + 1) * local_vocab)
    leaf = (
        _pack_input_ids(
            row_logits[..., shard],
            cu_seqlens,
            cu_seqlens_padded,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_layout=layout,
        )
        .clone()
        .requires_grad_()
    )
    active = (data["token_mask"] * data["sample_mask"].unsqueeze(-1))[:, 1:]
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            ratio_clip_min=0.2,
            ratio_clip_max=0.28,
            ratio_clip_c=3.0,
            use_on_policy_kl_approximation=True,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio=2.5,
            truncated_importance_sampling_ratio_min=None,
            token_level_loss=True,
            score_centering=True,
        )
    )
    loss_input, prepared = prepare_packed_loss_input(
        leaf,
        data,
        loss_fn,
        cu_seqlens,
        cu_seqlens_padded,
        vocab_parallel_rank=tp_rank,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
        chunk_size=3,
        cp_layout=layout,
    )
    term, head_mass = score_centering_term(
        loss_input["next_token_topk_logprobs"],
        prepared["generation_topk_logprobs"][:, 1:],
        (0.0, 2.5),
    )
    loss, metrics = loss_fn(
        data=prepared,
        global_valid_seqs=torch.tensor(float(data["sample_mask"].sum())),
        global_valid_toks=torch.tensor(float(active.sum())),
        **loss_input,
    )
    (grad,) = torch.autograd.grad(loss, leaf)
    out = {
        "sampled": loss_input["next_token_logprobs"].detach(),
        "head": loss_input["next_token_topk_logprobs"].detach(),
        "term": term.detach(),
        "head_mass": head_mass,
        "loss": loss.detach(),
        "grad": grad,
    }
    out.update({f"metric/{k}": torch.as_tensor(v) for k, v in metrics.items()})
    return out


def _run(rank, tp_size, cp_size, init_file):
    from nemo_rl.data_plane import materialize
    from nemo_rl.data_plane.schema import GENERATION_TOPK_FIELDS
    from nemo_rl.data_plane.worker_mixin import _topk_blocks, gather_topk_cp_blocks

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=300),
    )
    try:
        tp_rank, cp_rank = rank % tp_size, rank // tp_size
        tp_groups = [
            torch.distributed.new_group([c * tp_size + t for t in range(tp_size)])
            for c in range(cp_size)
        ]
        cp_groups = [
            torch.distributed.new_group([c * tp_size + t for c in range(cp_size)])
            for t in range(tp_size)
        ]
        groups = (
            tp_rank,
            tp_size,
            tp_groups[cp_rank],
            cp_rank,
            cp_size,
            cp_groups[tp_rank],
        )
        td = _wire(seed=3)
        padded = dict(layout="padded", pad_value_dict={}, pad_to_seqlen=_PAD_TO)
        whole = materialize(td, **padded)
        # The replica group (TP x CP) fetches through its source, rank 0.
        fetched = materialize(td.exclude(*GENERATION_TOPK_FIELDS), **padded)
        fetched.update(
            _topk_blocks(
                {f: td.get(f) for f in GENERATION_TOPK_FIELDS} if rank == 0 else {},
                split=(cp_rank, cp_size),
                pad_value_dict={},
                pad_to_seqlen=_PAD_TO,
                is_leader=rank == 0,
                src=0,
                group=torch.distributed.group.WORLD,
            )
        )
        logits = torch.randn(
            len(_LENGTHS), _PAD_TO, _V, generator=torch.Generator().manual_seed(11)
        )
        for start, end in _MICROBATCHES:
            before = whole.slice(start, end)
            after = fetched.slice(start, end)
            gather_topk_cp_blocks(after, groups[5])
            assert set(after) == set(before)
            for field in before:
                assert torch.equal(after[field], before[field]), field

            lengths = _LENGTHS[start:end]
            padded_lengths = [
                -(-length // (2 * cp_size)) * 2 * cp_size + 2 * cp_size * (i == 0)
                for i, length in enumerate(lengths)
            ]
            cu_seqlens = tuple(itertools.accumulate(lengths, initial=0))
            cu_seqlens_padded = tuple(itertools.accumulate(padded_lengths, initial=0))
            for layout, cu_q in itertools.product(
                ("zigzag", "contiguous"), (cu_seqlens, cu_seqlens_padded)
            ):
                kwargs = dict(
                    cu_seqlens=cu_q,
                    cu_seqlens_padded=cu_seqlens_padded,
                    groups=groups,
                    layout=layout,
                )
                old = _consumers(before, logits[start:end], **kwargs)
                new = _consumers(after, logits[start:end], **kwargs)
                assert old.keys() == new.keys()
                assert old["metric/score_centering_head_mass"] > 0
                for key in old:
                    torch.testing.assert_close(
                        new[key], old[key], rtol=0, atol=0, equal_nan=True
                    )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("tp_size", "cp_size"), [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4)]
)
def test_cp_blocks_feed_score_centering_unchanged(tmp_path, tp_size, cp_size):
    torch.multiprocessing.spawn(
        _run,
        args=(tp_size, cp_size, str(tmp_path / "gloo-init")),
        nprocs=tp_size * cp_size,
        join=True,
    )
