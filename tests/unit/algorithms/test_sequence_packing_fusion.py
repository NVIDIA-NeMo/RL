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
"""
Unit tests to ensure SequencePackingFusionLossWrapper produces identical results
to SequencePackingLossWrapper.

Uses distributed_test_runner (torch.multiprocessing.spawn) instead of Ray actors
so that pytest + code coverage work correctly.

For loss function, currently only supports ClippedPGLossFn.
"""

import functools

import pytest
import torch

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss import (
    ClippedPGLossFn,
    SequencePackingFusionLossWrapper,
    SequencePackingLossWrapper,
    prepare_loss_input,
    prepare_packed_loss_input,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _setup_2d_process_groups(rank, world_size, cp_size, tp_size):
    """Create 2D (cp, tp) process groups.

    Rank layout (outer cp, inner tp):
      [[0, 1, ..., tp_size-1],
       [tp_size, ..., 2*tp_size-1],
       ...]
    """
    cp_groups = []
    tp_groups = []

    for tp_rank in range(tp_size):
        ranks = [cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)]
        cp_groups.append(torch.distributed.new_group(ranks=ranks))

    for cp_rank in range(cp_size):
        ranks = [cp_rank * tp_size + tp_rank for tp_rank in range(tp_size)]
        tp_groups.append(torch.distributed.new_group(ranks=ranks))

    my_tp_rank = rank % tp_size
    my_cp_rank = rank // tp_size
    cp_group = cp_groups[my_tp_rank]
    tp_group = tp_groups[my_cp_rank]
    return my_cp_rank, my_tp_rank, cp_group, tp_group


def _build_test_case(cp_size, tp_size, my_tp_rank, cp_group):
    """Build a small packed batch with CP-aware packing."""
    from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
    from nemo_rl.models.megatron.data import _pack_sequences_for_megatron

    device = torch.device("cuda")
    torch.manual_seed(42)

    batch_size = 4
    max_seq_len = 512
    if max_seq_len % (2 * cp_size) != 0:
        max_seq_len = (max_seq_len // (2 * cp_size) + 1) * (2 * cp_size)

    vocab_size_total = 512
    assert vocab_size_total % tp_size == 0
    vocab_size_local = vocab_size_total // tp_size

    seq_lengths = torch.tensor(
        [max_seq_len // 4, max_seq_len // 2, max_seq_len // 3, max_seq_len * 3 // 4],
        dtype=torch.int32,
        device=device,
    )

    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
    token_mask = torch.zeros(
        batch_size, max_seq_len, dtype=torch.float32, device=device
    )
    for i in range(batch_size):
        L = int(seq_lengths[i].item())
        input_ids[i, :L] = torch.randint(0, vocab_size_total, (L,), device=device)
        token_mask[i, :L] = 1.0

    sample_mask = torch.ones(batch_size, dtype=torch.float32, device=device)
    advantages = 0.1 * torch.randn(batch_size, max_seq_len, device=device)
    prev_logprobs = 0.1 * torch.randn(batch_size, max_seq_len, device=device)
    generation_logprobs = 0.1 * torch.randn(batch_size, max_seq_len, device=device)
    reference_policy_logprobs = generation_logprobs.clone()

    data_dict = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": seq_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": advantages,
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
            "reference_policy_logprobs": reference_policy_logprobs,
        }
    )

    pad_to_multiple = cp_size * 2
    (
        _packed_input_ids,
        _packed_input_ids_cp,
        _packed_seq_params,
        cu_seqlens,
        cu_seqlens_padded,
    ) = _pack_sequences_for_megatron(
        input_ids,
        seq_lengths,
        pad_individual_seqs_to_multiple_of=pad_to_multiple,
        pad_packed_seq_to=max_seq_len * batch_size if cp_size > 1 else None,
        cp_rank=torch.distributed.get_rank(cp_group),
        cp_size=cp_size,
    )
    assert cu_seqlens_padded is not None

    full_logits = torch.randn(
        batch_size, max_seq_len, vocab_size_total, device=device, dtype=torch.float32
    )

    def make_logits_and_packed_logits():
        logits_local = (
            full_logits[
                :,
                :,
                my_tp_rank * vocab_size_local : (my_tp_rank + 1) * vocab_size_local,
            ]
            .clone()
            .detach()
            .requires_grad_(True)
        )

        total_padded_tokens = int(cu_seqlens_padded[-1].item())
        packed_logits = torch.zeros(
            1, total_padded_tokens // cp_size, vocab_size_local, device=device
        )

        run_seq = 0
        for i in range(batch_size):
            seq_len = int(seq_lengths[i].item())
            padded_seq_len = int(
                (cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]).item()
            )
            tmp = torch.zeros(1, padded_seq_len, vocab_size_local, device=device)
            tmp[:, :seq_len, :] = logits_local[i : i + 1, :seq_len, :]
            packed_logits[
                :,
                run_seq // cp_size : (run_seq + padded_seq_len) // cp_size,
                :,
            ] = _get_tokens_on_this_cp_rank(
                tmp, torch.distributed.get_rank(cp_group), cp_size
            )
            run_seq += padded_seq_len

        return logits_local, packed_logits

    loss_cfg = {
        "reference_policy_kl_penalty": 0.01,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.2,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "token_level_loss": True,
        "ratio_clip_c": None,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "truncated_importance_sampling_ratio": None,
        "sequence_level_importance_ratios": False,
        "force_on_policy_ratio": False,
    }

    valid_toks = int(torch.clamp(seq_lengths - 1, min=0).sum().item())
    global_valid_toks = torch.tensor(valid_toks, dtype=torch.float32, device=device)
    global_valid_seqs = torch.tensor(batch_size, dtype=torch.float32, device=device)

    return {
        "loss_cfg": loss_cfg,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens_padded,
        "data_dict": data_dict,
        "global_valid_seqs": global_valid_seqs,
        "global_valid_toks": global_valid_toks,
        "make_logits_and_packed_logits": make_logits_and_packed_logits,
    }


def _run_compare_sequence_packing_wrappers(rank, world_size, cp_size, tp_size):
    """Compare SequencePackingFusionLossWrapper vs SequencePackingLossWrapper.

    Verifies that the fused wrapper produces identical loss values and
    backward gradients w.r.t. vocab-parallel logits.
    """
    _my_cp_rank, my_tp_rank, cp_group, tp_group = _setup_2d_process_groups(
        rank, world_size, cp_size, tp_size
    )
    tc = _build_test_case(cp_size, tp_size, my_tp_rank, cp_group)
    base_loss_fn = ClippedPGLossFn(tc["loss_cfg"])
    data_dict = tc["data_dict"]

    baseline_wrapper = SequencePackingLossWrapper(
        loss_fn=base_loss_fn,
        prepare_fn=prepare_loss_input,
        cu_seqlens_q=tc["cu_seqlens"],
        cu_seqlens_q_padded=tc["cu_seqlens_padded"],
        vocab_parallel_rank=my_tp_rank,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )

    candidate_wrapper = SequencePackingFusionLossWrapper(
        loss_fn=base_loss_fn,
        prepare_fn=prepare_packed_loss_input,
        cu_seqlens_q=tc["cu_seqlens"],
        cu_seqlens_q_padded=tc["cu_seqlens_padded"],
        vocab_parallel_rank=my_tp_rank,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )

    # Baseline run
    baseline_logits, baseline_packed_logits = tc["make_logits_and_packed_logits"]()
    baseline_loss, _baseline_metrics = baseline_wrapper(
        baseline_packed_logits,
        data_dict,
        tc["global_valid_seqs"],
        tc["global_valid_toks"],
    )
    (baseline_loss / cp_size).backward()
    baseline_grad = baseline_logits.grad.clone()

    # Candidate run (fresh logits, identical values)
    candidate_logits, candidate_packed_logits = tc["make_logits_and_packed_logits"]()
    candidate_loss, _candidate_metrics = candidate_wrapper(
        candidate_packed_logits,
        data_dict,
        tc["global_valid_seqs"],
        tc["global_valid_toks"],
    )
    (candidate_loss / cp_size).backward()
    candidate_grad = candidate_logits.grad.clone()

    # Sanity: gradients must be non-None and non-zero
    assert baseline_grad.abs().sum() > 0, f"baseline grad is all zeros on rank {rank}"
    assert candidate_grad.abs().sum() > 0, f"candidate grad is all zeros on rank {rank}"

    # Forward: loss values must match
    torch.testing.assert_close(
        baseline_loss,
        candidate_loss,
        atol=1e-5,
        rtol=1e-5,
        msg=f"Loss mismatch on rank {rank}",
    )

    # Backward: gradients w.r.t. logits must match
    torch.testing.assert_close(
        baseline_grad,
        candidate_grad,
        atol=1e-5,
        rtol=1e-5,
        msg=f"Gradient mismatch on rank {rank}",
    )


def _run_compare_sequence_packing_wrappers_with_sampling(
    rank, world_size, cp_size, tp_size
):
    """Compare fused vs unfused wrappers with sampling params enabled."""
    _my_cp_rank, my_tp_rank, cp_group, tp_group = _setup_2d_process_groups(
        rank, world_size, cp_size, tp_size
    )
    tc = _build_test_case(cp_size, tp_size, my_tp_rank, cp_group)
    base_loss_fn = ClippedPGLossFn(tc["loss_cfg"])
    data_dict = tc["data_dict"]

    sampling_params = TrainingSamplingParams(top_k=8, top_p=0.9, temperature=1.0)
    prepare_loss_input_wrapped = functools.partial(
        prepare_loss_input, sampling_params=sampling_params
    )
    prepare_packed_loss_input_wrapped = functools.partial(
        prepare_packed_loss_input, sampling_params=sampling_params
    )

    baseline_wrapper = SequencePackingLossWrapper(
        loss_fn=base_loss_fn,
        prepare_fn=prepare_loss_input_wrapped,
        cu_seqlens_q=tc["cu_seqlens"],
        cu_seqlens_q_padded=tc["cu_seqlens_padded"],
        vocab_parallel_rank=my_tp_rank,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )

    candidate_wrapper = SequencePackingFusionLossWrapper(
        loss_fn=base_loss_fn,
        prepare_fn=prepare_packed_loss_input_wrapped,
        cu_seqlens_q=tc["cu_seqlens"],
        cu_seqlens_q_padded=tc["cu_seqlens_padded"],
        vocab_parallel_rank=my_tp_rank,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
    )

    # Baseline run
    baseline_logits, baseline_packed_logits = tc["make_logits_and_packed_logits"]()
    baseline_loss, baseline_metrics = baseline_wrapper(
        baseline_packed_logits,
        data_dict,
        tc["global_valid_seqs"],
        tc["global_valid_toks"],
    )
    (baseline_loss / cp_size).backward()
    baseline_grad = baseline_logits.grad.clone()

    # Candidate run (fresh logits, identical values)
    candidate_logits, candidate_packed_logits = tc["make_logits_and_packed_logits"]()
    candidate_loss, candidate_metrics = candidate_wrapper(
        candidate_packed_logits,
        data_dict,
        tc["global_valid_seqs"],
        tc["global_valid_toks"],
    )
    (candidate_loss / cp_size).backward()
    candidate_grad = candidate_logits.grad.clone()

    # Sanity: gradients must be non-None and non-zero
    assert baseline_grad.abs().sum() > 0, f"baseline grad is all zeros on rank {rank}"
    assert candidate_grad.abs().sum() > 0, f"candidate grad is all zeros on rank {rank}"

    # Forward: loss values must match
    torch.testing.assert_close(
        baseline_loss,
        candidate_loss,
        atol=1e-5,
        rtol=1e-5,
        msg=f"Loss mismatch with sampling params on rank {rank}",
    )

    # Metrics parity under sampling params
    assert set(baseline_metrics.keys()) == set(candidate_metrics.keys())
    for k in baseline_metrics:
        torch.testing.assert_close(
            torch.as_tensor(baseline_metrics[k], device="cuda"),
            torch.as_tensor(candidate_metrics[k], device="cuda"),
            atol=1e-5,
            rtol=1e-5,
            msg=f"Metric mismatch for key={k} on rank {rank}",
        )

    # Backward: gradients w.r.t. logits must match
    torch.testing.assert_close(
        baseline_grad,
        candidate_grad,
        atol=1e-5,
        rtol=1e-5,
        msg=f"Gradient mismatch with sampling params on rank {rank}",
    )


@pytest.mark.parametrize(
    "cp_tp",
    [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 4),
        (4, 2),
    ],
    ids=lambda cp_tp: f"cp{cp_tp[0]}_tp{cp_tp[1]}",
)
def test_sequence_packing_fusion_vs_baseline(distributed_test_runner, cp_tp):
    """Compare SequencePackingFusionLossWrapper vs SequencePackingLossWrapper.

    Verifies that the fused wrapper produces identical:
      - loss values
      - backward gradients w.r.t. vocab-parallel logits
    for different CP and TP configurations.
    """
    cp_size, tp_size = cp_tp
    world_size = cp_size * tp_size

    test_fn = functools.partial(
        _run_compare_sequence_packing_wrappers,
        cp_size=cp_size,
        tp_size=tp_size,
    )
    distributed_test_runner(test_fn, world_size=world_size)


@pytest.mark.parametrize(
    "cp_tp",
    [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ],
    ids=lambda cp_tp: f"sampling_cp{cp_tp[0]}_tp{cp_tp[1]}",
)
def test_sequence_packing_fusion_vs_baseline_with_sampling_params(
    distributed_test_runner, cp_tp
):
    """Compare fused vs unfused wrappers with top-k/top-p sampling params."""
    cp_size, tp_size = cp_tp
    world_size = cp_size * tp_size

    test_fn = functools.partial(
        _run_compare_sequence_packing_wrappers_with_sampling,
        cp_size=cp_size,
        tp_size=tp_size,
    )
    distributed_test_runner(test_fn, world_size=world_size)
